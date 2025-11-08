from collections import defaultdict

import numpy as np
import timm
import timm.models.vision_transformer as timm_vit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from loguru import logger
from transformers import AutoModel

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class CrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Stack multiple cross-attention layers
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

        # Feed-forward networks for each layer
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ]
        )

        self.ffn_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_layers)])

    def forward(self, query, key, value):
        # query: B, T_q, D
        # key, value: B, T_kv, D

        x = query
        all_attn_weights = []

        for i in range(self.num_layers):
            # Cross attention
            residual = x
            x = self.layer_norms[i](x)
            attn_output, attn_weights = self.layers[i](x, key, value)
            x = residual + attn_output

            # Feed forward
            residual = x
            x = self.ffn_norms[i](x)
            ffn_output = self.ffns[i](x)
            x = residual + ffn_output

            all_attn_weights.append(attn_weights)

        # Return output and attention weights from the last layer
        return x, all_attn_weights[-1]


TOKEN_IDS = {
    "camera0_rgb": "camera/ft",
    "tacthru_l_rgb": "sensor/ft",
    "tacthru_r_rgb": "sensor/ft",
}


def _get_model(model, shape: tuple, **kwargs) -> nn.Module:
    if isinstance(model, nn.Module):
        pass
    elif model is None:
        model = torch.nn.Identity()
        model.num_features = np.prod(shape)
    elif isinstance(model, str):
        if model == "mlp":
            n_channel = shape[-1]

            model = nn.Sequential(
                nn.Linear(n_channel, min(768, n_channel * 8)),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(min(768, n_channel * 8), min(768, n_channel * 64)),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(min(768, n_channel * 64), min(768, n_channel * 256)),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(min(768, n_channel * 256), 768),
            )
            model.num_features = 768
        elif model.startswith("hf::"):
            model = model.split("::")[-1]
            model = AutoModel.from_pretrained(model, **kwargs)
        else:  # if model.startswith("timm::"):
            model = model.split("::")[-1]
            use_group_norm = kwargs.pop("use_group_norm", False)
            kwargs["img_size"] = shape[1:]
            model = timm.create_model(model_name=model, in_chans=shape[0], **kwargs)
            patch_size = model.patch_embed.patch_size[0]
            # if use_group_norm:
            #     model = replace_submodules(
            #         root_module=model,
            #         predicate=lambda x: isinstance(x, nn.BatchNorm2d),
            #         func=lambda x: nn.GroupNorm(
            #             num_groups=(x.num_features // patch_size) if (x.num_features % patch_size == 0) else (x.num_features // 8),
            #             num_channels=x.num_features,
            #         ),
            #     )

    return model


def _get_transforms(key: str, shape: tuple) -> tuple[nn.Module, nn.Module]:
    if key in ["camera0_rgb"]:
        train_transforms = [
            torchvision.transforms.RandomResizedCrop(
                size=shape[1], scale=(0.85, 1.15), ratio=(0.9, 1.1), antialias=True
            ),
            torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.2, hue=0.1),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
        eval_transforms = [
            torchvision.transforms.Resize(size=shape[1], antialias=True),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
        return torch.nn.Sequential(*train_transforms), torch.nn.Sequential(*eval_transforms)
    if key in ["tacthru_l_rgb", "tacthru_r_rgb"]:
        train_transforms = [
            torchvision.transforms.RandomResizedCrop(
                size=shape[1], scale=(0.85, 1.15), ratio=(0.9, 1.1), antialias=True
            ),
            torchvision.transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.4, hue=0.1),
            torchvision.transforms.Normalize(mean=[0.4434, 0.4010, 0.4001], std=[0.2883, 0.2408, 0.2527]),
            torchvision.transforms.RandomGrayscale(p=0.25),
        ]
        eval_transforms = [
            torchvision.transforms.Resize(size=shape[1], antialias=True),
            torchvision.transforms.Normalize(mean=[0.4434, 0.4010, 0.4001], std=[0.2883, 0.2408, 0.2527]),
        ]
        return torch.nn.Sequential(*train_transforms), torch.nn.Sequential(*eval_transforms)
    return torch.nn.Identity(), torch.nn.Identity()


class TransformerTacThruObsEncoder(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        tac_lr_shared: bool,
        n_emb: int = 768,
        feature_fusion: str = None,
    ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        key = "camera0_rgb"
        self.sensor_l_keys = list()
        self.sensor_r_keys = list()
        self.proprio_keys = list()
        self.key_model_map = nn.ModuleDict()
        self.key_transform_map = nn.ModuleDict()
        self.key_eval_transform_map = nn.ModuleDict()
        self.key_projection_map = nn.ModuleDict()
        self.key_model_lr_map = dict()
        self.key_shape_map = dict()
        self.key_dropout_map = dict()

        self.num_effective_tokens = dict()

        obs_shape_meta = shape_meta["obs"]

        # Setup encoders
        sensor_type_to_key: dict[str, str] = {}
        for key, attr in obs_shape_meta.items():
            # Skipping ignored models
            if attr.get("ignore_by_policy", False):
                continue

            # Metadata
            shape = self.key_shape_map[key] = tuple(attr["shape"])
            type = attr.get("type", "low_dim")
            model_name = obs_shape_meta[key].get("encoder", None)
            model_pretrained = obs_shape_meta[key].get("encoder_pretrained", False)
            model_frozen: bool = obs_shape_meta[key].get("encoder_frozen", False)
            if (dropout_p := obs_shape_meta[key].get("dropout")) is not None:
                self.key_dropout_map[key] = dropout_p
            if key.startswith("tacthru_l"):
                self.sensor_l_keys.append(key)
            elif key.startswith("tacthru_r"):
                self.sensor_r_keys.append(key)
            elif type == "low_dim":
                self.proprio_keys.append(key)
            elif type == "rgb":
                pass
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

            # Sharing existing models
            if type in sensor_type_to_key and tac_lr_shared:
                shared_obs_key = sensor_type_to_key[type]
                self.key_model_map[key] = self.key_model_map[shared_obs_key]
                self.key_transform_map[key] = self.key_transform_map[shared_obs_key]
                self.key_eval_transform_map[key] = self.key_eval_transform_map[shared_obs_key]
                self.key_projection_map[key] = self.key_projection_map[shared_obs_key]
                self.key_shape_map[key] = self.key_shape_map[shared_obs_key]
                self.key_dropout_map[key] = self.key_dropout_map[shared_obs_key]
                continue

            # Setup encoder model and its output projection to embeddings
            kwargs = obs_shape_meta[key].get("encoder_args", {})
            kwargs["pretrained"] = model_pretrained
            encoder = obs_shape_meta[key].get("encoder")

            model = self.key_model_map[key] = _get_model(encoder, shape, **kwargs)
            if (ft_id := TOKEN_IDS.get(key)) is not None:
                if hasattr(model, "num_patches"):
                    self.num_effective_tokens[ft_id] = model.num_patches
                elif isinstance(encoder, str) and "dino" in encoder:
                    self.num_effective_tokens[ft_id] = {
                        "camera0_rgb": 256,
                        "tacthru_l_rgb": int((shape[1] // 14) ** 2),
                        "tacthru_r_rgb": int((shape[1] // 14) ** 2),
                    }[key]
                elif isinstance(encoder, str) and "clip" in encoder:
                    self.num_effective_tokens[ft_id] = 196
                elif hasattr(encoder.patch_embed, "num_patches"):
                    self.num_effective_tokens[ft_id] = encoder.patch_embed.num_patches
                elif isinstance(encoder, timm_vit.VisionTransformer):
                    self.num_effective_tokens[ft_id] = 196

            assert hasattr(model, "num_features"), f"Model {model_name} does not have num_features attribute"
            if isinstance(model_pretrained, str):
                state_dict = torch.load(model_pretrained, map_location="cpu", weights_only=True)
                model.load_state_dict(state_dict)

            if obs_shape_meta[key].get("append_adapter", False) or model.num_features != n_emb:
                self.key_projection_map[key] = nn.Sequential(
                    nn.Linear(in_features=model.num_features, out_features=n_emb),
                    nn.ReLU(),
                    nn.Linear(in_features=n_emb, out_features=n_emb),
                )
            else:
                self.key_projection_map[key] = nn.Identity()

            # Setup encoder model training
            model_lr: float = obs_shape_meta[key].get("encoder_lr", 1e-3)
            if model_pretrained is not False:
                model_lr *= 0.1
            if model_frozen:
                assert model_pretrained is not False, "Frozen model must be pretrained"
                model_lr = 0.0
                for param in model.parameters():
                    param.requires_grad = False
            self.key_model_lr_map[key] = model_lr

            # Setup training and evaluation input transforms
            self.key_transform_map[key], self.key_eval_transform_map[key] = _get_transforms(key, shape)

            logger.info(
                f"Encoder for [{key}]: {model_name}, shape: {shape}, pretrained: {model_pretrained}, lr: {model_lr}, frozen: {model_frozen}, dropout: {dropout_p}"
            )

        # Setup cross-modal fusion
        cross_attention_layers, cross_attention_dropout = 2, 0.1
        self.feature_fusion = feature_fusion
        if self.feature_fusion is not None:
            # assert self.feature_fusion in [None, "cross", "wristcam_merge", "sensor_merge"]
            # if self.feature_fusion in ["cross", "wristcam_merge", "sensor_merge"]:
            self.cross_attention_w = CrossAttention(
                n_emb,
                num_layers=cross_attention_layers,
                dropout=cross_attention_dropout,
            )
            self.cross_attention_s = CrossAttention(
                n_emb,
                num_layers=cross_attention_layers,
                dropout=cross_attention_dropout,
            )

        self.sensor_l_keys = sorted(self.sensor_l_keys)
        self.sensor_r_keys = sorted(self.sensor_r_keys)
        self.proprio_keys = sorted(self.proprio_keys)
        self.all_keys = ["camera0_rgb"] + self.sensor_l_keys + self.sensor_r_keys + self.proprio_keys
        self.n_emb = n_emb
        self.shape_meta = shape_meta
        self.key_shape_map = self.key_shape_map

        self.token_embds = nn.Embedding(len(self.all_keys), n_emb)

    def apply_feature_fusion(self, tokens: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[str]]:
        """
        Apply feature fusion between wristcam and sensor features
        Args:
            wristcam_features: B, T_w, D
            sensor_features: B, T_s, D
        Returns:
            fused_features: B, T_fused, D
            token_ids: list of token identifiers
        """
        if self.feature_fusion is None:
            # Remove register tokens
            for token_id in ["camera/ft", "sensor/ft"]:
                if token_id in tokens and (n_effective_tokens := self.num_effective_tokens.get(token_id)) is not None:
                    tokens[token_id] = tokens[token_id][..., -n_effective_tokens:, :]
            return tokens

        wristcam_tokens = tokens.pop("camera/ft")
        sensor_tokens = tokens.pop("sensor/ft", None)

        if self.feature_fusion == "cross":
            # Cross attention then concatenate tokens
            w2s, w2s_wt = self.cross_attention_w(wristcam_tokens, sensor_tokens, sensor_tokens)
            s2w, s2w_wt = self.cross_attention_s(sensor_tokens, wristcam_tokens, wristcam_tokens)
            tokens["sensor/cross_camera"] = w2s
            tokens["camera/cross_sensor"] = s2w

        elif self.feature_fusion == "wristcam_centric":
            all_tokens = torch.cat([wristcam_tokens, sensor_tokens], dim=1)
            s2a, s2a_wt = self.cross_attention_s(wristcam_tokens, all_tokens, all_tokens)

            tokens["camera/cross_all"] = s2a

        else:
            raise ValueError(f"Unknown feature_fusion method: {self.feature_fusion}")

        return tokens

    def forward(
        self,
        obs_dict,
        as_dict: bool = False,
        return_token_id: bool = False,
        no_randomize: bool = False,
    ):
        tokens = defaultdict(list)
        no_randomize = no_randomize or (not self.training)

        for i_key, key in enumerate(self.all_keys):
            x = obs_dict[key]
            B, T, *_ = x.shape
            transforms = self.key_eval_transform_map[key] if no_randomize else self.key_transform_map[key]
            model = self.key_model_map[key]
            projection = self.key_projection_map[key]

            x = transforms(x.reshape(B * T, *x.shape[2:]))
            x = projection(model(x))
            x = x.reshape(B, -1, self.n_emb)  # B, T, D

            # Add token embedding to the features
            embd = self.token_embds(torch.tensor(i_key, device=x.device)).unsqueeze(0).unsqueeze(0)  # 1, 1, D
            x = x + embd

            if (dropout_p := self.key_dropout_map.get(key, None)) is not None and not no_randomize:
                do_dropout = torch.rand(x.shape[0], device=x.device) < dropout_p
                x[do_dropout] = 0.0
            token_id = TOKEN_IDS.get(key, "low_dim")
            tokens[token_id].append(x)

        for token_id in tokens:
            tokens[token_id] = torch.cat(tokens[token_id], dim=1)

        tokens = self.apply_feature_fusion(tokens)

        # Encoder results
        if as_dict:
            return tokens
        else:
            all_tokens, token_ids = [], []
            for k in sorted(tokens.keys()):
                all_tokens.append(tokens[k])
                token_ids.extend([k] * tokens[k].shape[1])
            tokens = torch.cat(all_tokens, dim=1)  # B, T, D
            if return_token_id:
                return tokens, token_ids
            return tokens

    @torch.no_grad()
    def example_output(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            this_obs = torch.zeros((1, attr["horizon"]) + shape, dtype=self.dtype, device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict, as_dict=False, return_token_id=False)
        return example_output

    @torch.no_grad()
    def output_shape(self):
        example_output = self.example_output()
        return example_output.shape
