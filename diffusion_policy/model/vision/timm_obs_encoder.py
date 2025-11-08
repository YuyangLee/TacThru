import copy
import logging
import math

import hydra
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from loguru import logger
from timm.models.vision_transformer import VisionTransformer

from diffusion_policy.common.pytorch_util import replace_submodules
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


class TimmTacThruObsEncoder(ModuleAttrMixin):
    def __init__(
        self,
        shape_meta: dict,
        tac_lr_shared: bool,
        global_pool: str,
        transforms: list,
        use_group_norm: bool = False,  # replace BatchNorm with GroupNorm
        share_rgb_model: bool = False,  # use single rgb model for all rgb inputs
        imagenet_norm: bool = False,  # renormalize rgb input with imagenet normalization, assuming input in [0,1]
        feature_aggregation: str = "spatial_embedding",
        downsample_ratio: int = 32,
        position_encording: str = "learnable",
        mock_sensor_input: bool = False,
        sensor_dropout: float = 0.0,
    ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        wristcam_keys = list()
        sensor_l_keys = list()
        sensor_r_keys = list()
        low_dim_keys = list()

        obs_shape_meta = shape_meta["obs"]

        self.key_model_map = nn.ModuleDict()
        self.key_transform_map = nn.ModuleDict()
        self.key_model_lr_map = dict()
        self.key_shape_map = dict()

        self.sensor_dropout = nn.Identity()
        if sensor_dropout > 0.0:
            logger.info(f"Using tacthru dropout: {sensor_dropout}")
            self.sensor_dropout = nn.Dropout(sensor_dropout)

        assert global_pool == ""

        # GoPro vision model
        model_info = obs_shape_meta["camera0_rgb"]["encoder"]
        pretrained = obs_shape_meta["camera0_rgb"]["encoder_pretrained"]
        model = timm.create_model(
            model_name=model_info,
            pretrained=pretrained,
            global_pool=global_pool,  # means no pooling
            num_classes=0,  # remove classification layer
        )

        feature_dim = None

        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=((x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8)),
                    num_channels=x.num_features,
                ),
            )

        obs_shape_meta = shape_meta["obs"]
        image_shape = obs_shape_meta["camera0_rgb"]["shape"][1:]

        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == "RandomCrop"
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True),
            ] + transforms[1:]
        wristcam_transforms = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        ## GoPro: Feature Aggregation
        feature_map_shape = [x // downsample_ratio for x in image_shape]
        self.feature_aggregation = feature_aggregation
        if self.feature_aggregation == "all_tokens":
            # Use all tokens from ViT
            pass
        elif self.feature_aggregation is not None:
            logger.warning(f"vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!")
            self.feature_aggregation = None

        if self.feature_aggregation == "attention_pool_2d":
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim,
            )

        tacthru_l_embd_length, tacthru_r_embd_length = 0, 0
        tacthru_models = {}
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            key_type = attr.get("type", "low_dim")
            self.key_shape_map[key] = shape
            model_info = obs_shape_meta[key].get("encoder", None)
            model_pretrained = obs_shape_meta[key].get("encoder_pretrained", False)
            model_lr = obs_shape_meta[key].get("encoder_lr", 1e-3)
            model_frozen = obs_shape_meta[key].get("encoder_frozen", False)
            if model_pretrained:
                model_lr = float(model_lr)
                model_lr *= 0.1
            if key_type == "rgb":
                wristcam_keys.append(key)
                this_model = model
                self.key_model_map[key] = this_model
                this_transform = wristcam_transforms
                self.key_transform_map[key] = this_transform
                self.wristcam_model_name = model_info

            elif key_type == "low_dim":
                if not attr.get("ignore_by_policy", False):
                    low_dim_keys.append(key)

            # The rest are Sensor keys
            else:
                assert key_type.startswith("tac_")

                if key_type in tacthru_models and tac_lr_shared:
                    this_model, this_transform = tacthru_models[type]
                    self.key_model_map[key] = this_model
                    self.key_transform_map[key] = this_transform
                    # In shared mode, the lr will not be logged so that the params will only be added once to the optimizer
                else:
                    if key_type in [
                        "tac_rgb",
                        "tac_prox",
                        "tac_depth",
                        "tac_shear",
                        "tac_normal",
                    ]:
                        self.key_model_lr_map[key] = model_lr
                        in_chans = obs_shape_meta[key]["shape"][0]
                        if isinstance(model_info, str):
                            this_model = timm.create_model(
                                model_name=model_info,
                                pretrained=model_pretrained,
                                in_chans=in_chans,
                                num_classes=0,
                                global_pool=global_pool,
                            )
                        elif isinstance(model_info, nn.Module):
                            this_model = model_info  # Already a model
                            if model_pretrained:
                                logger.info(f"Loading pretrained model from {model_pretrained} for {key}")
                                state_dict = torch.load(model_pretrained, map_location="cpu")
                                this_model.load_state_dict(state_dict)
                            this_model.num_features = this_model.embed_dim
                        self.key_model_map[key] = this_model
                        self.key_transform_map[key] = torchvision.transforms.Resize(
                            size=obs_shape_meta[key]["shape"][1:], antialias=True
                        )
                        # logger.debug(f"Added model {model_info} for {key}")

                    elif key_type == "tac_sparse_shear":
                        this_model = torch.nn.Sequential(
                            torch.nn.Flatten(1),  # 64 x 2 -> 128,
                            torch.nn.Linear(128, 128),
                            torch.nn.ELU(),
                            torch.nn.Linear(128, 32),
                        )
                        this_model.num_features = 32
                        self.key_model_map[key] = this_model
                        self.key_transform_map[key] = torch.nn.Identity()
                        logger.debug(f"Added Sparse shear sensing MLP for {key}")

                    else:
                        raise RuntimeError(f"Unsupported obs type: {key_type}")

                if tac_lr_shared:
                    tacthru_models[key_type] = (
                        self.key_model_map[key],
                        self.key_transform_map[key],
                    )

                if key.startswith("tacthru_l"):
                    sensor_l_keys.append(key)
                    tacthru_l_embd_length += this_model.num_features
                elif key.startswith("tacthru_r"):
                    sensor_r_keys.append(key)
                    tacthru_r_embd_length += this_model.num_features
                else:
                    raise RuntimeError(f"Unsupported key for TacThru: {key}")

            if model_frozen:
                assert model_pretrained
                for param in this_model.parameters():
                    param.requires_grad = False

            if isinstance(this_model, VisionTransformer) and hasattr(this_model, "default_cfg"):
                this_model.default_cfg["num_classes"] = 0
                this_model.default_cfg["classifier"] = None
                this_model.head = nn.Identity()

        self.shape_meta = shape_meta

        self.model_name = model_info
        self.wristcam_keys = sorted(wristcam_keys)
        self.sensor_l_keys = sorted(sensor_l_keys)
        self.sensor_r_keys = sorted(sensor_r_keys)
        self.low_dim_keys = sorted(low_dim_keys)

        logger.info(f"GoPro keys:    {self.wristcam_keys}")
        logger.info(f"L Sensor keys: {self.sensor_l_keys}")
        logger.info(f"R Sensor keys: {self.sensor_r_keys}")
        logger.info(f"Low-Dim keys:  {self.low_dim_keys}")

        self.mock_sensor_input = mock_sensor_input
        if self.mock_sensor_input:
            logger.warning("Mocking sensor input. All sensor inputs will be zero !!")

    def _aggregate_feature(self, feature, obs_type, feature_aggregation: str = None):
        feature_aggregation = feature_aggregation or self.feature_aggregation

        # logger.info(f"aggregate_feature: obs_type={obs_type}, feature.shape={feature.shape}")
        if obs_type == "rgb" and self.wristcam_model_name.startswith("vit"):
            assert feature_aggregation is None  # vit uses the CLS token
            return feature[:, 0, :]
        elif obs_type.endswith("_rgb") and len(feature.shape) == 3:
            return feature[:, 0, :]

        # resnet
        assert len(feature.shape) == 4
        if feature_aggregation == "attention_pool_2d":
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2)  # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2)  # B, 7*7, 512

        if feature_aggregation == "avg":
            return torch.mean(feature, dim=[1])
        elif feature_aggregation == "max":
            return torch.amax(feature, dim=[1])
        elif feature_aggregation == "soft_attention":
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1)
        elif feature_aggregation == "spatial_embedding":
            return torch.mean(feature * self.spatial_embedding, dim=1)
        elif feature_aggregation == "transformer":
            zero_feature = torch.zeros(feature.shape[0], 1, feature.shape[-1], device=feature.device)
            if self.position_embedding.device != feature.device:
                self.position_embedding = self.position_embedding.to(feature.device)
            feature_with_pos_embedding = torch.concat([zero_feature, feature], dim=1) + self.position_embedding
            feature_output = self.aggregation_transformer(feature_with_pos_embedding)
            return feature_output[:, 0]
        else:
            assert feature_aggregation is None
            return feature

    def forward(self, obs_dict):
        features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]

        if self.mock_sensor_input:
            for key in self.sensor_l_keys + self.sensor_r_keys:
                obs_dict[key] = torch.zeros_like(obs_dict[key])

        # process rgb input
        for key in self.wristcam_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]
            img = img.reshape(B * T, *img.shape[2:])
            img = self.key_transform_map[key](img)
            raw_feature = self.key_model_map[key](img)
            feature = self._aggregate_feature(raw_feature, "rgb")
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            features.append(feature.reshape(B, -1))

        # tacthru_embds = [
        #     self.left_tacthru_embeddings(self.left_right_embd_idxs),
        #     self.right_tacthru_embeddings(self.left_right_embd_idxs),
        # ]
        for i_side, keys in enumerate([self.sensor_l_keys, self.sensor_r_keys]):
            if len(keys) == 0:
                continue
            sensor_fts = []
            for key in keys:
                obs_data = obs_dict[key]  # [B,T,dim,H,W]
                obs_data = obs_data / 255.0  # FIXME: After re-scaling dataset data to [0,1], remove this line
                B, T = obs_data.shape[:2]
                assert B == batch_size
                obs_data = obs_data.reshape(B * T, *obs_data.shape[2:])  # [B*T,dim,H,W]
                feature = self.key_model_map[key](self.key_transform_map[key](obs_data))
                feature = self._aggregate_feature(feature, key, "cls")  # if len(feature.shape) == 4 else feature
                feature = self.sensor_dropout(feature)
                sensor_fts.append(feature)  # B x T x N_ft_i

            sensor_fts = torch.concat(sensor_fts, dim=-1)  # + tacthru_embds[i_side]  # Feature shape: B x T x N_ft
            features.append(sensor_fts.reshape(B, -1))

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))

        # concatenate all features
        result = torch.cat(features, dim=-1)

        return result

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr["shape"])
            this_obs = torch.zeros((1, attr["horizon"]) + shape, dtype=self.dtype, device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1

        return example_output.shape
