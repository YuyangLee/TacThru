import copy
import os
import pathlib
import pickle
import random
from hashlib import md5

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
import tqdm
import wandb
from accelerate import Accelerator
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.dataset.base_dataset import BaseDataset, BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.val_utils import run_validation, stat_last_train_batch

sns.set()

OmegaConf.register_new_resolver("eval", eval, replace=True)


class ConsistentDropout(torch.nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)


LOGGER_SAVE_FILES = [
    "train.py",
    "diffusion_policy/workspace/train_diffusion_workspace.py",
    "diffusion_policy/workspace/base_workspace.py",
    "diffusion_policy/workspace/val_utils.py",
    "diffusion_policy/model/vision/transformer_obs_encoder.py",
    "diffusion_policy/model/diffusion/transformer_for_action_diffusion.py",
    "diffusion_policy/model/diffusion/transformers_ext.py",
]


class TrainDiffusionImageWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]
    exclude_keys = ["dataset", "train_dataloader", "val_dataset", "val_dataloader"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: BaseImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: BaseImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        model_size_mb = sum(p.numel() for p in self.model.parameters()) / 1e6
        encoder_size_mb = sum(p.numel() for p in self.model.obs_encoder.parameters()) / 1e6
        policy_size_mb = sum(p.numel() for p in self.model.model.parameters()) / 1e6
        logger.info(
            f"Model size: {model_size_mb:.2f}M (Encoder: {encoder_size_mb:.2f}M, Policy: {policy_size_mb:.2f}M)"
        )

        # gather params for obs encoder
        obs_encoders_params, obs_encoders_added_ids, extra_params = (
            list(),
            list(),
            list(),
        )
        for model_key, lr in self.model.obs_encoder.key_model_lr_map.items():
            model_params = []
            model_size = 0.0
            for p in self.model.obs_encoder.key_model_map[model_key].parameters():
                if p.requires_grad:
                    model_params.append(p)
                    obs_encoders_added_ids.append(id(p))
                    model_size += p.numel() * p.element_size() / 1e3

            if len(model_params) > 0:
                obs_encoders_params.append({"params": model_params, "lr": lr})
            logger.info(f"+ Encoder for {model_key}, trainable size: {model_size:.2f}K")
        for param in self.model.parameters():
            if param.requires_grad and id(param) not in obs_encoders_added_ids:
                extra_params.append(param)

        param_groups = [{"params": extra_params}] + obs_encoders_params

        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop("_target_")
        self.optimizer = torch.optim.AdamW(params=param_groups, **optimizer_cfg)

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys += ["optimizer"]

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        if cfg.training.debug:
            log_with = None
            init_kwargs = {}
        else:
            log_with = "wandb"

            wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
            wandb_cfg.pop("project")
            init_kwargs = {"wandb": wandb_cfg}

        accelerator = Accelerator(log_with=log_with)
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs=init_kwargs,
        )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.logger.info(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        if wandb.run is not None:
            for filepath in LOGGER_SAVE_FILES:
                wandb.save(filepath, policy="now")
                logger.info(f"Saved {filepath} to wandb.")

        # configure dataset
        self.dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
        self.train_dataloader = DataLoader(self.dataset, **cfg.dataloader)
        assert isinstance(self.dataset, BaseImageDataset) or isinstance(self.dataset, BaseDataset)
        self.val_dataset = self.dataset.get_validation_dataset()
        self.val_dataloader = DataLoader(self.val_dataset, **cfg.val_dataloader)
        assert isinstance(self.val_dataset, BaseImageDataset) or isinstance(self.val_dataset, BaseDataset)
        logger.info(f"Training data: {len(self.dataset)}, ({len(self.train_dataloader)} batches)")
        logger.info(f"Validation data: {len(self.val_dataset)}, ({len(self.val_dataloader)} batches)")

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, "normalizer.pkl")
        if accelerator.is_main_process:
            dataset_path = cfg.task.dataset.dataset_path
            all_keys = self.dataset.all_keys
            logger.info(f"All keys loaded in the dataset: {all_keys}")
            hasher = md5()
            with open(dataset_path, "rb") as f:
                hasher.update(f.read())
            keys_bytes = str(sorted(self.dataset.all_keys)).encode("utf-8")
            hasher.update(keys_bytes)
            dataset_file_md5 = hasher.hexdigest()[:7]
            ds_normalizer_path = os.path.join(os.path.dirname(dataset_path), f"normalizer_{dataset_file_md5}.pkl")
            if os.path.exists(ds_normalizer_path):
                normalizer = pickle.load(open(ds_normalizer_path, "rb"))
                logger.info(f"Loaded cached normalizer from {ds_normalizer_path}")
            else:
                normalizer = self.dataset.get_normalizer()
                pickle.dump(normalizer, open(ds_normalizer_path, "wb"))
                logger.info(f"Cached new normalizer to {ds_normalizer_path}")
            pickle.dump(normalizer, open(os.path.join(self.output_dir, "normalizer.pkl"), "wb"))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, "rb"))

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(self.train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
        )

        # device transfer
        self.model.to(self.device)
        if self.ema_model is not None:
            self.ema_model.to(self.device)
        optimizer_to(self.optimizer, self.device)

        # accelerator
        (
            self.train_dataloader,
            self.val_dataloader,
            self.model,
            self.optimizer,
            lr_scheduler,
        ) = accelerator.prepare(
            self.train_dataloader,
            self.val_dataloader,
            self.model,
            self.optimizer,
            lr_scheduler,
        )

        # save batch for sampling
        train_sampling_batch = None

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")

        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()

                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(
                    self.train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                        train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(self.train_dataloader) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (
                            cfg.training.max_train_steps - 1
                        ):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run validation
                if (
                    (self.epoch == 0 or (self.epoch % cfg.training.val_every) == cfg.training.val_every - 1)
                    and len(self.val_dataloader) > 0
                    and accelerator.is_main_process
                ):
                    val_step_log = run_validation(
                        self.model,
                        policy,
                        self.val_dataset,
                        self.val_dataloader,
                        self.epoch,
                        output_dir=self.output_dir,
                        debug=cfg.training.debug,
                        # enable_ctrl=cfg.policy.get("enable_ctrl", False),
                    )
                    step_log.update(val_step_log)

                # run diffusion sampling on a training batch
                if (
                    self.epoch == 0 or (self.epoch % cfg.training.sample_every) == cfg.training.sample_every - 1
                ) and accelerator.is_main_process:
                    sample_log = stat_last_train_batch(policy, train_sampling_batch, self.epoch, self.output_dir)
                    step_log.update(sample_log)

                # checkpoint
                if (
                    self.epoch == 0 or (self.epoch % cfg.training.checkpoint_every) == 0
                ) and accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace("/", "_")
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp
                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainDiffusionImageWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
