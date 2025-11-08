import sys

sys.path.append(".")

import pathlib

import hydra
from loguru import logger
from omegaconf import OmegaConf

from diffusion_policy.policy.diffusion_transformer_timm_policy import DiffusionTransformerTimmPolicy  # fmt: skip
from diffusion_policy.policy.diffusion_unet_timm_policy import DiffusionUnetTimmPolicy  # fmt: skip
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("oc.dir", lambda x: pathlib.Path(x).parent)


@hydra.main(version_base=None, config_path="../cfg/train", config_name="train_tf")
def main(cfg: OmegaConf):
    # Active TacThru observations
    obs_shape_meta = dict(cfg.task.shape_meta.obs)
    tac_active_keys = cfg.tac_active_keys
    for key in list(obs_shape_meta.keys()):
        if key.startswith("tac") and key not in tac_active_keys:
            del obs_shape_meta[key]
    cfg.task.shape_meta.obs = OmegaConf.create(obs_shape_meta)
    logger.info(f"Active observations: {list(cfg.task.shape_meta.obs.keys())}")

    if cfg.training.debug:
        cfg.training.num_epochs = 2
        cfg.training.max_train_steps = 3
        cfg.training.max_val_steps = 3
        cfg.training.rollout_every = 1
        cfg.training.checkpoint_every = 1
        cfg.training.val_every = 1
        cfg.training.sample_every = 1

    # Need to do all the modifications before resolving it
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)

    if cfg.get("load_ckpt"):
        workspace.load_checkpoint(cfg.load_ckpt)

    workspace.run()


if __name__ == "__main__":
    main()
