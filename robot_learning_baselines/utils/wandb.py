"""Utility functions for logging to Weights & Biases."""
import os
import wandb
import omegaconf

import jax
import jax.numpy as jnp
import einops as e

from orbax import checkpoint


def init_wandb(cfg, resume=False):
    """Initialise Weights & Biases run."""
    
    # initialise wandb config
    config = omegaconf.OmegaConf.to_container(cfg, resolve=False)
    wandb_config = config["wandb"]
    
    if resume:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            id=cfg.wandb.resume_run.id,
            resume="allow",
        )
    else:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=config,
            tags=cfg.wandb.tags,
            notes=cfg.wandb.notes,
        )
        wandb.run.name = cfg.wandb.experiment_name

    # define metrics

    # create tables for logging

    # define model checkpoint directory


