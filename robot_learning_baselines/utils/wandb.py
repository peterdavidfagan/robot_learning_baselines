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


def visualize_dataset(cfg, raw_batch):
    """
    Visualize raw dataset in wandb table.
    """
    batch_size = cfg.training.batch_size
    table_config = dict(cfg.wandb.dataset_visualization.columns)
    column_names = list(table_config.keys())
    
    data = []
    for i in range(batch_size):
        sample = []
        for name, value in table_config.items():
            section, field = name.split("/")
            if value == "image":
                image_data = raw_batch[section][field][i]
                sample.append(wandb.Image(e.rearrange(image_data, "seq h w c -> h (seq w) c")))
            elif value == "text":
                sample.append(raw_batch[section][field][i].decode())
            else:
                continue
        data.append(sample)
        
    table = wandb.Table(data=data, columns=column_names)    
    wandb.log({"dataset": table})
