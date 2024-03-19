"""Utility functions for logging to Weights & Biases."""
import os
import wandb
import omegaconf

from PIL import Image, ImageDraw
from matplotlib import cm

import jax
import jax.numpy as jnp
import numpy as np
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
    Visualize oxe raw dataset in wandb table.
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


def visualize_transporter_dataset(cfg, raw_batch):
    """
    Visualize transporter dataset.
    """
    batch_size = cfg.training.transporter_pick.batch_size
    table_config = dict(cfg.wandb.dataset_visualization.columns)
    column_names = list(table_config.keys())
    
    data = []
    for i in range(batch_size):
        rgb = Image.fromarray(np.asarray(raw_batch["rgbd"][i,:, :, :3], dtype=np.uint8))
        rgb_crop = Image.fromarray(np.asarray(raw_batch["rgbd_crop"][i, :, :, :3], dtype=np.uint8))
        

        depth = np.asarray(raw_batch["rgbd"][i,:, :, 3])
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = Image.fromarray(np.asarray(cm.Greys(depth)*255, dtype=np.uint8))
        depth_crop = np.asarray(raw_batch["rgbd_crop"][i,:, :, 3])
        depth_crop = (depth_crop - depth_crop.min()) / (depth_crop.max() - depth_crop.min())
        depth_crop = Image.fromarray(np.asarray(cm.Greys(depth_crop)*255, dtype=np.uint8))


        pick_heatmap = Image.fromarray(np.copy(rgb))
        pick_draw = ImageDraw.Draw(pick_heatmap)
        pick_draw.ellipse(
                (raw_batch["pixels"][0][i,0]-10, 
                 raw_batch["pixels"][0][i,1]-10, 
                 raw_batch["pixels"][0][i,0]+10, 
                 raw_batch["pixels"][0][i,1]+10), 
                fill=(255, 0, 0, 0))

        place_heatmap = Image.fromarray(np.copy(rgb))
        place_draw = ImageDraw.Draw(place_heatmap)
        place_draw.ellipse(
                (raw_batch["pixels"][1][i,0]-10, 
                 raw_batch["pixels"][1][i,1]-10, 
                 raw_batch["pixels"][1][i,0]+10, 
                 raw_batch["pixels"][1][i,1]+10), 
                fill=(255, 0, 0, 0))

        data.append([
                wandb.Image(rgb),
                wandb.Image(depth),
                wandb.Image(rgb_crop),
                wandb.Image(depth_crop),
                wandb.Image(pick_heatmap),
                wandb.Image(place_heatmap),
                ])
        
    table = wandb.Table(data=data, columns=column_names)
    wandb.log({"dataset": table})


def visualize_transporter_predictions(cfg, transporter, raw_batch, epoch):
    """
    Visualize transporter predictions.
    """
    batch_size = cfg.training.transporter_pick.batch_size
    table_config = dict(cfg.wandb.prediction_visualization.columns)
    column_names = list(table_config.keys())

    pick_pred = transporter.pick_model_state.apply_fn(
            {"params": transporter.pick_model_state.params,
             "batch_stats": transporter.pick_model_state.batch_stats},
            raw_batch["rgbd_normalized"],
            train=False).__array__().copy()

    place_pred = transporter.place_model_state.apply_fn(
            {"params": transporter.place_model_state.params},
            raw_batch["rgbd_normalized"],
            raw_batch["rgbd_crop_normalized"],
            train=False)

    data = []
    for i in range(cfg.training.transporter_pick.batch_size):
        # inspect input data
        rgb = np.asarray(raw_batch["rgbd"][i,:, :, :3], dtype=np.uint8)
        pick_rgb = rgb.copy()
        pick_rgb = Image.fromarray(pick_rgb)
        pick_draw = ImageDraw.Draw(pick_rgb)
        pick_draw.ellipse(
                (raw_batch["pixels"][0][i,0]-10, 
                 raw_batch["pixels"][0][i,1]-10, 
                 raw_batch["pixels"][0][i,0]+10, 
                 raw_batch["pixels"][0][i,1]+10), 
                fill=(255, 0, 0, 0))
        
        place_rgb = rgb.copy()
        place_rgb = Image.fromarray(place_rgb)
        place_draw = ImageDraw.Draw(place_rgb)
        place_draw.ellipse(
                (raw_batch["pixels"][1][i,0]-10,
                raw_batch["pixels"][1][i,1]-10,
                raw_batch["pixels"][1][i,0]+10,
                raw_batch["pixels"][1][i,1]+10),
                fill=(255, 0, 0, 0))

        # inspect model predictions
        pick_pred_ = pick_pred[i,:].copy()
        pick_pred_ = (pick_pred_ - pick_pred_.min()) / ((pick_pred_.max() - pick_pred_.min()))
        pick_heatmap = pick_pred_.reshape((160, 320))
        pick_heatmap = Image.fromarray(np.asarray(cm.viridis(pick_heatmap)*255, dtype=np.uint8))

        place_pred_ = place_pred[i,:].copy()
        place_pred_ = (place_pred_ - place_pred_.min()) / ((place_pred_.max() - place_pred_.min()))
        place_heatmap = place_pred_.reshape((160, 320))
        place_heatmap = Image.fromarray(np.asarray(cm.viridis(place_heatmap)*255, dtype=np.uint8))

        data.append([
                wandb.Image(pick_rgb),
                wandb.Image(place_rgb),
                wandb.Image(pick_heatmap),
                wandb.Image(place_heatmap),
                ])
    
    table = wandb.Table(data=data, columns=column_names)
    wandb.log({f"model_predictions_epoch_{epoch}": table})
