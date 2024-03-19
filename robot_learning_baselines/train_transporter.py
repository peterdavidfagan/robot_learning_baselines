"""Training transporter"""

# standard libraries
import os
from time import time
from tqdm import tqdm

# linear algebra and deep learning frameworks
import numpy as np 
import einops as e
import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
import optax

# dataset
import tensorflow_datasets as tfds
from envlogger import reader

# model architecture
from transporter_networks.transporter import (
        Transporter,
        TransporterNetwork,
        TransporterPlaceNetwork,
        create_transporter_train_state,
        create_transporter_place_train_state,
        pick_train_step,
        place_train_step,
        )

# experiment config and tracking
import hydra
from hydra.utils import call, instantiate
from omegaconf import DictConfig
import wandb


# custom training pipeline utilities
from utils.data import (
    load_transporter_dataset,
    preprocess_transporter_batch,
)

from utils.pipeline import (
    inspect_model,
    setup_checkpointing,
    create_optimizer,
)

from utils.wandb import (
    init_wandb,
    visualize_transporter_dataset,
)


@hydra.main(version_base=None, config_path="./config", config_name="transporter")
def main(cfg: DictConfig) -> None:
    """Model training loop."""
    assert jax.default_backend() != "cpu" # ensure accelerator is available
    
    key = random.PRNGKey(0)
    pick_model_key, place_model_key = jax.random.split(key, 2)
    
    train_data = load_transporter_dataset(cfg.dataset)
    
    if cfg.wandb.use:
        init_wandb(cfg)
        batch = next(train_data.as_numpy_iterator())
        (rgbd, rgbd_crop), (rgbd_normalized, rgbd_crop_normalized), pixels, ids = preprocess_transporter_batch(batch)
        batch = {
                "rgbd": rgbd,
                "rgbd_crop": rgbd_crop,
                "pixels": pixels,
                }
        visualize_transporter_dataset(cfg, batch)

    pick_chkpt_manager = setup_checkpointing(cfg.training.transporter_pick) # set up model checkpointing   
    place_chkpt_manager = setup_checkpointing(cfg.training.transporter_place) # set up model checkpointing 
    
    pick_optimizer = instantiate(cfg.training.transporter_pick.optimizer)
    place_optimizer = instantiate(cfg.training.transporter_place.optimizer)
    
    pick_model = TransporterNetwork(config=cfg.architecture.transporter.pick)
    place_model = TransporterPlaceNetwork(config=cfg.architecture.transporter.place)
   

    batch = next(train_data.as_numpy_iterator())
    (rgbd, rgbd_crop), (rgbd_normalized, rgbd_crop_normalized), pixels, ids = preprocess_transporter_batch(batch)
    pick_train_state = create_transporter_train_state(
            rgbd_normalized,
            pick_model,
            pick_model_key,
            pick_optimizer,
            )
    data = {"rgbd": rgbd_normalized}
    inspect_model(pick_model, {"params": pick_model_key}, data)

    place_train_state = create_transporter_place_train_state(
            rgbd_normalized,
            rgbd_crop_normalized,
            place_model,
            place_model_key,
            place_optimizer,
            )
    data = {"rgbd": rgbd_normalized, "rgbd_crop": rgbd_crop_normalized}
    inspect_model(place_model, {"params": place_model_key}, data)

    transporter = Transporter(
            pick_model_state = pick_train_state,
            place_model_state = place_train_state,
            )


if __name__ == "__main__":
    main()
