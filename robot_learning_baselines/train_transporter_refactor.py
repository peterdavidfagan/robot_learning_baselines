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
    load_hf_transporter_dataset,
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
    visualize_transporter_predictions,
)


@hydra.main(version_base=None, config_path=".")
def main(cfg: DictConfig) -> None:
    """Model training loop."""
    
    assert jax.default_backend() != "cpu" # ensure accelerator is available
    cfg = cfg["config"] # some hacky and wacky stuff from hydra (TODO: revise)

    key = random.PRNGKey(0)
    pick_model_key, place_model_key = jax.random.split(key, 2)
    
    train_data = load_hf_transporter_dataset(cfg.dataset)
    cardinality =  train_data.reduce(0, lambda x,_: x+1).numpy()

    if cfg.wandb.use:
        init_wandb(cfg)
        batch = next(train_data.as_numpy_iterator())
        (rgbd, rgbd_crop), (rgbd_normalized, rgbd_crop_normalized), pixels, ids = preprocess_transporter_batch(
            jnp.asarray(batch['pick_rgb']), 
            jnp.asarray(batch['pick_depth']), 
            jnp.asarray(batch['pick_pixel_coords']), 
            jnp.asarray(batch['place_pixel_coords']),
            )
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
    (rgbd, rgbd_crop), (rgbd_normalized, rgbd_crop_normalized), pixels, ids = preprocess_transporter_batch( 
            jnp.asarray(batch['pick_rgb']), 
            jnp.asarray(batch['pick_depth']), 
            jnp.asarray(batch['pick_pixel_coords']), 
            jnp.asarray(batch['place_pixel_coords'])
            )
    eval_data = {
            "rgbd": rgbd,
            "rgbd_crop": rgbd_crop,
            "rgbd_normalized": rgbd_normalized,
            "rgbd_crop_normalized": rgbd_crop_normalized,
            "pixels": pixels,
            "ids": ids,
            }
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

    # train both pick/place models together
    for epoch in range(cfg.training.transporter_pick.num_epochs):
        
        # epoch metrics
        metrics_history = {
            "loss": [],
        }

        # shuffle dataset
        train_data_epoch = train_data.shuffle(16)
        
        # TODO: get dataset size and use tqdm
        for batch in tqdm(train_data_epoch, leave=False, total=cardinality):
            (rgbd, rgbd_crop), (rgbd_normalized, rgbd_crop_normalized), pixels, ids = preprocess_transporter_batch( 
                jnp.asarray(batch['pick_rgb']), 
                jnp.asarray(batch['pick_depth']), 
                jnp.asarray(batch['pick_pixel_coords']), 
                jnp.asarray(batch['place_pixel_coords']),
                )

            # compute ce loss for pick network and update pick network
            pick_train_state, pick_loss = pick_train_step(
                    transporter.pick_model_state, 
                    rgbd_normalized, 
                    ids[0])
            transporter = transporter.replace(pick_model_state=pick_train_state) 
            
            # compute ce loss for place networks and update place network
            place_train_state, place_loss = place_train_step(
                    transporter.place_model_state,
                    rgbd_normalized,
                    rgbd_crop_normalized, 
                    ids[1],
                    )
            transporter = transporter.replace(place_model_state=place_train_state)
            

        # report epoch metrics (optionally add to wandb)
        pick_loss_epoch = transporter.pick_model_state.metrics.compute()
        place_loss_epoch = transporter.place_model_state.metrics.compute()
        print(f"Epoch {epoch}: pick_loss: {pick_loss_epoch}, place_loss: {place_loss_epoch}")
        
        if cfg.wandb.use and (epoch%5==0):
            wandb.log({
                "pick_loss": pick_loss_epoch,
                "place_loss": place_loss_epoch,
                "epoch": epoch
                })
            visualize_transporter_predictions(cfg, transporter, eval_data, epoch)

        # reset metrics after epoch
        transporter.pick_model_state.replace(metrics=pick_train_state.metrics.empty())
        transporter.place_model_state.replace(metrics=place_train_state.metrics.empty())

        # save model checkpoint
        pick_chkpt_manager.save(epoch, pick_train_state)
        place_chkpt_manager.save(epoch, place_train_state)



if __name__ == "__main__":
    main()
