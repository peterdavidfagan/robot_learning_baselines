"""Training transporter"""

import wandb
from tqdm import tqdm
from PIL import Image, ImageDraw
from matplotlib import cm
import numpy as np 

import tensorflow_datasets as tfds
from envlogger import reader

from transporter_networks.transporter import (
        Transporter,
        TransporterNetwork,
        TransporterPlaceNetwork,
        create_transporter_train_state,
        create_transporter_place_train_state,
        pick_train_step,
        place_train_step,
        )
from transporter_networks.goal_transporter import GoalConditionedTransporter

from rearrangement_benchmark.tasks.rearrangement import RearrangementEnv

from train_utils.wandb import (
        init_wandb, 
        )

from train_utils.pipeline import (
        load_dataset,
        )

import einops as e
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

import hydra
from hydra import compose, initialize
from hydra.utils import call, instantiate


@hydra.main(version_base=None, config_path="./config", config_name="transporter")
def main(cfg: DictConfig) -> None:
    """Model training loop."""
    assert jax.default_backend() != "cpu" # ensure accelerator is available
    
    key = jax.random.PRNGKey(0)
    model1_key, model2_key = jax.random.split(key, 2)

    # TODO: complete these
    #chkpt_manager = setup_checkpointing(cfg.training) # set up model checkpointing   
    pick_optimizer = instantiate(config.training.pick.optimizer)
    place_optimizer = instantiate(config.training.place.optimizer)
    pick_model = TransporterNetwork(config=config.model.pick)
    place_model = TransporterPlaceNetwork(config=config.model.place)
    
    # TODO: replace dummy data with actual batch from dataset
    #batch = next(train_data.as_numpy_iterator())
    dummy_rgbd_data = jnp.ones((10, 160, 320, 4), dtype=jnp.float32)
    dummy_rgbd_crop_data = jnp.ones((10, 64, 64, 4), dtype=jnp.float32)
    
    pick_train_state = create_transporter_train_state(
            dummy_rgbd_data,
            pick_model,
            model1_key,
            pick_optimizer,
            )
    # inspect_transporter_model()

    place_train_state = create_transporter_place_train_state(
            dummy_rgbd_data,
            dummy_rgbd_crop_data,
            place_model,
            model2_key,
            place_optimizer,
            )
    # inspect_transporter_model()

    transporter = Transporter(
            pick_model_state = pick_train_state,
            place_model_state = place_train_state,
            )


    raise NotImplementedError

if __name__ == "__main__":
    assert jax.default_backend() != "cpu" # ensure accelerator is available
    
    key = jax.random.PRNGKey(0)
    model1_key, model2_key = jax.random.split(key, 2)
    
    # train_data = 
    # cardinality =  train_data.reduce(0, lambda x,_: x+1).numpy()

    if config.wandb.use:
        init_wandb(config)
        #visualize_dataset(cfg, next(train_data.as_numpy_iterator()))

    
    # TODO: complete these
    #chkpt_manager = setup_checkpointing(cfg.training) # set up model checkpointing   
    pick_optimizer = instantiate(config.training.pick.optimizer)
    place_optimizer = instantiate(config.training.place.optimizer)
    pick_model = TransporterNetwork(config=config.model.pick)
    place_model = TransporterPlaceNetwork(config=config.model.place)
    
    
    
    # TODO: replace dummy data with actual batch from dataset
    #batch = next(train_data.as_numpy_iterator())
    dummy_rgbd_data = jnp.ones((10, 160, 320, 4), dtype=jnp.float32)
    dummy_rgbd_crop_data = jnp.ones((10, 64, 64, 4), dtype=jnp.float32)
    
    pick_train_state = create_transporter_train_state(
            dummy_rgbd_data,
            pick_model,
            model1_key,
            pick_optimizer,
            )
    # inspect_transporter_model()

    place_train_state = create_transporter_place_train_state(
            dummy_rgbd_data,
            dummy_rgbd_crop_data,
            place_model,
            model2_key,
            place_optimizer,
            )
    # inspect_transporter_model()

    transporter = Transporter(
            pick_model_state = pick_train_state,
            place_model_state = place_train_state,
            )


    # TODO: migrate process_batch to utilities file
    def process_batch(batch):
        """
        Process batch of data.

        For now some of the data processing happens within the training loop.
        This is not ideal and data should be precomputed, this is temporary until training pipeline is more mature
        """
        def pose_to_pixel_space(pose, camera_intrinsics, camera_extrinsics):
            """Map pose to pixel space."""
            # convert world coordinates to camera coordinates
            pose = jnp.concatenate([pose, jnp.ones(1)], axis=-1)
            camera_coords = camera_extrinsics @ pose
            camera_coords = camera_coords[:3] / camera_coords[3]

            # convert camera coordinates to pixel coordinates
            pixel_coords = camera_intrinsics @ camera_coords
            pixel_coords = pixel_coords[:2] / pixel_coords[2]
            pixel_coords = jnp.round(pixel_coords).astype(jnp.int32)

            return pixel_coords
        
        def pixel_to_idx(pixel, image_width=640):
            """Convert pixel x,y coordinate to one-hot vector."""
            pixel = jnp.round(pixel).astype(jnp.int32)
            return (pixel[1] * image_width) + pixel[0]
            
        def crop_rgbd(rgbd, pixel_coords, crop_size=(64, 64)):
            """Crop rgbd image around pixel coordinates."""
            x, y = pixel_coords
            
            # get starting coords for cropping
            starting_x = jnp.round(x-(crop_size[0]//2)).astype(jnp.int32)
            starting_y = jnp.round(y-(crop_size[1]//2)).astype(jnp.int32)
            
            return jax.lax.dynamic_slice(rgbd, (starting_y, starting_x, 0), (crop_size[0], crop_size[1], 4))
       
        def normalize_rgbd(rgbd):
            """Normalize rgbd image."""
            # divide by 255 to get values between 0 and 1
            rgbd.at[:, :, :3].set(rgbd[:, :, :3] / 255.0)

            # calculate channel-wise mean
            mean = jnp.mean(rgbd, axis=(0, 1))
            std = jnp.std(rgbd, axis=(0, 1))
            rgbd = (rgbd - mean) / std
            return rgbd

        b, h, w, c = batch["rgbd"].shape
        # TODO: move downsampling params to config, dynamically create from input image size
        h_downsample = 3
        w_downsample = 2
        
        # overwrite pick height with object height of 0.425 as currently the pick height in raw dataset is pre-grasp height
        pick_pose = batch["pick_pose"][:, :3].copy()
        pick_pose[:, 2] = 0.425
        place_pose = batch["place_pose"][:, :3].copy()
        
        # map poses to pixels and pixel indices using camera intrinsics and extrinsics
        camera_intrinsics = e.repeat(batch["camera_intrinsics"], "b h w -> (b repeat) h w", repeat=2)
        camera_extrinsics = e.repeat(batch["camera_extrinsics"], "b h w -> (b repeat) h w", repeat=2)
        poses, ps = e.pack([pick_pose, place_pose], "* l") # 2b 3
        pixels = jax.vmap(pose_to_pixel_space, (0, 0, 0), 0)(poses, camera_intrinsics, camera_extrinsics) # 2b 2
        
        # account for downsampling of image
        # x coord corresponds to height, y coord corresponds to width
        pixels, ps = e.pack([pixels[:, 0] // w_downsample, pixels[:, 1] // h_downsample], "b *")
        pick_pixels, place_pixels = e.unpack(pixels, [[b], [b]], "* p")
        ids = jax.vmap(pixel_to_idx, (0, None), 0)(pixels, w // w_downsample)
        pick_ids, place_ids = e.unpack(ids, [[b], [b]], "*")
        
        # downsample image
        rgbd = jax.vmap(jax.image.resize, in_axes=(0, None, None), out_axes=0)(batch["rgbd"],(h // h_downsample, w // w_downsample, 4), "nearest")
        
        # normalize rgbd image
        rgbd_normalized = jax.vmap(normalize_rgbd, 0)(rgbd)
        
        # crop rgbd image about pick location
        rgbd_crop = jax.vmap(crop_rgbd, (0, 0), 0)(rgbd, pick_pixels)
        rgbd_crop_normalized = jax.vmap(normalize_rgbd, 0)(rgbd_crop)
    
        return (rgbd, rgbd_crop), (rgbd_normalized, rgbd_normalized), (pick_pixels, place_pixels), (pick_ids, place_ids)


    # TODO: Completely refactor what is below this line


    ds = load_dataset(config)
    
    # log demonstration dataset batch to wandb for sanity check (move to utils)
    if config.wandb.use:
        ds_prefetch = ds.shuffle(config.training.batch_size).batch(config.training.batch_size).as_numpy_iterator()
        batch = next(ds_prefetch)

        b, h, w, c = batch["rgbd"].shape
        (rgbd, rgbd_crop), (rgbd_normalized, rgbd_crop_normalized), pixels, ids = process_batch(batch)

        data = []
        for i in range(config.training.batch_size):
            rgb = Image.fromarray(np.asarray(rgbd[i,:, :, :3], dtype=np.uint8))
            rgb_crop = Image.fromarray(np.asarray(rgbd_crop[i, :, :, :3], dtype=np.uint8))
            depth = np.asarray(rgbd[i,:, :, 3])
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth = Image.fromarray(np.asarray(cm.Greys(depth)*255, dtype=np.uint8))
            

            pick_heatmap = Image.fromarray(np.copy(rgb))
            pick_draw = ImageDraw.Draw(pick_heatmap)
            pick_draw.ellipse(
                    (pixels[0][i,0]-10, 
                     pixels[0][i,1]-10, 
                     pixels[0][i,0]+10, 
                     pixels[0][i,1]+10), 
                    fill=(255, 0, 0, 0))

            place_heatmap = Image.fromarray(np.copy(rgb))
            place_draw = ImageDraw.Draw(place_heatmap)
            place_draw.ellipse(
                    (pixels[1][i,0]-10, 
                     pixels[1][i,1]-10, 
                     pixels[1][i,0]+10, 
                     pixels[1][i,1]+10), 
                    fill=(255, 0, 0, 0))

            data.append([
                    wandb.Image(rgb),
                    wandb.Image(depth),
                    wandb.Image(rgb_crop),
                    wandb.Image(pick_heatmap),
                    wandb.Image(place_heatmap),
                    ])
            
        table = wandb.Table(data=data, columns=["rgb", "depth", "rgb_crop", "pick_location", "place_location"])
        run.log({"dataset": table})

    for epoch in range(config.training.num_epochs):
        # shuffle dataset
        ds_batched = ds.shuffle(config.training.batch_size).batch(config.training.batch_size).as_numpy_iterator()
        
        # TODO: get dataset size and use tqdm
        for batch in ds_batched:
            (rgbd, rgbd_crop), (rgbd_normalized, rgbd_crop_normalized), pixels, ids = process_batch(batch)
            
            # compute ce loss for pick network and update pick network
            pick_train_state, pick_loss = pick_train_step(transporter.pick_model_state, rgbd_normalized, ids[0])
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
        if config.wandb.use:
            run.log({
                "pick_loss": pick_loss_epoch,
                "place_loss": place_loss_epoch,
                "epoch": epoch
                })

        # reset metrics after epoch
        transporter.pick_model_state.replace(metrics=pick_train_state.metrics.empty())
        transporter.place_model_state.replace(metrics=place_train_state.metrics.empty())

        # inspect model predictions on random subset of dataset
        if config.wandb.use:
            ds_prefetch = ds.shuffle(config.training.batch_size).batch(config.training.batch_size).as_numpy_iterator()
            batch = next(ds_prefetch)
            (rgbd, rgbd_crop), (rgbd_normalized, rgbd_crop_normalized), pixels, ids = process_batch(batch)
            
            # start with pick model
            pick_pred = transporter.pick_model_state.apply_fn(
                    {"params": transporter.pick_model_state.params,
                     "batch_stats": transporter.pick_model_state.batch_stats},
                    rgbd_normalized,
                    train=False).__array__().copy()

            place_pred = transporter.place_model_state.apply_fn(
                    {"params": transporter.place_model_state.params},
            #            "batch_stats": transporter.place_model_state.batch_stats},
                    rgbd_normalized,
                    rgbd_crop_normalized,
                    train=False)

            data = []
            for i in range(config.training.batch_size):
                # inspect input data
                rgb = np.asarray(rgbd[i,:, :, :3], dtype=np.uint8)
                pick_rgb = rgb.copy()
                pick_rgb = Image.fromarray(pick_rgb)
                pick_draw = ImageDraw.Draw(pick_rgb)
                pick_draw.ellipse(
                        (pixels[0][i,0]-10, 
                         pixels[0][i,1]-10, 
                         pixels[0][i,0]+10, 
                         pixels[0][i,1]+10), 
                        fill=(255, 0, 0, 0))
                
                place_rgb = rgb.copy()
                place_rgb = Image.fromarray(place_rgb)
                place_draw = ImageDraw.Draw(place_rgb)
                place_draw.ellipse(
                        (pixels[1][i,0]-10,
                        pixels[1][i,1]-10,
                        pixels[1][i,0]+10,
                        pixels[1][i,1]+10),
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
            
            table = wandb.Table(data=data, columns=["pick_target", "place_target", "pick_heatmap", "place_heatmap"])
            run.log({f"model_predictions_epoch_{epoch}": table})
