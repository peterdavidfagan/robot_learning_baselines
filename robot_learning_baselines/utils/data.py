"""Utilities for loading and inspecting datasets."""

# standard libraries
import os
import shutil
import urllib.request

# dataset
import rlds
import tensorflow as tf
import tensorflow_datasets as tfds
from octo.data.oxe import make_oxe_dataset_kwargs, make_oxe_dataset_kwargs_and_weights
from octo.data.dataset import make_interleaved_dataset, make_single_dataset

import jax
import jax.numpy as jnp
import einops as e


## Data Loading ##

def load_transporter_dataset(cfg):
    """Load transporter dataset."""
    
    def transporter_step(step):
        return {"rgbd": tf.concat([
                            tf.cast(step["observation"]["overhead_camera/rgb"][0], dtype=tf.float32), 
                            tf.expand_dims(step["observation"]["overhead_camera/depth"][0], axis=-1)
                            ], axis=-1),
                "pick_pose": step["action"][0][:7],
                "place_pose": step["action"][0][7:],
                "camera_intrinsics": step["camera_intrinsics"][0],
                "camera_extrinsics": step["camera_extrinsics"][0],
                }
    
    def episode_step_to_transition(episode):
        episode[rlds.STEPS] = rlds.transformations.batch(episode[rlds.STEPS], 
                size=2, 
                shift=1, 
                drop_remainder=True).map(transporter_step)
        return episode
    
    ds = tfds.builder_from_directory(f"{cfg.tfds_data_dir}/{cfg.dataset}").as_dataset(split="train")
    ds = ds.map(episode_step_to_transition)
    ds = ds.flat_map(lambda x: x[rlds.STEPS]) # convert from episodes to steps
    ds = ds.batch(cfg.batch_size)

    return ds 


def oxe_load_single_dataset(cfg):
    dataset_kwargs = make_oxe_dataset_kwargs(
        cfg.dataset,
        cfg.tfds_data_dir,
            )
    dataset = make_single_dataset(
            dataset_kwargs, 
            train=True,
            traj_transform_kwargs = {
                "window_size": 2, # for octo we will take a history of two
                },
            frame_transform_kwargs = {
                "resize_size": (280,280)
                },
            )
    train_dataset = (
        dataset.flatten() # flattens trajectories into individual frames
        .shuffle(cfg.shuffle_buffer_size) # shuffles the frames
        .batch(cfg.batch_size) # batches the frames
    )

    return train_dataset

def oxe_load_dataset(cfg):
    """Load dataset using the oxe dataset loader."""
    dataset_kwargs_list, sample_weights = make_oxe_dataset_kwargs_and_weights(
    "oxe_magic_soup",
    cfd.tfds_data_dir,
    load_camera_views=("primary", "wrist"),
    )

    # each element of `dataset_kwargs_list` can be used with `make_single_dataset`, but let's
    # use the more powerful `make_interleaved_dataset` to combine them for us!
    dataset = make_interleaved_dataset(
        dataset_kwargs_list,
        sample_weights,
        train=True,
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        batch_size=config.batch_size,
        traj_transform_kwargs=dict(
            goal_relabeling_strategy="uniform",  # let's get some goal images
            window_size=2,  # let's get some history
            future_action_window_size=3,  # let's get some future actions for action chunking
            subsample_length=100,  # subsampling long trajectories improves shuffling a lot
        ),
        frame_transform_kwargs=dict(
            image_augment_kwargs=dict(
                augment_order=["random_resized_crop", "random_brightness"],
                random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
                random_brightness=[0.1],
            ),
            resize_size=dict(
                primary=(256, 256),
                wrist=(128, 128),
            ),
            # If parallelism options are not provided, they will default to tf.Data.AUTOTUNE.
            # However, we would highly recommend setting them manually if you run into issues
            # with memory or dataloading speed. Frame transforms are usually the speed
            # bottleneck (due to image decoding, augmentation, and resizing), so you can set
            # this to a very high value if you have a lot of CPU cores. Keep in mind that more
            # parallel calls also use more memory, though.
            num_parallel_calls=64,
        ),
        # Same spiel as above about performance, although trajectory transforms and data reading
        # are usually not the speed bottleneck. One reason to manually set these is if you want
        # to reduce memory usage (since autotune may spawn way more threads than necessary).
        traj_transform_threads=16,
        traj_read_threads=16,
    )

    # Another performance knob to tune is the number of batches to prefetch -- again,
    # the default of tf.data.AUTOTUNE can sometimes use more memory than necessary.
    iterator = dataset.iterator(prefetch=1)

    return iterator

## Data Preprocessing ##

def preprocess_transporter_batch(batch):
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


def preprocess_batch(batch, text_tokenize_fn, action_head_type="diffusion", dummy=False):
    """
    Preprocess a batch of data.
    """

    # process raw data
    text = [task.decode("utf-8") for task in batch["task"]["language_instruction"]]
    text_tokens = text_tokenize_fn(
            text,
            )["input_ids"]
    images = batch["observation"]["image_primary"]
    gt_action = jnp.take(batch["action"], -1, axis=1)

    # adapt raw data for different action heads
    if action_head_type=="diffusion":
        if dummy:
            time = jnp.ones((images.shape[0], 1))
            actions = jnp.take(batch["action"], -1, axis=1)
            data = {
                    "images": images,
                    "text_tokens": text_tokens,
                    "time": time,
                    "noisy_actions": actions,
                    }
        else:
            data = {
                    "images": images,
                    "text_tokens": text_tokens,
                    "gt_action": gt_action,
                    }
    
    else:
        if dummy:
            data = {
                    "images": images,
                    "text_tokens": text_tokens,
                    }
        else:
            data = {
                    "images": images,
                    "text_tokens": text_tokens,
                    "gt_action": gt_action,
                    }

    return data
