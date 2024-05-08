"""Utilities for loading, inspecting and processing datasets."""

# standard libraries
import os
import tarfile
import shutil
import urllib.request
from functools import partial

# dataset
from huggingface_hub import hf_hub_download
import rlds
import tensorflow as tf
import tensorflow_datasets as tfds
from octo.data.oxe import make_oxe_dataset_kwargs, make_oxe_dataset_kwargs_and_weights
from octo.data.dataset import make_interleaved_dataset, make_single_dataset

import jax
import jax.numpy as jnp
import einops as e


## Data Loading ##

def load_hf_transporter_dataset(cfg):
    """Load transporter dataset."""
    
    def transporter_step(step):
        return {
            "pick_rgb": step["observation"]["overhead_camera/rgb"][1],
            "pick_depth": step["observation"]["overhead_camera/depth"][1],
            "pick_pixel_coords": step["action"]["pixel_coords"][0],
            "place_rgb": step["observation"]["overhead_camera/rgb"][2],
            "place_depth": step["observation"]["overhead_camera/depth"][2],
            "place_pixel_coords": step["action"]["pixel_coords"][1],
            }

    def episode_step_to_transition(episode):
        episode[rlds.STEPS] = rlds.transformations.batch(episode[rlds.STEPS],
                size=3,
                shift=2,
                drop_remainder=True).map(transporter_step)
        return episode

    def image_augmentation(step):
        """
        Randomly apply augmentations to images.
        """

        def augment_image(image):
            transformations = [
                lambda x: tf.image.random_brightness(x, max_delta=0.2),
                lambda x: tf.image.random_contrast(x, lower=0.5, upper=1.5),
                lambda x: tf.image.random_saturation(x, lower=0.5, upper=1.5),
                lambda x: tf.image.random_hue(x, max_delta=0.2)
            ]
            for transform in transformations:
                if tf.random.uniform([], minval=0, maxval=1) < 0.5:  # Adjust probability as needed
                    image = transform(image)
            return image


        return {"pick_rgb": augment_image(step["pick_rgb"]),
                "pick_depth": step["pick_depth"],
                "pick_pixel_coords": step["pick_pixel_coords"],
                "place_rgb": augment_image(step["place_rgb"]),
                "place_depth": step["place_depth"],
                "place_pixel_coords": step["place_pixel_coords"],
                }
        
    # download data from huggingface
    DOWNLOAD_PATH="/tmp/transporter_dataset"
    for file in cfg["huggingface"]["files"]:
        hf_hub_download(
            repo_id=f"{cfg['huggingface']['entity']}/{cfg['huggingface']['repo']}",
            repo_type="dataset",
            filename=file,
            local_dir=DOWNLOAD_PATH,
        )

        COMPRESSED_FILEPATH=os.path.join(DOWNLOAD_PATH, file)
        with tarfile.open(COMPRESSED_FILEPATH, 'r:xz') as tar:
            tar.extractall(path=DOWNLOAD_PATH)
        os.remove(COMPRESSED_FILEPATH)

    # TODO: account for multuple dataset files, build each and define sampling
    # load with tfds
    ds = tfds.builder_from_directory(DOWNLOAD_PATH).as_dataset(split="train")
   
    # process and batch data
    ds = ds.map(episode_step_to_transition)
    ds = ds.flat_map(lambda x: x[rlds.STEPS]) # convert from episodes to steps
    ds = ds.map(image_augmentation)
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

@partial(jax.jit, static_argnums=(4,))
def preprocess_transporter(rgb, depth, pick_pixels, place_pixels, crop_idx):
    # crop images
    rgb_crop_raw = jax.lax.slice(rgb, (crop_idx[0], crop_idx[1], 0), (crop_idx[2], crop_idx[3],3))
    depth_crop_raw = jax.lax.slice(depth, (crop_idx[0], crop_idx[1]), (crop_idx[2], crop_idx[3]))

    # process depth
    nan_mask = jnp.isnan(depth_crop_raw)
    inf_mask = jnp.isinf(depth_crop_raw)
    mask = jnp.logical_or(nan_mask, inf_mask)
    max_val = jnp.max(depth_crop_raw, initial=0, where=~mask)
    depth_crop_filled = jnp.where(~mask, depth_crop_raw, max_val) # for now fill with max_val and hope the q-network learns to compensate

    # normalize and concatenate
    rgb_crop = jax.nn.standardize(rgb_crop_raw / 255.0)
    depth_crop = jax.nn.standardize(depth_crop_filled)
    depth_crop = e.rearrange(depth_crop, "h w -> h w 1")
    rgbd_crop, _ = e.pack([rgb_crop, depth_crop], 'h w *')
    rgbd_crop_raw, _ = e.pack([rgb_crop_raw, depth_crop_filled], 'h w *')

    # adjust pixel coords
    pick_pixels = pick_pixels.at[0].add(-crop_idx[1])
    pick_pixels = pick_pixels.at[1].add(-crop_idx[0])
    place_pixels = place_pixels.at[0].add(-crop_idx[1])
    place_pixels = place_pixels.at[1].add(-crop_idx[0])

    # crop image about pick location
    u_min = jnp.max(jnp.asarray([0, pick_pixels[1]-50]))
    v_min = jnp.max(jnp.asarray([0, pick_pixels[0]-50]))
    rgbd_pick_crop = jax.lax.dynamic_slice(
        rgbd_crop, 
        (u_min, v_min, 0), 
        (100, 100, 4)
        )
    rgbd_pick_crop_raw = jax.lax.dynamic_slice(
        rgbd_crop_raw, 
        (u_min, v_min, 0), 
        (100, 100, 4)
        )

    # generate one-hot ids for pixels
    image_width = rgbd_crop.shape[1]
    pick_id = (pick_pixels[1] *  image_width) + pick_pixels[0]
    place_id =  (place_pixels[1] *  image_width) + place_pixels[0]

    return  (rgbd_crop_raw, rgbd_pick_crop_raw), (rgbd_crop, rgbd_pick_crop), (pick_pixels, place_pixels), (pick_id, place_id)


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
