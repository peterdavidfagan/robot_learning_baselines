"""Utilities for loading and inspecting datasets."""

# standard libraries
import os
import shutil
import urllib.request

# dataset
import tensorflow as tf
import tensorflow_datasets as tfds
from octo.data.oxe import make_oxe_dataset_kwargs, make_oxe_dataset_kwargs_and_weights
from octo.data.dataset import make_interleaved_dataset, make_single_dataset

import jax.numpy as jnp

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


def preprocess_batch(batch, text_tokenize_fn, action_head_type="diffusion", dummy=False):
    """
    Preprocess a batch of data.
    """

    # tokenize text
    text = [task.decode("utf-8") for task in batch["task"]["language_instruction"]]
    text_tokens = text_tokenize_fn(
            text,
            )["input_ids"]

    # get image observations
    images = batch["observation"]["image_primary"]
    
    # get action
    gt_action = jnp.take(batch["action"], -1, axis=1)

    # create dummy data for diffusion-based model init
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
