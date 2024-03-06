"""Training Utilities."""
# standard libraries
import os
import shutil
import urllib.request

# dataset
import tensorflow as tf
import tensorflow_datasets as tfds
from octo.data.oxe import make_oxe_dataset_kwargs, make_oxe_dataset_kwargs_and_weights
from octo.data.dataset import make_interleaved_dataset, make_single_dataset

# deep learning framework
import optax
import orbax.checkpoint as ocp


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

def setup_checkpointing(cfg):
    """Set up checkpointing."""
    if os.path.exists(cfg.checkpoint_dir):
        # remove old files
        shutil.rmtree(cfg.checkpoint_dir)
        # recreate tmp directory 
        os.makedirs(cfg.checkpoint_dir)

    # setup checkpointing
    chkpt_options = ocp.CheckpointManagerOptions(
        max_to_keep=cfg.max_checkpoints,
        save_interval_steps=cfg.save_interval,
    )

    chkpt_manager = ocp.CheckpointManager(
        cfg.checkpoint_dir,
        ocp.PyTreeCheckpointer(),
        options=chkpt_options,
    )

    return chkpt_manager


def create_warmup_cosine_lr_fn(config):
    """
    Create learning rate schedule.

    source: https://flax.readthedocs.io/en/latest/guides/lr_schedule.html
    """
    # linear warmup
    warmup_fn = optax.linear_schedule(
        init_value=config.initial_lr,
        end_value=config.peak_lr,
        transition_steps=config.warmup_epochs * config.steps_per_epoch,
    )

    # cosine decay
    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=config.base_lr, decay_steps=cosine_epochs * config.steps_per_epoch
    )

    # join schedules
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn], boundaries=[config.warmup_epochs * config.steps_per_epoch]
    )

    return schedule_fn
