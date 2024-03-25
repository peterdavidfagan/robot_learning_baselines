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
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp


# introspecting model architecture
def inspect_model(model, variables, data, method="__call__"):
    """
    Prints tabulated information about flax model.
    """
    print(
        model.tabulate(
            variables, 
            **data, 
            method=method,
            )
        )


# saving model checkpoints
def setup_checkpointing(cfg, reinitialise=True):
    """Set up checkpointing."""
    if os.path.exists(cfg.checkpoint_dir) and reinitialise:
        # remove old files
        shutil.rmtree(cfg.checkpoint_dir)
    
    # create checkpoint directory 
    if reinitialise:
        os.makedirs(cfg.checkpoint_dir)

    # setup checkpoint manager
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


def create_optimizer(cfg, lr_schedule="reciprocal_sqrt"):
    """
    Instantiate opitmizer based on config.
    """

    if lr_schedule=="reciprocal_sqrt":
        
        def reciprocal_sqrt_schedule(
                init_value,
                peak_value,
                warmup_steps,
                ) -> optax.Schedule:
            """
            Constructs reciprocal sqrt schedule.
            """
            def schedule(count):
                if count <= warmup_steps:
                    lr_value = (peak_value - init_value) * (count / warmup_steps)
                else:
                    lr_value = peak_value * (1 / jnp.sqrt(count - warmup_steps))

                return lr_value
            
            return schedule

        learning_rate_scheduler = reciprocal_sqrt_schedule(
                init_value = cfg.training.initial_lr,
                peak_value=cfg.training.peak_lr,
                warmup_steps = cfg.training.warmup_steps
                )

        optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.training.max_grad_norm),
            optax.adamw(learning_rate_scheduler, weight_decay=cfg.training.weight_decay),
        )

    elif lr_schedule=="cosine_decay":
        learning_rate_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=cfg.training.initial_lr,
            peak_value=cfg.training.peak_lr,
            warmup_steps=cfg.training.warmup_epochs * cfg.training.steps_per_epoch,
            decay_steps=(cfg.training.num_epochs - cfg.training.warmup_epochs)
            * cfg.training.steps_per_epoch,
            end_value=cfg.training.end_lr,
        )

        optimizer = optax.chain(
            optax.clip(1.0),
            #optax.clip_by_global_norm(cfg.training.max_grad_norm),
            optax.adamw(learning_rate_scheduler, weight_decay=cfg.training.weight_decay),
        )
    
    elif lr_schedule=="constant":
        learning_rate_scheduler = optax.constant_schedule(cfg.training.initial_lr)
        optimizer = optax.chain(
            optax.clip(cfg.training.max_grad_norm),
            optax.adamw(learning_rate_scheduler, weight_decay=cfg.training.weight_decay),
        )

    else:
        raise NotImplementedError
    
    return optimizer, learning_rate_scheduler

