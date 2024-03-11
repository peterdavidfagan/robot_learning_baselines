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


def create_optimizer(cfg):
    """
    Instantiate opitmizer based on config.

    For now only one option is supported.
    """

    learning_rate_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=cfg.training.initial_lr,
        peak_value=cfg.training.peak_lr,
        warmup_steps=cfg.training.warmup_epochs * cfg.training.steps_per_epoch,
        decay_steps=(cfg.training.num_epochs - cfg.training.warmup_epochs)
        * cfg.training.steps_per_epoch,
        end_value=cfg.training.end_lr,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training.max_grad_norm),
        optax.adamw(learning_rate_scheduler, weight_decay=cfg.training.weight_decay),
    )
    
    return optimizer

