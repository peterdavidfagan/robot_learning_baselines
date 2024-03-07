"""Training script for concept learning model."""
# standard libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# model architecture/train state
from multi_modal_transformers.models.octo.octo import Octo
from multi_modal_transformers.models.octo.octo import create_octo_train_state

# deep learning framework
import jax.numpy as jnp
import jax.random as random
from jax.nn import softmax
import optax
import orbax.checkpoint as ocp

# tokenizer from huggingface
from transformers import AutoTokenizer

# experiment tracking
import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from clu import metrics

from utils.pipeline import (
    oxe_load_single_dataset,
    oxe_load_dataset,
    setup_checkpointing,
)

from utils.wandb import init_wandb

@hydra.main(version_base=None, config_path="./config", config_name="octo-base")
def main(cfg: DictConfig) -> None:
    """Model training loop."""
    # initialize weights and biases
    if cfg.wandb.use:
        init_wandb(cfg)

    # set up jax random number generator
    key = random.PRNGKey(0)
    key, model_key, dropout_key, image_tokenizer_key, diffusion_key = random.split(key, 5)
    rngs = {
            "params": model_key,
            "patch_encoding": image_tokenizer_key,
            "dropout": dropout_key,
            "diffusion": diffusion_key,
            }

    # variables to track training progress
    BEST_MODEL = None
    TEST_LOSS = np.inf

    # load the dataset
    train_data = oxe_load_single_dataset(cfg.dataset) # for now debug the datapipeline
    #train_data = oxe_load_dataset(cfg.data.open-x-embodiment, cfg.training.decoder_only)

    # setup experiment checkpointing for concept learner model
    chkpt_manager = setup_checkpointing(cfg.training)

    # instantiate model and text tokenizer
    model = Octo(cfg.architecture.multi_modal_transformer)
    text_tokenizer = instantiate(cfg.architecture.multi_modal_transformer.tokenizers.text.tokenizer)
    
    # instantiate model optimizer
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
    
    # create sample inputs for model initialization
    batch = next(train_data.as_numpy_iterator())
    text = [task.decode() for task in batch["task"]["language_instruction"]]
    tokenizer = AutoTokenizer.from_pretrained('t5-base', model_max_length=16)
    text_tokens = tokenizer(
            text, 
            return_tensors="jax", 
            max_length=16, # hardcode while debugging
            padding="max_length", 
            truncation=True,
            )["input_ids"]
    images = batch["observation"]["image_primary"]    
    time = jnp.ones((images.shape[0], 1))
    noisy_actions = jnp.ones((images.shape[0], 8))
    
    # initialize the training state
    train_state = create_octo_train_state(
        text_tokens,
        images,
        text_tokenizer,
        {"time": time, "noisy_actions": noisy_actions},
        rngs,
        model,
        optimizer
        )

    # training loop
    for epoch in range(cfg.training.num_epochs):
        print(f"Epoch {epoch + 1}/{cfg.training.num_epochs}")
        # shuffle dataset and create iterator
        #train_data = train_data.shuffle(10)

        #train_data_iter = train_data.as_numpy_iterator()

        # training
        #for batch in train_data_iter:
        #    train_state, grads = bc_train_step(
        #        train_state,
        #        batch["concept"],
        #        batch["trajectory"]["frame"],
        #        batch["trajectory"]["action"],
        #        batch["target_action"],
        #        dropout_key,
        #        image_tokenizer_key,
        #    )

        # save checkpoint
        #chkpt_manager.save(epoch, train_state)


if __name__ == "__main__":
    main()

