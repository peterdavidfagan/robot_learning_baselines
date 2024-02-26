"""Training script for concept learning model."""
# standard libraries
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
from omegaconf import DictConfig
from clu import metrics

# training utilities
from utils.octo import (
    create_train_state,
)

from utils.pipeline import (
    oxe_load_single_dataset,
    oxe_load_dataset,
    setup_checkpointing,
)

#from utils.wandb import (
#    init_wandb,
    # create_dataset_artifact,
#    create_model_artifact,
#    log_dataset_batch,
#    track_gradients,
#)

# model architecture
from multi_modal_transformers.models.octo import Octo

@hydra.main(version_base=None, config_path="./config", config_name="octo-base")
def main(cfg: DictConfig) -> None:
    """Model training loop."""
    # set up jax random number generator
    key = random.PRNGKey(0)
    key, model_key, dropout_key, image_tokenizer_key = random.split(key, 4)

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
    text_tokenizer = AutoTokenizer.from_pretrained('t5-base')
    
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

    # TODO: set up weights and biases

    # instantiate training state
    batch = next(train_data.as_numpy_iterator())
    text = [task.decode() for task in batch["task"]["language_instruction"]]
    text_ids = text_tokenizer(text, return_tensors="jax", padding=True, truncation=True)["input_ids"]
    #print(batch["observation"]["image_primary"].shape)
    #print(batch["action"].shape)
    #print(text_ids)
    train_state = create_train_state(
        text_ids,
        batch["observation"]["image_primary"],
        batch["action"],
        model_key,
        dropout_key,
        image_tokenizer_key,
        model,
        optimizer
            )

    #BEST_MODEL = train_state.params  # initialise best model

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

