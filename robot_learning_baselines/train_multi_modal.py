"""Training script for concept learning model."""
# standard libraries
import os
import gc
from time import time
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

# model architecture/train state
from multi_modal_transformers.models.octo.octo import Octo
from multi_modal_transformers.models.octo.octo import create_octo_train_state

# deep learning framework
import jax
import jax.numpy as jnp
import jax.random as random
from jax.nn import softmax
import flax.linen as nn
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

# custom training pipeline utilities
from utils.data import (
    oxe_load_single_dataset,
    oxe_load_dataset,
    preprocess_batch,
)

from utils.pipeline import (
    inspect_model,
    setup_checkpointing,
    create_optimizer,
)

from utils.wandb import (
    init_wandb,
    visualize_dataset,
)

@hydra.main(version_base=None, config_path="./config", config_name="octo-base")
def main(cfg: DictConfig) -> None:
    """Model training loop."""

    key = random.PRNGKey(0)
    key, model_key, dropout_key, image_tokenizer_key, diffusion_key = random.split(key, 5)
    rngs = {
            "params": model_key,
            "patch_encoding": image_tokenizer_key,
            "dropout": dropout_key,
            "diffusion": diffusion_key,
            }

    train_data = oxe_load_single_dataset(cfg.dataset) # load dataset for debugging
    #train_data = oxe_load_dataset(cfg.data.open-x-embodiment, cfg.training.decoder_only) # load dataset
    
    if cfg.wandb.use: # optionally initialise wandb
        init_wandb(cfg)
        visualize_dataset(cfg, next(train_data.as_numpy_iterator()))
        
    
    chkpt_manager = setup_checkpointing(cfg.training) # set up model checkpointing   
    optimizer = create_optimizer(cfg) # instantiate model optimizer
    model = Octo(cfg.architecture.multi_modal_transformer) # instantiate model
    text_tokenizer = instantiate(cfg.architecture.multi_modal_transformer.tokenizers.text.tokenizer) # instantiate text tokenizer
    text_tokenize_fn = partial(text_tokenizer, 
                               return_tensors="jax", 
                               max_length=16, # hardcode while debugging
                               padding="max_length", 
                               truncation=True
                               )
    
    # initialize the training state
    batch = next(train_data.as_numpy_iterator())
    input_data = preprocess_batch(batch, text_tokenize_fn, dummy=True)
    inspect_model(model, rngs, input_data, method="predict_diffusion_denoise_term")
    train_state = create_octo_train_state(
        input_data["text_tokens"],
        input_data["images"],
        text_tokenizer,
        {"time": input_data["time"], "noisy_actions": input_data["noisy_actions"]},
        rngs,
        model,
        optimizer
        )

    for epoch in tqdm(range(cfg.training.num_epochs), leave=False):
        
        # epoch metrics
        metrics_history = {
            "denoise_loss": [],
        }

        # shuffle dataset and create iterator
        train_data = train_data.shuffle(10)
        train_data_iter = train_data.as_numpy_iterator()
    
        for batch in train_data_iter:
            data = preprocess_batch(batch, train_state.text_tokenize_fn)
            train_state = train_state.diffusion_train_step(
                    model, 
                    train_state, 
                    data["text_tokens"], 
                    data["images"], 
                    data["gt_action"],
                    )

        # compute and track metrics
        for metric, value in train_state.metrics.compute().items():
            metrics_history[f"{metric}"].append(value)
        train_state = train_state.replace(metrics=train_state.metrics.empty())
        
        if cfg.wandb.use:
            wandb.log({
                        "denoise_loss": metrics_history["denoise_loss"][-1],
                    })
        print(f"Epoch {epoch} train loss: {metrics_history['denoise_loss'][-1]}")


        # save model checkpoint
        chkpt_manager.save(epoch, train_state)


if __name__ == "__main__":
    main()

