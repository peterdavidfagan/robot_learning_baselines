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
    variables = model.init(
        rngs, 
        input_data["text_tokens"],
        input_data["images"],
        method="generate_readouts"
    )
    #inspect_model(model, rngs, input_data, method="predict_continuous_action")
    
    while True:
        print("...")
        start = time()
        model.apply(variables, input_data["text_tokens"], input_data["images"], rngs=rngs, method="generate_readouts")
        end = time()
        print("Time: {}".format(end-start))

if __name__ == "__main__":
    main()
