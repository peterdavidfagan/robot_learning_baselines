"""Script to inspect model GPU utilisation and inference performance."""
# standard libraries
import os
from time import time
from functools import partial

# linear algebra and deep learning frameworks
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn

# model architecture/train state
from multi_modal_transformers.models.octo.octo import Octo
from transformers import AutoTokenizer

# experiment tracking
import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm import tqdm 

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


def benchmark_inference(cfg, model, variables, rngs, input_data, method, num_passes=100):
    """
    Generate benchmarks for model inference.
    """
    # perform preliminary forward pass 
    model.apply(variables, **input_data, rngs=rngs, method=method)

    # test model inference over a number of iterations
    inference_latency = []
    for _ in tqdm(range(num_passes)):
        start = time()
        for _ in tqdm(range(20)):
            model.apply(variables, **input_data, rngs=rngs, method=method)
        end = time()
        inference_time = (end - start) / 20
        inference_latency.append(inference_time)
        wandb.log({
                f"{cfg.training.batch_size}_batch_inference_latency": inference_time
            })
    
    wandb.log({
                f"{cfg.training.batch_size}_batch_mean_inference_latency": np.mean(inference_latency)
            })

def compute_flops(cfg, model, variables, rngs, input_data, method):
    """
    Compute model floating point operations. 
    """
    raise NotImplementedError


@hydra.main(version_base=None, config_path=".")
def main(cfg: DictConfig) -> None:
    """Performance benchmarks for multi modal model architectures."""
    assert jax.default_backend() != "cpu" # ensure accelerator is available
    cfg = cfg["config"] # some hacky and wacky stuff from hydra (TODO: revise)
    
    init_wandb(cfg)

    key = random.PRNGKey(0)
    key, model_key, dropout_key, image_tokenizer_key, diffusion_key = random.split(key, 5)
    rngs = {
            "params": model_key,
            "patch_encoding": image_tokenizer_key,
            "dropout": dropout_key,
            "diffusion": diffusion_key,
            }

    # load dataset and preprocess batch
    train_data = oxe_load_single_dataset(cfg.dataset)
    text_tokenizer = instantiate(cfg.architecture.multi_modal_transformer.tokenizers.text.tokenizer) # instantiate text tokenizer
    text_tokenize_fn = partial(text_tokenizer, 
                               return_tensors="jax", 
                               max_length=16, # hardcode while debugging
                               padding="max_length", 
                               truncation=True
                               )
    batch = next(train_data.as_numpy_iterator())
    input_data = preprocess_batch(batch, text_tokenize_fn, action_head_type="placeholder", dummy=True)
    del batch
    
    # initialise model params
    model = Octo(cfg.architecture.multi_modal_transformer) # instantiate model
    variables = model.init(
        rngs, 
        input_data["text_tokens"],
        input_data["images"],
        method="generate_readouts"
    )
    
    # test model inference speed 
    benchmark_inference(cfg, model, variables, rngs, input_data, method="generate_readouts")

    # compute model FLOPs

if __name__ == "__main__":
    main()
