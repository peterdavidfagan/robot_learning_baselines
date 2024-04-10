"""Training script for concept learning model."""
# standard libraries
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".9"
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
import tensorflow as tf

# tokenizer from huggingface
from transformers import AutoTokenizer

# experiment tracking
import wandb
import hydra
from hydra.utils import instantiate, call
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
    visualize_multi_modal_predictions,
    track_gradients,
)

@hydra.main(version_base=None, config_path=".")
def main(cfg: DictConfig) -> None:
    """Model training loop."""
    
    assert jax.default_backend() != "cpu" # ensure accelerator is available
    cfg = cfg["config"] # some hacky and wacky stuff from hydra (TODO: revise)

    key = random.PRNGKey(0)
    key, model_key, dropout_key, image_tokenizer_key, diffusion_key = random.split(key, 5)
    rngs = {
            "params": model_key,
            "patch_encoding": image_tokenizer_key,
            "dropout": dropout_key,
            "diffusion": diffusion_key,
            }

    train_data = oxe_load_single_dataset(cfg.dataset) # load dataset for debugging
    cardinality = 3177 # hardcode while debugging
    #train_data = oxe_load_dataset(cfg.data.open-x-embodiment, cfg.training.decoder_only) # load dataset
    #cardinality =  train_data.reduce(0, lambda x,_: x+1).numpy()
    
    if cfg.wandb.use: # optionally initialise wandb
        init_wandb(cfg)
        visualize_dataset(cfg, next(train_data.as_numpy_iterator()))
    
    chkpt_manager = setup_checkpointing(cfg.training) # set up model checkpointing   
    optimizer, lr_scheduler = create_optimizer(cfg, lr_schedule="cosine_decay") # instantiate model optimizer
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
    input_data = preprocess_batch(
            batch, 
            text_tokenize_fn, 
            action_head_type=cfg.architecture.multi_modal_transformer.prediction_type, 
            dummy=True
            )
    #inspect_model(model, rngs, input_data, method=cfg.architecture.multi_modal_transformer.forward_method)

    # for now due to api we need to generate time + noisy actions data, this should be fixed in future
    input_data = preprocess_batch(
            batch, 
            text_tokenize_fn, 
            dummy=True
            )
    train_state = create_octo_train_state(
        input_data["text_tokens"],
        input_data["images"],
        text_tokenizer,
        {"time": input_data["time"], "noisy_actions": input_data["noisy_actions"]},
        rngs,
        model,
        optimizer,
        method=cfg.architecture.multi_modal_transformer.forward_method
        )
    
    # TODO: remove debug run once finished debugging
    if cfg.debug_run: # try overfitting a single sample
        data = preprocess_batch(
                batch, 
                train_state.text_tokenize_fn, 
                action_head_type=cfg.architecture.multi_modal_transformer.prediction_type, 
                dummy=False
                )

        loss = 1e3
        i = 0
        while loss > 1e-2:
            if cfg.architecture.multi_modal_transformer.prediction_type == "continuous":
                train_state, grads = train_state.continuous_train_step(
                        model, 
                        train_state, 
                        data["text_tokens"], 
                        data["images"], 
                        data["gt_action"],
                        )
            elif cfg.architecture.multi_modal_transformer.prediction_type == "categorical":
                train_state, grads = train_state.categorical_train_step(
                        model, 
                        train_state, 
                        data["text_tokens"], 
                        data["images"], 
                        data["gt_action"],
                        )
            elif cfg.architecture.multi_modal_transformer.prediction_type == "diffusion":
                train_state, grads = train_state.diffusion_train_step(
                        model, 
                        train_state, 
                        data["text_tokens"], 
                        data["images"], 
                        data["gt_action"],
                        )
            else:
                raise NotImplementedError

            i += 1
            loss = train_state.metrics.compute()["loss"]
            print(loss)
            train_state = train_state.replace(metrics=train_state.metrics.empty())

            if cfg.wandb.use:
                wandb.log({
                            "loss": loss,
                            "learning_rate": lr_scheduler(train_state.step),
                        })
            
            if i % 10 == 0:
                track_gradients(cfg, grads)
                #visualize_multi_modal_predictions(
                #        train_state, 
                #        model, 
                #        {key: data[key] for key in ["text_tokens", "images"]}, 
                #        data["gt_action"], 
                #        0, 
                #        method=cfg.architecture.multi_modal_transformer.action_heads.forward_method) # hardcode while debugging
    
    else:
        for epoch in tqdm(range(cfg.training.num_epochs), leave=False):
            
            # epoch metrics
            metrics_history = {
                "loss": [],
            }

            # shuffle dataset and create iterator
            train_data = train_data.shuffle(10)
            train_data_iter = train_data.as_numpy_iterator()

            for batch in tqdm(train_data_iter, leave=False, total=cardinality):
                if cfg.architecture.multi_modal_transformer.action_heads.type == "continuous":
                    data = preprocess_batch(batch, train_state.text_tokenize_fn, action_head_type="continuous", dummy=False)
                    train_state = train_state.continuous_train_step(
                            model, 
                            train_state, 
                            data["text_tokens"], 
                            data["images"], 
                            data["gt_action"],
                            )
                elif cfg.architecture.multi_modal_transformer.action_heads.type == "diffusion":
                    data = preprocess_batch(batch, train_state.text_tokenize_fn, action_head_type="diffusion", dummy=False)
                    train_state = train_state.diffusion_train_step(
                            model, 
                            train_state, 
                            data["text_tokens"], 
                            data["images"], 
                            data["gt_action"],
                            )
                else:
                    raise NotImplementedError


            # compute and track metrics
            for metric, value in train_state.metrics.compute().items():
                metrics_history[f"{metric}"].append(value)
            train_state = train_state.replace(metrics=train_state.metrics.empty())
            
            if cfg.wandb.use:
                wandb.log({
                            "epoch_loss": metrics_history["loss"][-1],
                            "epoch": epoch,
                        })
            print(f"Epoch {epoch} train loss: {metrics_history['loss'][-1]}")


            # save model checkpoint
            chkpt_manager.save(epoch, train_state)


if __name__ == "__main__":
    main()

