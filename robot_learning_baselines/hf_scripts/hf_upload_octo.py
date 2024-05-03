"""
A script to export model files and upload to huggingface.
"""
import os
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as random
from jax.experimental import jax2tf
import flax
from flax.training import orbax_utils
import orbax
import tensorflow as tf

# model architecture/train state
from multi_modal_transformers.models.octo.octo import Octo
from multi_modal_transformers.models.octo.octo import create_octo_train_state

# tokenizer from huggingface
from transformers import AutoTokenizer

import hydra
from hydra.utils import instantiate, call
from omegaconf import DictConfig

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

from utils.hugging_face import (
        push_model, 
        )

from utils.wandb import (
    init_wandb,
    visualize_dataset,
    visualize_multi_modal_predictions,
    track_gradients,
)


@hydra.main(version_base=None, config_path="..")
def main(cfg: DictConfig) -> None:
    cfg = cfg["config"] # some hacky and wacky stuff from hydra (TODO: revise)

    # upload to hugging face
    # compress the checkpoint dir
    import tarfile
    compressed_file = os.path.join(cfg.hf_upload.checkpoint_dir.split("/")[-1].join("/"), "checkpoint.tar.xz")
    with tarfile.open(compressed_file, "w:xz") as tar:
        tar.add(cfg.hf_upload.checkpoint_dir, arcname=".")

    print(compressed_file)

    push_model(
        entity = cfg.hf_upload.entity,
        repo_name = cfg.hf_upload.repo,
        branch = cfg.hf_upload.branch,
        checkpoint_dir = compressed_file,
            )
    
    # TODO: spend time getting onnx version working
    assert jax.default_backend() != "cpu" # ensure accelerator is available

    key = random.PRNGKey(0)
    key, model_key, dropout_key, image_tokenizer_key, diffusion_key = random.split(key, 5)
    rngs = {
            "params": model_key,
            "patch_encoding": image_tokenizer_key,
            "dropout": dropout_key,
            "diffusion": diffusion_key,
            }

    train_data = oxe_load_single_dataset(cfg.dataset) # load dataset for debugging
    
    chkpt_manager = setup_checkpointing(cfg.training, reinitialise=False) # set up model checkpointing   
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
    inspect_model(model, rngs, input_data, method=cfg.architecture.multi_modal_transformer.forward_method)
    

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
    

    # load model using orbax
    train_state = chkpt_manager.restore(cfg.hf_upload.checkpoint_dir, items=train_state)


    #if cfg.architecture.multi_modal_transformer.prediction_type == "continuous":
    #    def predict(text_tokens, images):
    #        return train_state.apply_fn(
    #                {"params": train_state.params}, 
    #                text_tokens, 
    #                images,
    #                method="predict_continuous_action")

    #elif cfg.architecture.multi_modal_transformer.prediction_type == "categorical":
    #    def predict(text_tokens, images):                              
    #        return train_state.apply_fn(
    #                {"params": train_state.params},
    #                text_tokens, 
    #                images,
    #                rngs=train_state.rngs,
    #                method="predict_action_logits")

    #elif cfg.architecture.multi_modal_transformer.prediction_type == "diffusion":
    #    def predict(text_tokens, images):                              
    #        return train_state.apply_fn(
    #                {"params": train_state.params},
    #                text_tokens, 
    #                images, 
    #                method="predict_diffusion_action")
    
    #else:
    #    raise NotImplementedError

    # convert model to tflite
    #tf_predict = tf.function(
    #        jax2tf.convert(predict, enable_xla=True),
    #        input_signature=[
    #            tf.TensorSpec(shape=input_data["text_tokens"].shape, dtype=tf.int32, name='text_tokens'),
    #            tf.TensorSpec(shape=input_data["images"].shape, dtype=tf.float32, name='images'),
    #            ],
    #        autograph=False)
    
    #converter = tf.lite.TFLiteConverter.from_concrete_functions(
    #    [tf_predict.get_concrete_function()], tf_predict)

    #tflite_float_model = converter.convert()
    
    # apply quantisation

    # convert model to onnx

    # upload onnx model to huggingface


if __name__=="__main__":
    main()
