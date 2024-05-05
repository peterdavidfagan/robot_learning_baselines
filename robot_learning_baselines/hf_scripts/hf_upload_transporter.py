"""
A script to export model files and upload to huggingface.
"""
# standard libraries
import os
from time import time
from tqdm import tqdm

# linear algebra and deep learning frameworks
import numpy as np 
import einops as e
import jax
import jax.numpy as jnp
import jax.random as random
import flax.linen as nn
import optax

# dataset
import tensorflow as tf
import tensorflow_datasets as tfds

# model architecture
from transporter_networks.transporter import (
        Transporter,
        TransporterNetwork,
        TransporterPlaceNetwork,
        create_transporter_train_state,
        create_transporter_place_train_state,
        pick_train_step,
        place_train_step,
        )

# experiment config and tracking
import hydra
from hydra.utils import call, instantiate
from omegaconf import DictConfig
import wandb


# custom training pipeline utilities
from robot_learning_baselines.utils.data import (
    load_hf_transporter_dataset,
    preprocess_transporter_batch,
)

from robot_learning_baselines.utils.pipeline import (
    inspect_model,
    setup_checkpointing,
    create_optimizer,
)

from robot_learning_baselines.utils.hugging_face import (
    push_model, 
)

from robot_learning_baselines.utils.wandb import (
    init_wandb,
    visualize_transporter_dataset,
    visualize_transporter_predictions,
)


@hydra.main(version_base=None, config_path="..")
def main(cfg: DictConfig) -> None:
    """Model training loop."""
    cfg = cfg["config"] # some hacky and wacky stuff from hydra (TODO: revise)


    # upload model checkpoints to hugging face
    import tarfile
    path, filename = os.path.split(cfg.hf_upload.checkpoint_dir)
    compressed_file = os.path.join(path, "checkpoint.tar.xz")
    with tarfile.open(compressed_file, "w:xz") as tar:
        tar.add(cfg.hf_upload.checkpoint_dir, arcname=".")

    push_model(
        entity = cfg.hf_upload.entity,
        repo_name = cfg.hf_upload.repo,
        branch = cfg.hf_upload.branch,
        upload_path = compressed_file,
        )

    assert jax.default_backend() != "cpu" # ensure accelerator is available

    key = random.PRNGKey(0)
    pick_model_key, place_model_key = jax.random.split(key, 2)
    
    train_data = load_hf_transporter_dataset(cfg.dataset)

    pick_chkpt_manager = setup_checkpointing(cfg.training.transporter_pick, reinitialise=False) # set up model checkpointing   
    place_chkpt_manager = setup_checkpointing(cfg.training.transporter_place, reinitialise=False) # set up model checkpointing 
    
    pick_optimizer = instantiate(cfg.training.transporter_pick.optimizer)
    place_optimizer = instantiate(cfg.training.transporter_place.optimizer)
    
    pick_model = TransporterNetwork(config=cfg.architecture.transporter.pick)
    place_model = TransporterPlaceNetwork(config=cfg.architecture.transporter.place)

    batch = next(train_data.as_numpy_iterator())
    (rgbd, rgbd_crop), (rgbd_normalized, rgbd_crop_normalized), pixels, ids = preprocess_transporter_batch( 
            jnp.asarray(batch['pick_rgb']), 
            jnp.asarray(batch['pick_depth']), 
            jnp.asarray(batch['pick_pixel_coords']), 
            jnp.asarray(batch['place_pixel_coords'])
            )

    pick_train_state = create_transporter_train_state(
            rgbd_normalized,
            pick_model,
            pick_model_key,
            pick_optimizer,
            )
    data = {"rgbd": rgbd_normalized}
    inspect_model(pick_model, {"params": pick_model_key}, data)

    place_train_state = create_transporter_place_train_state(
            rgbd_normalized,
            rgbd_crop_normalized,
            place_model,
            place_model_key,
            place_optimizer,
            )
    data = {"rgbd": rgbd_normalized, "rgbd_crop": rgbd_crop_normalized}
    inspect_model(place_model, {"params": place_model_key}, data)

    # load checkpoints
    pick_train_state = pick_chkpt_manager.restore(cfg.hf_upload.pick_checkpoint_dir, items=pick_train_state)
    place_train_state = place_chkpt_manager.restore(cfg.hf_upload.place_checkpoint_dir, items=place_train_state)
    
    # TODO: spend time getting onnx version working

    def predict(rgbd, rgbd_crop):
        pick_q_vals = pick_train_state.apply_fn({"params": pick_train_state.params},
                rgbd,
                train=False,
                )

        place_q_vals = place_train_state.apply_fn({"params": place_train_state.params},# "batch_stats": state.batch_stats}, 
                rgbd,
                rgbd_crop,
                train=False,
                )
        return pick_q_vals, place_q_vals

   
    # create tensorflow prediction function
    from jax.experimental import jax2tf
    tf_predict = tf.function(
           jax2tf.convert(predict, enable_xla=False),
           input_signature=[
                tf.TensorSpec(shape=(1,360,360,4), dtype=tf.float64, name='rgbd'),
                tf.TensorSpec(shape=(1,100,100,4), dtype=tf.float64, name='rgbd_crop'),
               ],
           autograph=False)

    # convert prediction function to onnx model
    import onnx
    import tf2onnx
    onnx_model, _ = tf2onnx.convert.from_function(
        function=tf_predict,
        input_signature=[
            tf.TensorSpec(shape=(1,360,360,4), dtype=tf.float64, name='rgbd'),
            tf.TensorSpec(shape=(1,100,100,4), dtype=tf.float64, name='rgbd_crop'),
        ],
        )
    onnx.save(onnx_model, './transporter.onnx')

    # upload onnx model to huggingface
    push_model(
        entity = cfg.hf_upload.entity,
        repo_name = cfg.hf_upload.repo,
        branch = cfg.hf_upload.branch,
        upload_path = "./transporter.onnx",
        )
    
    # convert prediction function to tflite
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
       [tf_predict.get_concrete_function()], tf_predict)
    converter.target_spec.supported_ops = [
      tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
      tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]
    tflite_float_model = converter.convert()
        
    # apply quantisation
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    f = open("transporter.tflite", 'wb')
    f.write(tflite_quantized_model)
    f.close()

    # upload tflite model to huggingface
    push_model(
        entity = cfg.hf_upload.entity,
        repo_name = cfg.hf_upload.repo,
        branch = cfg.hf_upload.branch,
        upload_path = "./transporter.tflite",
        )

if __name__=="__main__":
    main()
