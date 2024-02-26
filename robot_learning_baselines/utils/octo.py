"""Training Utilities."""
# standard libraries
import numpy as np

# deep learning framework
import optax
import jax
import jax.numpy as jnp
import jaxlib
from jaxlib.xla_extension import ArrayImpl
from flax import struct
from flax.training import train_state
import einops as e

# metrics
from clu import metrics
from omegaconf import DictConfig


# Define Flax Training State
@struct.dataclass
class Metrics(metrics.Collection):
    """Training metrics."""

    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    """Train state."""

    metrics: Metrics
    dropout_key: ArrayImpl
    image_tokenizer_key: ArrayImpl


def create_train_state(
        task, 
        observation, 
        action, 
        model_key, 
        dropout_key, 
        image_tokenizer_key, 
        model, 
        optimizer):
    """Create initial training state."""
    variables = model.init(
        {
            "params": model_key,
            "dropout": dropout_key,
            "patch_encoding": image_tokenizer_key,
        },
        task,
        observation,
        #action,
    )

    params = variables["params"]

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        metrics=Metrics.empty(),
        dropout_key=dropout_key,
        image_tokenizer_key=image_tokenizer_key,
    )

