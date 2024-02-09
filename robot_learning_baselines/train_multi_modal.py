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

# experiment tracking
import wandb
import hydra
from omegaconf import DictConfig
from clu import metrics

# training utilities
from utils.pipeline import (
    load_dataset,
    setup_checkpointing,
)

from utils.decoder_only import (
    bc_train_step,
    create_train_state,
    compute_metrics,
    compute_saliency_maps,
)

from utils.wandb import (
    init_wandb,
    # create_dataset_artifact,
    create_model_artifact,
    log_dataset_batch,
    track_gradients,
)

from utils.visualization import (
    sample_eval_puzzle,
    visualize_rollout,
)

from evaluate_decoder_only import evaluate_puzzles

# model architecture
from multi_modal_transformers.decoder_only import DecoderOnly

# logging
from utils.logger import get_logger

LOG = get_logger(__name__)


@hydra.main(version_base=None, config_path="./config")
def main(cfg: DictConfig) -> None:
    """Model training loop."""
    # set up jax random number generator
    key = random.PRNGKey(0)
    key, model_key, dropout_key, image_tokenizer_key = random.split(key, 4)

    # variables to track training progress
    BEST_MODEL = None
    TEST_LOSS = np.inf

    # load the dataset
    train_data, test_data = load_dataset(cfg.data.open-x-embodiment, cfg.training.decoder_only)

    # setup experiment checkpointing for concept learner model
    chkpt_manager = setup_checkpointing(cfg.training.decoder_only)

    # instantiate model
    batch = next(train_data.as_numpy_iterator())
    decoder_only = DecoderOnly.initialize_from_config(cfg=cfg.model.decoder_only)

    # instantiate model optimizer
    learning_rate_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=cfg.training.decoder_only.initial_lr,
        peak_value=cfg.training.decoder_only.peak_lr,
        warmup_steps=cfg.training.decoder_only.warmup_epochs * cfg.training.decoder_only.steps_per_epoch,
        decay_steps=(cfg.training.decoder_only.num_epochs - cfg.training.decoder_only.warmup_epochs)
        * cfg.training.decoder_only.steps_per_epoch,
        end_value=cfg.training.decoder_only.end_lr,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training.decoder_only.max_grad_norm),
        optax.adamw(learning_rate_scheduler, weight_decay=cfg.training.decoder_only.weight_decay),
    )

    # set up weights and biases
    if cfg.wandb.use:
        if cfg.wandb.resume:
            tables = init_wandb(cfg, resume=True)
        else:
            tables = init_wandb(cfg)
            # TODO: consider logic to account or local vs hosted datasets
            # create_dataset_artifact(cfg.data.decoder_only)
            log_dataset_batch(cfg.data.decoder_only, batch)

    # instantiate training state
    train_state = create_train_state(
        batch["concept"],
        batch["trajectory"]["frame"],
        batch["trajectory"]["action"],
        model_key,
        dropout_key,
        image_tokenizer_key,
        decoder_only,
        optimizer,
        cfg,
    )

    # inspect model before training
    print(
        decoder_only.tabulate(
            {"params": model_key, "dropout": dropout_key, "patch_encoding": image_tokenizer_key},
            batch["concept"],
            batch["trajectory"]["frame"],
            batch["trajectory"]["action"],
        )
    )

    BEST_MODEL = train_state.params  # initialise best model

    # load previous checkpoint
    if cfg.wandb.resume:
        # download last saved checkpoint from wandb
        art = wandb.use_artifact(
            "ipab-rad/visual_concept_planner/"
            + cfg.wandb.experiment_name
            + f"-{cfg.wandb.resume_run.load_epoch}:latest"
        )
        checkpoint_path = os.path.join(cfg.training.checkpoint_dir, str(cfg.wandb.resume_run.load_epoch))
        art_dir = art.download(checkpoint_path)

        # restore train state from checkpoint
        train_state = chkpt_manager.restore(art_dir, items=train_state)

    # training loop
    start = 0 if not cfg.wandb.resume else (train_state.step // cfg.training.steps_per_epoch)
    for epoch in range(start, cfg.training.decoder_only.num_epochs):

        # shuffle dataset and create iterator
        train_data = train_data.shuffle(10)

        train_data_iter = train_data.as_numpy_iterator()
        test_data_iter = test_data.as_numpy_iterator()

        # epoch metrics
        metrics_history = {
            "decoder_only_train_loss": [],
            "decoder_only_test_loss": [],
            "decoder_only_train_meta_loss": [],
            "decoder_only_test_meta_loss": [],
        }

        # training
        for batch in train_data_iter:
            train_state, grads = bc_train_step(
                train_state,
                batch["concept"],
                batch["trajectory"]["frame"],
                batch["trajectory"]["action"],
                batch["target_action"],
                dropout_key,
                image_tokenizer_key,
            )

            train_state, _, _ = compute_metrics(
                train_state,
                batch["concept"],
                batch["trajectory"]["frame"],
                batch["trajectory"]["action"],
                batch["target_action"],
                dropout_key,
                image_tokenizer_key,
                "train",
            )

        for metric, value in train_state.metrics.compute().items():
            metrics_history[f"decoder_only_train_{metric}"].append(value)

        print(f"Epoch {epoch} train loss: {metrics_history['decoder_only_train_loss'][-1]}")

        # reset metrics for next epoch
        train_state = train_state.replace(metrics=train_state.metrics.empty())

        # save checkpoint
        chkpt_manager.save(epoch, train_state)
        if epoch % cfg.training.decoder_only.save_interval == 0 and epoch != 0 and cfg.wandb.use:
            create_model_artifact(cfg.model, cfg.training.decoder_only, str(epoch))

        # evaluate on test dataset
        test_state = train_state
        num_test_batches = len(test_data)
        for idx, batch in enumerate(test_data_iter):
            test_state, logits, target = compute_metrics(
                test_state,
                batch["concept"],
                batch["trajectory"]["frame"],
                batch["trajectory"]["action"],
                batch["target_action"],
                dropout_key,
                image_tokenizer_key,
                "test",
            )

        for metric, value in test_state.metrics.compute().items():
            metrics_history[f"decoder_only_test_{metric}"].append(value)

        print(f"Epoch {epoch} test loss: {metrics_history['decoder_only_test_loss'][-1]}")

        # check if loss is lower than previous best
        if metrics_history["decoder_only_test_loss"][-1] < TEST_LOSS:
            BEST_MODEL = train_state.params
            TEST_LOSS = metrics_history["decoder_only_test_loss"][-1]


    if cfg.wandb.use:
        # log final metrics
        wandb.log({"Evaluation Predictions Table": tables["eval_prediction_table"]})
        wandb.log({"Evaluation Visualizations Table": tables["eval_visualisation_table"]})
        wandb.log({"Train Visualizations Table": tables["train_visualisation_table"]})
        if cfg.wandb.track_gradients:
            wandb.log({"Gradient Table": tables["gradient_table"]})

        # log best model as artifact
        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(os.path.join(cfg.training.decoder_only.checkpoint_dir, "best_model"), BEST_MODEL)
        create_model_artifact(cfg.model, cfg.training.decoder_only, "best_model")


if __name__ == "__main__":
    main()

