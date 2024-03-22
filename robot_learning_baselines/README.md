
# Local Setup 

## Downloading Datasets

### OXE 

### Tranporter Network Datasets

## Running Individual Experiments

All experiments are tracked on Weights and Biases in the [robot_learning_baselines](https://wandb.ai/ipab-rad/robot_learning_baselines) project.

To train a multi modal model run:

```bash
python train_multi_modal.py +config=<config_filename>
```

To train transporter network run:

```bash
python train_transporter.py +config=<config_filename>
```

## Running Hyperparameter Sweep

```bash
cd hyper_parameter_tuning
wandb sweep <sweep.yaml>
wandb agent <wandb generated serial code>
```

# Kubernetes Setup

Details to be added.

# Deploying Models
Details for deploying models using the [ros2_robotics_research_toolkit](https://github.com/peterdavidfagan/ros2_robotics_research_toolkit) to be added in future.
