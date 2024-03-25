[![docs](https://github.com/peterdavidfagan/robot_learning_baselines/actions/workflows/pages.yaml/badge.svg)](https://github.com/peterdavidfagan/robot_learning_baselines/blob/main/.github/workflows/pages.yaml)
[![open_x_embodiment_data_transfer](https://github.com/peterdavidfagan/robot_learning_baselines/actions/workflows/open-x-embodiment-data-transfer.yaml/badge.svg)](https://github.com/peterdavidfagan/robot_learning_baselines/blob/main/.github/workflows/open-x-embodiment-data-transfer.yaml)
[![experiments](https://img.shields.io/badge/wandb-experiments?style=flat&labelColor=%2300000&color=%23FFFF00)](https://wandb.ai/ipab-rad/robot_learning_baselines)

# Robot Learning Baselines
[**[Training Runs]**](https://wandb.ai/ipab-rad/robot_learning_baselines) &ensp; [**[Pretrained Models]**](https://github.com/peterdavidfagan/robot_learning_baselines/tree/main) &ensp; [**[Documentation]**](https://peterdavidfagan.com/robot_learning_baselines/) &ensp;

A set of baseline models for learning from demonstration with supporting training/evaluation scripts.

<img src="./assets/robot_learning.jpeg" height=300/>


# Configuring your Local Development Environment

The project currently only supports Linux OS, this is a direct consequence of some of the project dependencies. This may be revised in future.

This repository manages certain dependencies as submodules, to ensure you have cloned the requires submodules run:

```bash
git submodule update --init --recursive
```

With submodules installed, it should be possible to build the Python virtual environment on machines running Linux OS via poetry with the following command:

```bash
poetry install
```
