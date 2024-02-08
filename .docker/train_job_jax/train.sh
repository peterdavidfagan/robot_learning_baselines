#!/bin/bash

# activate built poetry env
cd /app
source $(poetry env info --path)/bin/activate

# login to wandb
wandb login

# execute training command
cd /app/code_refresh/robot_learning_baselines.git/robot_learning_baselines/train_multi_modal.py
exec "$@"
