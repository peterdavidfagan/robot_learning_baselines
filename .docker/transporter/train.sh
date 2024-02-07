#!/bin/bash

# activate built poetry env
cd /app
source $(poetry env info --path)/bin/activate

# login to wandb
wandb login

# execute training command
cd /app
exec "$@"
