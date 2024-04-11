#!/bin/bash
poetry shell

# config
experiment_name="performance_benchmark"
notes="performance benchmark"
tags="performance benchmark" 
parent_dir=$(realpath ..)
cd ${parent_dir}

# Run basic performance benchmarks
python perf_bench_mutli_modal.py +config=octo-categorical config.wandb.experiment_name="octo_categorical_${experiment_name}"
python perf_bench_mutli_modal.py +config=octo-categorical-compressed config.wandb.experiment_name="octo_categorical_${experiment_name}"
