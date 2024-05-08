#!/bin/bash
poetry shell

# config
parent_dir=$(realpath ..)
cd ${parent_dir}

# Run basic performance benchmarks
python train_transporter +config=transporter-real
python train_transporter +config=transporter-sim

# Upload models to huggingface
cd hf_scripts
python hf_upload_transporter.py +config=transporter-real
python hf_upload_transporter.py +config=transporter-sim
