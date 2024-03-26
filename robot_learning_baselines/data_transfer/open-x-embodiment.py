"""
Utilities for downloading and loading the OpenX dataset.

adapted from: https://github.com/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb
"""

import os
import pathlib
import tqdm
import subprocess
import argparse
import pandas as pd
from functools import partial

import jax
import tensorflow as tf
import tensorflow_datasets as tfds

from transformers import FlaxT5EncoderModel, AutoTokenizer, AutoConfig
from multi_modal_transformers.tokenizers.text.t5_base import T5Tokenizer

import hydra
from hydra.utils import instantiate, call
from omegaconf import DictConfig

from robot_learning_baselines.utils.data import (
    oxe_load_single_dataset,
    oxe_load_dataset,
    preprocess_batch,
)


MISSING_DATASETS = [
        "fractal20220817_data",
        "furniture_bench_dataset_converted_externally_to_rlds",
        "cmu_playing_with_food",
    ]

def dataset2version(dataset_name):
  if dataset_name == 'robo_net':
    version = '1.0.0'
  elif dataset_name == 'language_table':
    version = '0.0.1'
  else:
    version = '0.1.0'
  return f'{dataset_name}/{version}'

def download_datasets(datasets, data_dir):
  """Download a dataset from OpenX."""
  os.makedirs(data_dir, exist_ok=True)
  for dataset in tqdm.tqdm(datasets):
    dataset_version = dataset2version(dataset)
    local_dataset_dir = os.path.join(data_dir, dataset)
    os.makedirs(local_dataset_dir, exist_ok=True)
    subprocess.run(['gsutil', '-m', 'cp', '-r', f'gs://gresearch/robotics/{dataset_version}/', local_dataset_dir], check=True)

@hydra.main(version_base=None, config_path="..")
def main(cfg: DictConfig) -> None:
    """Raw data download, preprocessing and saving of preprocessed data."""
    
    cfg = cfg["config"]

    # for now we care about datasets with Franka Emika Panda robot and natural language instructions
    # update to config similar to OCTO OXE
    path = pathlib.Path(__file__).parent.absolute()
    df = pd.read_excel(
        f"{path}/artifacts/open-x-embodiment.xlsx",
        skiprows=14
    )

    # filter for specific datasets
    df = df[
        (df["Robot"]=="Franka") & 
        (df["Language Annotations"].notna()) &
        (df["# RGB Cams"] > 0) &
        (df["Registered Dataset Name"].notna())
        ]

    DATASETS = df["Registered Dataset Name"].to_list()
    DATASETS = [d for d in DATASETS if d not in MISSING_DATASETS]
    download_datasets(DATASETS, cfg.dataset.tfds_data_dir)


    ## TODO: improve the below section

    # read dataset
    ds = oxe_load_single_dataset(cfg.dataset)

    # generate language embeddings with t5
    t5_tokenizer = AutoTokenizer.from_pretrained('t5-base')
    t5_tokenize_fn = partial(t5_tokenizer, 
                               return_tensors="jax", 
                               max_length=16, # hardcode while debugging
                               padding="max_length", 
                               truncation=True
                               )
    t5_model = T5Tokenizer()
    batch = next(ds.as_numpy_iterator())
    text = [task.decode("utf-8") for task in batch["task"]["language_instruction"]]
    text_tokens = t5_tokenize_fn(text)["input_ids"]
    t5_model_params = t5_model.init(jax.random.PRNGKey(0), text_tokens)
    
    def embed_instructions(instruction):
        text = list(instruction.numpy().astype(str))
        text_tokens = t5_tokenize_fn(text)["input_ids"]
        embedding = t5_model.apply(t5_model_params, text_tokens)
        return embedding
    
    def map_function(example):
        language_instruction = example["task"]["language_instruction"]
        embedding = tf.py_function(embed_instructions, [language_instruction], tf.float32)
        example["task"]["t5_language_embedding"] = embedding
        return example
    
    ds = ds.map(map_function)

    # save dataset with language embeddings to disk
    ds.save(cfg.dataset.save_dir)

if __name__=="__main__":
    main()

    
