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

import tensorflow as tf
import tensorflow_datasets as tfds


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

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='RLDS data transfer')
    # TODO: remove data transfer argument
    parser.add_argument(
            "--data_transfer", 
            type=bool, 
            help="whether to download the datasets or not", 
            default=True)
    parser.add_argument(
            "--data_dir", 
            type=str, 
            help="local directory to store the datasets", 
            default='/mnt/hdd/openx_datasets')
    args = parser.parse_args()

    # for now we care about datasets with Franka Emika Panda robot and natural language instructions
    path = pathlib.Path(__file__).parent.absolute()
    df = pd.read_excel(
        f"{path}/artifacts/open-x-embodiment.xlsx",
        skiprows=14
    )

    # filter for franka + language annotations
    df = df[
        (df["Robot"]=="Franka") & 
        (df["Language Annotations"].notna()) &
        (df["# RGB Cams"] > 0) &
        (df["Registered Dataset Name"].notna())
        ]

    DATASETS = df["Registered Dataset Name"].to_list()
    DATASETS = [d for d in DATASETS if d not in MISSING_DATASETS]
    LOCAL_MOUNT = args.data_dir
    download_datasets(DATASETS, LOCAL_MOUNT)
