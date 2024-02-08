"""
Utilities for downloading and loading the OpenX dataset.

adapted from: https://github.com/google-deepmind/open_x_embodiment/blob/main/colabs/Open_X_Embodiment_Datasets.ipynb
"""

import os
import tqdm
import subprocess
import argparse

import tensorflow as tf
import tensorflow_datasets as tfds

# OpenX is composed of multiple datasets. We provide a list of all the datasets
DATASETS = [
    'fractal20220817_data',
    'kuka',
    'bridge',
    'taco_play',
    'jaco_play',
    'berkeley_cable_routing',
    'roboturk',
    'nyu_door_opening_surprising_effectiveness',
    'viola',
    'berkeley_autolab_ur5',
    'toto',
    'language_table',
    'columbia_cairlab_pusht_real',
    'stanford_kuka_multimodal_dataset_converted_externally_to_rlds',
    'nyu_rot_dataset_converted_externally_to_rlds',
    'stanford_hydra_dataset_converted_externally_to_rlds',
    'austin_buds_dataset_converted_externally_to_rlds',
    'nyu_franka_play_dataset_converted_externally_to_rlds',
    'maniskill_dataset_converted_externally_to_rlds',
    'cmu_franka_exploration_dataset_converted_externally_to_rlds',
    'ucsd_kitchen_dataset_converted_externally_to_rlds',
    'ucsd_pick_and_place_dataset_converted_externally_to_rlds',
    'austin_sailor_dataset_converted_externally_to_rlds',
    'austin_sirius_dataset_converted_externally_to_rlds',
    'bc_z',
    'usc_cloth_sim_converted_externally_to_rlds',
    'utokyo_pr2_opening_fridge_converted_externally_to_rlds',
    'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds',
    'utokyo_saytap_converted_externally_to_rlds',
    'utokyo_xarm_pick_and_place_converted_externally_to_rlds',
    'utokyo_xarm_bimanual_converted_externally_to_rlds',
    'robo_net',
    'berkeley_mvp_converted_externally_to_rlds',
    'berkeley_rpt_converted_externally_to_rlds',
    'kaist_nonprehensile_converted_externally_to_rlds',
    'stanford_mask_vit_converted_externally_to_rlds',
    'tokyo_u_lsmo_converted_externally_to_rlds',
    'dlr_sara_pour_converted_externally_to_rlds',
    'dlr_sara_grid_clamp_converted_externally_to_rlds',
    'dlr_edan_shared_control_converted_externally_to_rlds',
    'asu_table_top_converted_externally_to_rlds',
    'stanford_robocook_converted_externally_to_rlds',
    'eth_agent_affordances',
    'imperialcollege_sawyer_wrist_cam',
    'iamlab_cmu_pickup_insert_converted_externally_to_rlds',
    'uiuc_d3field',
    'utaustin_mutex',
    'berkeley_fanuc_manipulation',
    'cmu_play_fusion',
    'cmu_stretch',
    'berkeley_gnm_recon',
    'berkeley_gnm_cory_hall',
    'berkeley_gnm_sac_son'
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


# dataloader is incomplete only used for debugging for now
def create_dataloader():
    """Convert RLDS datasets to format for training RT-X models."""
    # RLDS utility functions
    def episode2steps(episode):
        return episode['steps']
    def step_map_fn(step):
      return {
          'observation': {
              'natural_language_instruction': step['observation']['natural_language_instruction'],
              'image': tf.image.resize(step['observation']['image'], (128, 128)),
          },
          'action': tf.concat([
              step['action']['future/xyz_residual'],
          ], axis=-1)
      }

    # load rlds dataset
    dataset = tfds.load(
        'bc_z:0.1.0', 
        data_dir='/mnt/hdd/openx_datasets',
        split='train[:10]'
        )

    # convert to steps
    dataset = dataset.map(
            episode2steps,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
            ).flat_map(lambda x: x)


    # convert steps to required format
    dataseet = dataset.map(step_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(1000).batch(32).prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='RLDS dataset loader')
    parser.add_argument(
            "--data_transfer", 
            type=bool, 
            help="whether to download the datasets or not", 
            default=False)
    parser.add_argument(
            "--data_dir", 
            type=str, 
            help="local directory to store the datasets", 
            default='/mnt/hdd/openx_datasets')
    args = parser.parse_args()

    if args.data_transfer:

        # for now we care about datasets with Franka Emika Panda robot and natural language instructions
        df = pd.read_excel(
            "./artifacts/open-x-embodiment.xlsx",
            skiprows=14
        )

        # filter for franka + language annotations
        df = df[
            (df["Robot"]=="Franka") & 
            (df["Language Annotations"].notna()) &
            (df["# RGB Cams"] > 0)
            ]

        DATASETS = df["Registered Dataset Name"].to_list()
        LOCAL_MOUNT = args.data_dir
        download_datasets(DATASETS, LOCAL_MOUNT)
    else:
        # local config
        TEST_DATASETS = ['bc_z']
        LOCAL_MOUNT = '/mnt/hdd/openx_datasets'

        # try loading a dataset
        dataset = create_dataloader()
    
        # inspect the dataset
        for batch in dataset:
            print(batch)
            break
