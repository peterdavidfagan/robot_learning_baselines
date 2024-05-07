"""Utilities for interfacing with hugging face."""
import os
from glob import glob
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub.repocard import metadata_eval_result, metadata_save

def push_model(
        upload_path: str,
        entity: str = "peterdavidfagan",
        repo_name: str = "robot_learning_baselines",
        branch: str = "main",
        ):
    """
    Uploads model to hugging face repository.
    """
    api = HfApi()
    
    # ensure repo exists
    repo_id = f"{entity}/{repo_name}"
    repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
        private=False,
    )

    # create model branch
    api.create_branch(
            repo_id=repo_id, 
            branch=branch,
            repo_type="model",
            exist_ok=True, 
            )

    # upload requested files
    if os.path.isdir(upload_path):
        api.upload_folder(
                folder_path=upload_path,
                repo_id=repo_id, 
                repo_type="model",
                multi_commits=True,
                )
    elif os.path.isfile(upload_path):
        api.upload_file(
            path_or_fileobj=upload_path,
            path_in_repo=upload_path.split("/")[-1], 
            repo_id=repo_id, 
            repo_type="model",
        )
    else:
        raise NotImplementedError
