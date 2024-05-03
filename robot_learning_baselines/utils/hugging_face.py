"""Utilities for interfacing with hugging face."""
import os
from glob import glob
from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from huggingface_hub.repocard import metadata_eval_result, metadata_save

def push_model(
        branch: str,
        checkpoint_dir: str,
        entity: str = "peterdavidfagan",
        repo_name: str = "robot_learning_baselines",
        ):
    """
    Uploads model to hugging face repository.
    """
    api = HfApi()
    
    repo_id = f"{entity}/{repo_name}"
    repo_url = api.create_repo(
        repo_id=repo_id,
        exist_ok=True,
        private=False,
    )

    # generate model card
    model_card = f"""
# (Robot Learning Baselines) Test**

OMG what a great model this is.
    """

    # operations to upload flax model checkpoint
    #operations=[]
    #def compile_model_upload_ops(src_path):
    #        if os.path.isfile(src_path):
    #            print(src_path)
    #            dest_path = src_path.replace(checkpoint_dir, "")
    #            print(dest_path)
    #            operations.append(CommitOperationAdd(path_in_repo=dest_path, path_or_fileobj=src_path))
    #        else:
    #            for item in os.listdir(src_path + "/"):
    #                item = os.path.join(src_path, item)
    #                if os.path.isfile(item):
    #                    print(item)
    #                    dest_path = src_path.replace(checkpoint_dir, "")
    #                    print(dest_path)
    #                    operations.append(CommitOperationAdd(path_in_repo=dest_path, path_or_fileobj=item))
    #                else:
    #                    compile_model_upload_ops(item)
    #compile_model_upload_ops(checkpoint_dir)
    
    #for filepath in glob(checkpoint_dir + "/**/*", recursive=True):
    #    if os.path.isfile(filepath):
    #        operations.append(CommitOperationAdd(path_in_repo="/", path_or_fileobj=filepath))

    # create model branch
    api.create_branch(
            repo_id=repo_id, 
            branch=branch,
            repo_type="model",
            exist_ok=True, 
            )
    if os.path.isdir(checkpoint_dir):
        api.upload_folder(
                folder_path=checkpoint_dir,
                repo_id=repo_id, 
                repo_type="model",
                multi_commits=True,
                )
    elif os.path.isfile(checkpoint_dir):
        api.upload_file(
            path_or_fileobj=checkpoint_dir,
            path_in_repo="checkpoint.tar.xz", 
            repo_id=repo_id, 
            repo_type="model",
        )
    else:
        raise NotImplementedError

    # commit changes to branch
    #api.create_commit(
    #        repo_id=repo_id,
    #        commit_message="Nice Model Dude",
    #        operations=operations,
    #        repo_type="model",
    #        )
