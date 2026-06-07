# Important libraries
import os
import requests
from huggingface_hub import login,HfApi,create_repo
from huggingface_hub.utils import RepositoryNotFoundError,HfHubHTTPError

# Login credentials and hugging face repository

repo_id = "Lokeshnathy/boston-housing-dataset"              

repo_type = "dataset"
api=HfApi(token=os.getenv("HF_TOKEN"))
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id,repo_type=repo_type,private=False)
    print(f"Space '{repo_id}' created.")
api.upload_folder(
    folder_path="data",
    repo_id=repo_id,
    repo_type=repo_type)
