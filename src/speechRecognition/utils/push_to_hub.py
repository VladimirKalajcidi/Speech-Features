import argparse
from huggingface_hub import HfApi, create_repo, login


parser = argparse.ArgumentParser(description="app")
parser.add_argument("-m", "--model", help="model path", required=True)
parser.add_argument("-d", "--directory", help="hub directory", required=True)
args = parser.parse_args()

login()

create_repo(
    repo_id=args.directory,
    repo_type="model"
)

api = HfApi()
api.upload_folder(
    folder_path=args.model,
    repo_id=args.directory,
    repo_type="model",
)

api.upload_file(
    path_or_fileobj="tokenizer.json",
    path_in_repo="tokenizer.json",
    repo_id=args.directory,
    repo_type="model",
)