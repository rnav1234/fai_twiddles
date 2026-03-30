#
# uvx --from huggingface_hub hf download yunmorning/broken-model --local-dir ./broken --max-workers 1
# ^^ was still exhausting all my disk space on Chromebook. Gemini suggested a Python wrapper to filter all 
# repo files during a download...
#
import argparse
import os
from huggingface_hub import HfApi, snapshot_download

parser = argparse.ArgumentParser(
        description='FriendliAI EP tester',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d','--dir', required=True, help='local directory to write into')
parser.add_argument('-r','--repo', required=True, help='HF repo name')
parser.add_argument('-m','--max', type=int, help='max file size in GBs')
args = parser.parse_args()

repo_id = args.repo  # "yunmorning/broken-model"
local_dir = args.dir # "./broken"
gbs = args.max or 1 # default 1GB
max_size_bytes = gbs * 1024 * 1024 * 1024  
#max_size_bytes = 1 * 1024 * 1024 * 1024  # 1GB

print(f"{repo_id=} {local_dir=} {max_size_bytes=}")

api = HfApi()

# 1. List all files in the repository
repo_info = api.repo_info(repo_id=repo_id, files_metadata=True)

# 2. Filter for files smaller than 1GB
allowed_files = [
    f.rfilename for f in repo_info.siblings 
    if f.size is not None and f.size < max_size_bytes
]

print(f"Found {len(allowed_files)} files under 1GB. Starting download...")

# 3. Download only the allowed files
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    allow_patterns=allowed_files,
    max_workers=1  # Recommended for Chromebook stability
)
