from huggingface_hub import snapshot_download
import os
from pathlib import Path


ASR_REPO_ID = "Sunbird/sunbird-asr"
asr_model_dir = Path(os.getcwd()) / "models/sunbird-asr"

snapshot_download(repo_id=ASR_REPO_ID, local_dir=asr_model_dir)

