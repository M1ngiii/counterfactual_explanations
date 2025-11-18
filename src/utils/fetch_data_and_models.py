from pathlib import Path
from huggingface_hub import hf_hub_download
import shutil
import subprocess
import urllib

ROOT = Path(__file__).resolve().parents[2]

# Data paths
DATA_DEST_DIR = ROOT / "data" / "CounterEval"
MODEL_DEST_DIR = ROOT / "data" / "models" / "trained_3.1.8b"

# CounterEval paths
GITHUB_REPO = "https://github.com/anitera/CounterEval.git" 
GITHUB_MODEL_SUBDIR = Path("models") / "trained_3.1.8b"

# Adult dataset paths (Carla)
ADULT_DEST_DIR = ROOT / "data" / "adult"
ADULT_FILENAME = "adult.csv"
ADULT_URL = (
    "https://raw.githubusercontent.com/carla-recourse/cf-data/"
    "master/data/adult/preprocessed/adult.csv"
)

FILES = [
    "cleaned_cf_dataset.csv",
    "full_cf_dataset.csv",
    "participant_background.csv",
    "survey_question.csv",
]

def download_countereval_dataset(filename: str):
    """Download one CounterEval CSV from HuggingFace"""
    DATA_DEST_DIR.mkdir(parents=True, exist_ok=True)
    dest_path = DATA_DEST_DIR / filename

    if dest_path.exists():
        print(f"[skip] {filename} already exists at {dest_path}")
        return

    cache_path = hf_hub_download(
        repo_id="anitera/CounterEval",
        filename=filename,
        repo_type="dataset",
    )
    shutil.copy2(cache_path, dest_path)
    print(f"[ok] Downloaded {filename} to {dest_path}")


def download_model():
    """Download fine-tuned LLaMA 3.1 8B model"""
    # If model dir exists and is non-empty, assume it's already there
    if MODEL_DEST_DIR.exists() and any(MODEL_DEST_DIR.iterdir()):
        print(f"[skip] Model directory already populated at {MODEL_DEST_DIR}")
        return

    tmp_repo = ROOT / ".cache" / "CounterEval_copy"
    tmp_repo.parent.mkdir(parents=True, exist_ok=True)

    if not tmp_repo.exists():
        print("[info] Cloning CounterEval repo...")
        subprocess.run(["git", "clone", GITHUB_REPO, str(tmp_repo)], check=True)
        subprocess.run(["git", "lfs", "install"], check=True)
        subprocess.run(["git", "lfs", "pull"], cwd=tmp_repo, check=True)

    src_dir = tmp_repo / GITHUB_MODEL_SUBDIR
    if not src_dir.exists():
        raise FileNotFoundError(f"Could not find model directory at {src_dir}")

    MODEL_DEST_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copytree(src_dir, MODEL_DEST_DIR, dirs_exist_ok=True)
    shutil.rmtree(tmp_repo, ignore_errors=True)
    print(f"[ok] Copied CounterEval model to {MODEL_DEST_DIR}")


def main():
    # 1. Download CounterEval CSVs
    for fname in FILES:
        download_countereval_dataset(fname)

    # 2. Download the model
    download_model()

if __name__ == "__main__":
    main()