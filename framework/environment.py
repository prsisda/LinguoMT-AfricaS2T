from __future__ import annotations
import sys
import torch


def detect_environment() -> dict:
    try:
        from google.colab import drive as _drive
        in_colab = True
    except Exception:
        _drive = None
        in_colab = False

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    return {
        "in_colab": in_colab,
        "run_local": not in_colab,
        "_colab_drive": _drive,
        "device": device,
        "cuda_available": torch.cuda.is_available(),
    }


def install_colab_dependencies(extras: list[str] | None = None) -> None:
    import subprocess
    base = [
        "transformers>=4.40", "datasets", "sacrebleu", "librosa", "soundfile",
        "sentencepiece", "accelerate", "jiwer", "pandas==2.2.2", "pyarrow>=15.0.0",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "-q", "install", "-U"] + base + (extras or []),
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "-q", "install", "torchcodec",
         "--extra-index-url", "https://download.pytorch.org/whl/cu121"],
        check=False,
    )
    print("Dependencies installed.")


def mount_google_drive(drive_module) -> bool:
    if drive_module is None:
        return False
    try:
        drive_module.mount("/content/drive")
        return True
    except Exception as e:
        print(f"Drive mount skipped: {e}")
        return False
