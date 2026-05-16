#!/usr/bin/env python3
# Patch LinguoMT FLEURS + SeamlessM4T script so it runs safely in VS Code/local Python
# and still keeps Google Drive backup when executed inside Google Colab.
#
# Usage from your project root:
#     python patch_linguomt_vscode.py notebooks/LinguoMT_FLEURS_SeamlessM4T_FULL_RUN_All_Supported_African_English.py
#
# It creates:
#     notebooks/LinguoMT_FLEURS_SeamlessM4T_FULL_RUN_All_Supported_African_English_VSCODE_READY.py

from __future__ import annotations

import re
import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        print(f"[WARN] Could not find exact block for: {label}")
        return text
    return text.replace(old, new, 1)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python patch_linguomt_vscode.py path/to/script.py")
        sys.exit(1)

    src = Path(sys.argv[1])
    if not src.exists():
        raise FileNotFoundError(f"Input script not found: {src}")

    text = src.read_text(encoding="utf-8")

    # 1) Make plotting safe in local VS Code/terminal execution.
    if "import sys" not in text:
        text = text.replace("import time\nfrom pathlib import Path", "import time\nimport sys\nfrom pathlib import Path", 1)

    old_import = "import sacrebleu\nimport matplotlib.pyplot as plt\nfrom jiwer import wer, cer"
    new_import = '''import sacrebleu

# VS Code/local-script compatibility:
# Use a non-interactive backend outside Colab so plt.show() does not block execution.
import matplotlib
if "google.colab" not in sys.modules:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from jiwer import wer, cer'''
    text = replace_once(text, old_import, new_import, "matplotlib local backend")

    old_save_plot = '''def save_plot(path):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    print("Saved figure:", path)'''

    new_save_plot = '''def save_plot(path):
    # Save a matplotlib figure safely in both Colab and local VS Code scripts.
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    if globals().get("IN_COLAB", False):
        plt.show()
    plt.close()
    print("Saved figure:", path)'''

    text = replace_once(text, old_save_plot, new_save_plot, "save_plot local-safe version")

    text = text.replace(
        '''plt.savefig(fig_path, dpi=200)
    plt.show()
    print("Saved:", fig_path)''',
        '''plt.savefig(fig_path, dpi=200)
    if globals().get("IN_COLAB", False):
        plt.show()
    plt.close()
    print("Saved:", fig_path)'''
    )

    # 2) Ensure Google Colab / Drive imports are optional locally.
    drive_block_pattern = re.compile(
        r"from pathlib import Path\nfrom datetime import datetime\nimport shutil\nimport pandas as pd\n\n"
        r"IN_COLAB = False\n\n"
        r"try:\n    from google\.colab import drive\n    IN_COLAB = True\nexcept ImportError:\n    IN_COLAB = False\n\n"
        r"if IN_COLAB:\n    drive\.mount\(\"/content/drive\"\)\n    GOOGLE_DRIVE_AVAILABLE = True\n    print\(\"Google Drive mounted\.\"\)\nelse:\n    GOOGLE_DRIVE_AVAILABLE = False\n    print\(\"Not running in Google Colab\. Google Drive backup will be skipped locally\.\"\)",
        flags=re.MULTILINE,
    )

    drive_block_new = '''from pathlib import Path
from datetime import datetime
import shutil
import pandas as pd

# Local/Colab compatibility:
try:
    from google.colab import drive
    IN_COLAB = True
except Exception:
    drive = None
    IN_COLAB = False

if IN_COLAB:
    drive.mount("/content/drive")
    GOOGLE_DRIVE_AVAILABLE = True
    print("Google Drive mounted.")
else:
    GOOGLE_DRIVE_AVAILABLE = False
    print("Not running in Google Colab. Google Drive backup will be skipped locally.")'''

    text, n = drive_block_pattern.subn(drive_block_new, text, count=1)
    if n == 0:
        print("[WARN] Could not patch Google Drive mount block; it may already be patched.")

    # 3) Guard final Google Drive backup so local VS Code run does not try /content/drive.
    start_marker = "# =========================================================\n# Save to Google Drive\n# ========================================================="
    end_marker = "# %% [markdown] id=\"01fb7c65\""
    start = text.find(start_marker)
    end = text.find(end_marker, start)

    final_drive_new = '''# =========================================================
# Save to Google Drive
# =========================================================
# This runs only in Google Colab. In VS Code/local Python, results stay local.

if globals().get("GOOGLE_DRIVE_AVAILABLE", False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    DRIVE_BACKUP_ROOT = Path(
        "/content/drive/MyDrive/" + BASE_OUTPUT_FOLDER_NAME
    )
    DRIVE_BACKUP_ROOT.mkdir(parents=True, exist_ok=True)
    DRIVE_BACKUP_DIR = DRIVE_BACKUP_ROOT / f"experiment_{timestamp}"

    if DRIVE_BACKUP_DIR.exists():
        shutil.rmtree(DRIVE_BACKUP_DIR)

    shutil.copytree(BASE_OUTPUT_DIR, DRIVE_BACKUP_DIR)

    print("=================================================")
    print("Saved complete experiment backup to Google Drive")
    print(DRIVE_BACKUP_DIR)
    print("=================================================")

    zip_path = shutil.make_archive(
        str(DRIVE_BACKUP_DIR),
        'zip',
        root_dir=BASE_OUTPUT_DIR
    )

    print("ZIP backup created:")
    print(zip_path)
else:
    print("Google Drive backup skipped because this script is running outside Google Colab.")
    print("Local results are available at:", BASE_OUTPUT_DIR)
'''

    if start != -1 and end != -1:
        text = text[:start] + final_drive_new + "\n" + text[end:]
    else:
        print("[WARN] Could not patch final Google Drive backup block; it may already be patched or formatted differently.")

    # 4) Add robust audio extraction and use it in audio translation/prep.
    if "def extract_audio_array(" not in text:
        insert_after = '''def audio_duration(audio, sr=TARGET_SR):
    audio = np.asarray(audio, dtype=np.float32).flatten()
    return len(audio) / float(sr)
'''
        helper = r'''

def extract_audio_array(audio_object_or_array, target_sr=TARGET_SR):
    # Return a flat float32 waveform from a Hugging Face audio object or raw array.
    if isinstance(audio_object_or_array, dict):
        sr = audio_object_or_array.get("sampling_rate", target_sr)
        audio = audio_object_or_array.get("array")
    else:
        sr = target_sr
        audio = audio_object_or_array

    if audio is None:
        return np.zeros(int(MIN_DURATION * target_sr), dtype=np.float32)

    try:
        audio = np.asarray(audio, dtype=np.float32)
    except Exception:
        audio = np.array(audio, dtype=object)

    if getattr(audio, "dtype", None) == object:
        flattened_parts = []
        for part in audio:
            flattened_parts.append(np.asarray(part, dtype=np.float32).flatten())
        audio = np.concatenate(flattened_parts) if flattened_parts else np.array([], dtype=np.float32)

    audio = np.asarray(audio, dtype=np.float32).flatten()
    audio = np.nan_to_num(audio)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr).astype(np.float32)

    return ensure_min_audio_length(audio, MIN_DURATION, target_sr)
'''
        text = replace_once(text, insert_after, insert_after + helper, "extract_audio_array helper")

    trans_pattern = re.compile(
        r"def translate_audio_direct\(audio_object_or_array, source_lang_code, target_lang_code, normalize_audio=True, improved=True\):\n"
        r'    """Translate one audio input with SeamlessM4T-v2\.\n\n'
        r"    This is the core speech-to-text translation call\. Optional trimming and\n"
        r"    normalization are controlled by the optimization strategy wrapper\.\n"
        r'    """\n'
        r".*?"
        r"    return processor\.decode\(\n"
        r"        generated_tokens\[0\]\.tolist\(\)\[0\],\n"
        r"        skip_special_tokens=True,\n"
        r"    \)\n",
        flags=re.DOTALL,
    )

    trans_new = '''def translate_audio_direct(audio_object_or_array, source_lang_code, target_lang_code, normalize_audio=True, improved=True):
    # Translate one audio input with SeamlessM4T-v2.
    audio = extract_audio_array(audio_object_or_array, TARGET_SR)

    if improved:
        audio = trim_silence(audio)

    if normalize_audio:
        audio = normalize_audio_waveform(audio)

    audio = ensure_min_audio_length(audio, MIN_DURATION, TARGET_SR)

    inputs = processor(
        audio=audio,
        sampling_rate=TARGET_SR,
        src_lang=source_lang_code,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            tgt_lang=target_lang_code,
            generate_speech=False,
        )

    return processor.decode(
        generated_tokens[0].tolist()[0],
        skip_special_tokens=True,
    )
'''
    text, n = trans_pattern.subn(trans_new, text, count=1)
    if n == 0:
        print("[WARN] Could not patch translate_audio_direct; check manually.")

    prep_pattern = re.compile(
        r"def prepare_audio_for_strategy\(audio_object_or_array, strategy\):\n"
        r'    """Apply the selected optimization strategy while keeping the evaluation schema stable\."""\n'
        r".*?"
        r"    return audio\.astype\(np\.float32\)\n",
        flags=re.DOTALL,
    )
    prep_new = '''def prepare_audio_for_strategy(audio_object_or_array, strategy):
    # Apply the selected optimization strategy while keeping the evaluation schema stable.
    audio = extract_audio_array(audio_object_or_array, TARGET_SR)

    if strategy.get("trim", False):
        audio = trim_silence(audio)

    if strategy.get("normalize", False):
        audio = normalize_audio_waveform(audio)

    audio = ensure_min_audio_length(audio, MIN_DURATION, TARGET_SR)
    return audio.astype(np.float32)
'''
    text, n = prep_pattern.subn(prep_new, text, count=1)
    if n == 0:
        print("[WARN] Could not patch prepare_audio_for_strategy; check manually.")

    out = src.with_name(src.stem + "_VSCODE_READY.py")
    out.write_text(text, encoding="utf-8")

    print(f"[OK] Wrote patched script: {out}")
    print("Run it with:")
    print(f"    python {out}")


if __name__ == "__main__":
    main()
