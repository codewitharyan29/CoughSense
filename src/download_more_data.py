"""
Download + auto-sort MORE cough data into data/raw/covid and data/raw/healthy.

Run this on YOUR laptop (unrestricted internet), NOT in the Claude sandbox.

Dataset: COUGHVID v3 (EPFL) — ~30,000 crowdsourced cough recordings from Zenodo.
  Zenodo record: https://zenodo.org/records/7024894

WHAT THIS DOES
  1. Downloads the COUGHVID zip from Zenodo (large — several GB, be patient)
  2. Unzips it
  3. Reads metadata_compiled.csv for COVID status labels
  4. Converts .webm/.ogg clips -> resampled, and copies a balanced subset
     into data/raw/covid and data/raw/healthy
  5. You then re-run build_dataset.py etc. exactly as before — no code changes.

REQUIREMENTS
  pip install requests tqdm pydub librosa soundfile pandas
  ffmpeg installed (you already have it)

USAGE
  python download_more_data.py --max-per-class 500

  --max-per-class N   how many clips per class to add (default 500).
                      Keep classes balanced. Start small (e.g. 300) to test.
"""

import argparse
import os
import zipfile
import shutil
import pandas as pd
from tqdm import tqdm

# ---- Config ----
ZENODO_ZIP_URL = "https://zenodo.org/records/7024894/files/public_dataset.zip?download=1"
DOWNLOAD_DIR = "../data/coughvid_download"
RAW_DIR = "../data/raw"
TARGET_SR = 22050


def download_file(url, dest):
    """Stream-download a large file with a progress bar."""
    import requests
    if os.path.exists(dest):
        print(f"Already downloaded: {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading {url}\n  -> {dest}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))


def convert_and_copy(src_path, dst_path):
    """Convert webm/ogg -> wav at target sample rate using librosa + soundfile."""
    import librosa
    import soundfile as sf
    try:
        y, _ = librosa.load(src_path, sr=TARGET_SR)
        if len(y) < TARGET_SR * 0.3:  # skip clips under 0.3s (likely empty)
            return False
        sf.write(dst_path, y, TARGET_SR)
        return True
    except Exception as e:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=500)
    args = parser.parse_args()

    zip_path = os.path.join(DOWNLOAD_DIR, "coughvid.zip")
    extract_dir = os.path.join(DOWNLOAD_DIR, "extracted")

    # 1. Download
    download_file(ZENODO_ZIP_URL, zip_path)

    # 2. Unzip
    if not os.path.exists(extract_dir):
        print("Extracting (this takes a while)...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
    print(f"Extracted to {extract_dir}")

    # 3. Find metadata csv
    meta_path = None
    for root, _, files in os.walk(extract_dir):
        for f in files:
            if f == "metadata_compiled.csv":
                meta_path = os.path.join(root, f)
                break
    if not meta_path:
        print("ERROR: metadata_compiled.csv not found. Check the extracted folder manually.")
        return
    audio_root = os.path.dirname(meta_path)
    print(f"Metadata: {meta_path}")

    df = pd.read_csv(meta_path)
    print(f"Total rows in metadata: {len(df)}")

    # ---- QUALITY FILTER 1: must actually contain a cough ----
    # cough_detected is a 0-1 probability from COUGHVID's own cough classifier.
    # >=0.8 is the threshold the COUGHVID paper itself uses for a reliable cough.
    if "cough_detected" in df.columns:
        before = len(df)
        df = df[df["cough_detected"] >= 0.8].copy()
        print(f"After cough_detected>=0.8 filter: {len(df)} (removed {before-len(df)} non-cough/noise)")

    # ---- LABEL CHOICE: prefer real user status over machine-guessed SSL ----
    status_col = "status" if "status" in df.columns else "status_SSL"
    df = df[df[status_col].isin(["COVID-19", "healthy"])].copy()

    # ---- QUALITY FILTER 2: take the CLEANEST recordings first (highest SNR) ----
    if "SNR" in df.columns:
        df = df.sort_values("SNR", ascending=False)
        print("Sorted by SNR — taking highest-quality recordings first")

    print(f"Usable labeled cough rows after all filters: {len(df)}")

    os.makedirs(os.path.join(RAW_DIR, "covid"), exist_ok=True)
    os.makedirs(os.path.join(RAW_DIR, "healthy"), exist_ok=True)

    counts = {"COVID-19": 0, "healthy": 0}
    label_folder = {"COVID-19": "covid", "healthy": "healthy"}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        label = row[status_col]
        if counts[label] >= args.max_per_class:
            if all(counts[k] >= args.max_per_class for k in counts):
                break
            continue

        uuid = row["uuid"]
        # find the audio file (webm or ogg)
        src = None
        for ext in (".webm", ".ogg"):
            cand = os.path.join(audio_root, uuid + ext)
            if os.path.exists(cand):
                src = cand
                break
        if not src:
            continue

        dst = os.path.join(RAW_DIR, label_folder[label], f"coughvid_{uuid}.wav")
        if convert_and_copy(src, dst):
            counts[label] += 1

    print(f"\nDone. Added:")
    print(f"  covid:   +{counts['COVID-19']}")
    print(f"  healthy: +{counts['healthy']}")
    print(f"\nNow re-run the pipeline:")
    print(f"  python build_dataset.py")
    print(f"  python ml_baseline.py")
    print(f"  python build_augmented_dataset.py")
    print(f"  python dl_model_augmented.py")
    print(f"  python cross_validate.py")


if __name__ == "__main__":
    main()