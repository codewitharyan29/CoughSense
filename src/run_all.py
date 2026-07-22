"""
ONE script to run the whole CoughSense pipeline end to end.

Runs every stage in the correct order so you don't have to type each command.

USAGE (from the src/ folder):

  # Normal run — uses whatever audio is already in data/raw/
  python run_all.py

  # Also download + add more data FIRST (COUGHVID), then run everything
  python run_all.py --download --max-per-class 300

  # Skip the slow CNN cross-validation (faster)
  python run_all.py --skip-cv

Stages, in order:
  1. (optional) download_more_data.py   — fetch + sort extra clips
  2. build_dataset.py                    — features for classical ML
  3. ml_baseline.py                      — train RF + XGBoost
  4. build_augmented_dataset.py          — augmented spectrograms
  5. dl_model_augmented.py               — train CNN
  6. cross_validate.py                   — 5-fold CV numbers
  7. explain.py                          — SHAP + ROC + confusion plots

Each stage is just the existing script — nothing is duplicated, this only
chains them so there's a single command to run.
"""

import argparse
import subprocess
import sys
import time


def run(cmd, label):
    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)
    t0 = time.time()
    result = subprocess.run([sys.executable] + cmd)
    if result.returncode != 0:
        print(f"\n!! Stage failed: {label} (exit {result.returncode})")
        print("   Fix the error above, then re-run. Earlier stages don't need redoing.")
        sys.exit(result.returncode)
    print(f"   done in {time.time()-t0:.0f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--download", action="store_true", help="download + add COUGHVID data first")
    p.add_argument("--max-per-class", type=int, default=300, help="clips per class when downloading")
    p.add_argument("--skip-cv", action="store_true", help="skip slow CNN cross-validation")
    args = p.parse_args()

    if args.download:
        run(["download_more_data.py", "--max-per-class", str(args.max_per_class)], "1/7  Download + sort extra data")

    run(["build_dataset.py"], "2/7  Extract features (classical ML)")
    run(["ml_baseline.py"], "3/7  Train Random Forest + XGBoost")
    run(["build_augmented_dataset.py"], "4/7  Build augmented spectrograms")
    run(["dl_model_augmented.py"], "5/7  Train CNN")
    if not args.skip_cv:
        run(["cross_validate.py"], "6/7  5-fold cross-validation")
    else:
        print("\n(skipping cross-validation)")
    run(["explain.py"], "7/7  SHAP + ROC + confusion plots")

    print("\n" + "=" * 60)
    print("  ALL DONE")
    print("=" * 60)
    print("  Models  -> ../models/")
    print("  Figures -> ../reports/figures/")
    print("  Next: start backend  ->  uvicorn serve:app --reload --port 8000")


if __name__ == "__main__":
    main()
