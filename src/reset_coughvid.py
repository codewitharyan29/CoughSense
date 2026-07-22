"""
Remove previously-added COUGHVID clips from data/raw, keeping the original
clinical Virufy clips intact.

All COUGHVID files were saved with a 'coughvid_' prefix, so this is safe:
it only deletes those, never the Virufy clinical clips.

Run this BEFORE re-running the download with the new quality filters, so the
clean filtered clips don't mix with the old unfiltered ones.

  python reset_coughvid.py
"""

import os
import glob

RAW_DIR = "../data/raw"


def main():
    removed = 0
    for cls in ["covid", "healthy"]:
        pattern = os.path.join(RAW_DIR, cls, "coughvid_*.wav")
        files = glob.glob(pattern)
        for f in files:
            os.remove(f)
            removed += 1
        print(f"{cls}: removed {len(files)} COUGHVID clips")

    # Report what remains (should be the clinical Virufy clips)
    for cls in ["covid", "healthy"]:
        remaining = len(os.listdir(os.path.join(RAW_DIR, cls)))
        print(f"{cls}: {remaining} clinical clips remain")

    print(f"\nTotal COUGHVID clips removed: {removed}")
    print("data/raw is now back to clean clinical-only. Safe to re-download with filters.")


if __name__ == "__main__":
    main()
