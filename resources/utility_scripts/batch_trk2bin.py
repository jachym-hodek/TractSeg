#!/usr/bin/env python3
import sys
import argparse
import subprocess
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(
        description="Batch-generate bundle masks from TRK for each subject ID."
    )
    p.add_argument("--root", default="/home/jachymhodek/Coding/TractSeg/example_training_data",
                   help="Root folder that contains HCP105_Zenodo_NewTrkFormat, bundle_masks, hcp_masks/masks")
    p.add_argument("--trk2bin", default="/home/jachymhodek/Coding/TractSeg/TractSeg/resources/utility_scripts/trk_2_binary.py",
                   help="Path to trk_2_binary.py")
    p.add_argument("--ids-file", help="Text file: one ID per line. If omitted, IDs are auto-detected from HCP dir.")
    p.add_argument("--pattern", default="*.trk", help="Glob for tract files (default: *.trk)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output NIfTIs")
    p.add_argument("--dry-run", action="store_true", help="Print actions without running")
    return p.parse_args()

def read_ids(hcp_dir: Path, ids_file: str | None):
    if ids_file:
        ids = [line.strip() for line in Path(ids_file).read_text().splitlines() if line.strip()]
    else:
        # autodetect: any subdir with a "tracts" folder
        ids = [p.name for p in sorted(hcp_dir.iterdir()) if (p / "tracts").is_dir()]
    return ids

def main():
    args = parse_args()
    root = Path(args.root)
    hcp_dir = root / "HCP105_Zenodo_NewTrkFormat"
    out_root = root / "bundle_masks"
    masks_dir = root / "hcp_masks" / "masks"
    trk2bin = Path(args.trk2bin)

    if not trk2bin.exists():
        print(f"ERROR: trk_2_binary.py not found at {trk2bin}", file=sys.stderr)
        sys.exit(1)

    ids = read_ids(hcp_dir, args.ids_file)
    if not ids:
        print("No IDs found. Check --root or provide --ids-file.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(ids)} ID(s).")

    total_jobs = 0
    done, skipped, failed = 0, 0, 0

    for sid in ids:
        trk_dir = hcp_dir / sid / "tracts"
        if not trk_dir.is_dir():
            print(f"[{sid}] WARNING: trk dir missing: {trk_dir}")
            continue

        mask_path = masks_dir / f"{sid}_nodif_brain_mask.nii.gz"
        if not mask_path.exists():
            print(f"[{sid}] WARNING: mask not found: {mask_path} -> skipping this ID")
            continue

        out_dir = out_root / sid
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

        trk_files = sorted(trk_dir.glob(args.pattern))
        if not trk_files:
            print(f"[{sid}] WARNING: no TRK files matching {args.pattern}")
            continue

        print(f"[{sid}] {len(trk_files)} tract(s)")

        for trk in trk_files:
            seg = trk.stem  # e.g., AF_left
            out_path = out_dir / f"{seg}.nii.gz"
            total_jobs += 1

            if out_path.exists() and not args.overwrite:
                print(f"  - {seg}: exists -> skip (use --overwrite to regenerate)")
                skipped += 1
                continue

            cmd = [sys.executable, str(trk2bin), str(trk), str(out_path), str(mask_path)]
            print(f"  - {seg}: {' '.join(cmd)}")
            if args.dry_run:
                continue

            try:
                subprocess.run(cmd, check=True)
                done += 1
            except subprocess.CalledProcessError as e:
                print(f"    ERROR: command failed with code {e.returncode}")
                failed += 1

    print("\nSummary:")
    print(f"  total planned: {total_jobs}")
    print(f"  done:          {done}")
    print(f"  skipped:       {skipped}")
    print(f"  failed:        {failed}")

if __name__ == "__main__":
    main()
