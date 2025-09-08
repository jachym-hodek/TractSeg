#!/usr/bin/env python3
"""
reorganize_tractseg_dataset.py

Reorganize your current folder structure into TractSeg's expected per-subject layout:

  OUT_ROOT/HCP/<SUBJECT_ID>/
    mrtrix_peaks.nii.gz           # OPTIONAL if provided via --peaks-root
    bundle_masks.nii.gz           # Built by stacking per-bundle masks
    tracts/ *.trk                 # OPTIONAL if provided via --trk-root

Assumptions based on the user's screenshots and notes:
- You currently have per-bundle masks under something like:
      BUNDLE_MASKS_ROOT/<SUBJECT_ID>/*.nii.gz    (one NIfTI per bundle)
- You have streamlines under something like:
      TRK_ROOT/<SUBJECT_ID>/tracts/*.trk         (optional; copied if present)
- You *may* have peaks each in:
      PEAKS_ROOT/<SUBJECT_ID>/mrtrix_peaks.nii.gz  (optional; copied if present)

You can adapt the roots with command-line arguments.
The script will:
- Discover subjects from the --bundle-masks-root (preferred), falling back to the union with other roots.
- For each subject, stack its per-bundle masks into a single 4D 'bundle_masks.nii.gz'.
- Copy/symlink peaks and .trk files if available.
- Skip work if outputs already exist with the expected shape (unless --overwrite).

Usage example:
  python reorganize_tractseg_dataset.py \
      --out-root /abs/path/out \
      --bundle-masks-root /abs/path/bundle_masks \
      --trk-root /abs/path/HCP105_something \
      --peaks-root /abs/path/peaks \
      --copy-mode symlink

If you want a fixed, explicit bundle channel order across subjects, pass a file with one bundle
basename per line (without directory). Otherwise the script uses alphabetical order per subject.
  --bundle-order-file /abs/path/bundle_order.txt

This will also write the resolved order to:
  OUT_ROOT/bundle_order_resolved.txt
so you can keep it consistent with TractSeg's dataset_specific_utils.get_bundle_names().
"""

import argparse
from pathlib import Path
import shutil
import sys
import re
import numpy as np
import nibabel as nib

def discover_subjects_from_root(root: Path) -> set:
    subs = set()
    if not root:
        return subs
    if not root.exists():
        return subs
    # Preferred: subfolders named as subjects
    for p in root.iterdir():
        if p.is_dir():
            subs.add(p.name)
    # Fallback: look for files like <subject>_something.nii.gz directly inside
    if not subs:
        for p in root.glob("*.nii*"):
            m = re.match(r"^([A-Za-z0-9_\-]+)[\._].*$", p.name)
            if m:
                subs.add(m.group(1))
    return subs

def load_bundle_order(bundle_dir: Path, bundle_order_file: Path | None):
    """
    Returns a tuple: (bundle_names, bundle_paths)
    bundle_names: list[str] channel order
    bundle_paths: list[Path] paths in the same order
    """
    if bundle_order_file:
        names = [ln.strip() for ln in bundle_order_file.read_text().splitlines() if ln.strip() and not ln.strip().startswith("#")]
        # Map names to paths inside bundle_dir
        paths = []
        missing = []
        for n in names:
            cand = list(bundle_dir.glob(n))
            if not cand:
                # try with .nii.gz forced
                cand = list(bundle_dir.glob(n + ".nii.gz"))
            if not cand:
                missing.append(n)
            else:
                # if multiple, take the first match deterministically (sorted)
                cand = sorted(cand)
                paths.append(cand[0])
        if missing:
            raise FileNotFoundError(f"The following bundles from --bundle-order-file were not found in {bundle_dir}:\n  " + "\n  ".join(missing))
        return names, paths
    else:
        # alphabetical order over nifti masks in the subject bundle dir
        files = sorted([p for p in bundle_dir.iterdir() if p.suffix in [".nii", ".gz"] or p.name.endswith(".nii.gz")])
        # Filter to just .nii or .nii.gz explicitly
        files = [p for p in files if p.name.endswith(".nii") or p.name.endswith(".nii.gz")]
        names = [p.name for p in files]
        return names, files

def stack_bundle_masks(mask_paths: list[Path], dtype=np.uint8):
    """Load 3D binary masks and stack along 4th dim → (X,Y,Z,N)."""
    if not mask_paths:
        raise ValueError("No per-bundle mask files provided")
    ref_img = nib.load(str(mask_paths[0]))
    ref_data = ref_img.get_fdata()
    vol_shape = ref_data.shape[:3]
    # sanity: ensure 3D masks
    if ref_data.ndim != 3:
        raise ValueError(f"Reference mask {mask_paths[0].name} is not 3D (shape {ref_data.shape})")
    stacked = []
    for p in mask_paths:
        img = nib.load(str(p))
        data = img.get_fdata()
        if data.ndim != 3:
            raise ValueError(f"Mask {p.name} is not 3D (shape {data.shape})")
        if data.shape != vol_shape:
            raise ValueError(f"Mask {p.name} shape {data.shape} != reference {vol_shape}")
        # binarize defensively
        data_bin = (data > 0.5).astype(dtype)
        stacked.append(data_bin[..., None])
    out = np.concatenate(stacked, axis=3).astype(dtype)
    # Use ref header/affine
    out_img = nib.Nifti1Image(out, affine=ref_img.affine, header=ref_img.header)
    out_img.header.set_data_dtype(dtype)
    out_img.update_header()
    return out_img

def copy_or_link(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        # Use relative symlink if possible
        try:
            rel = src.resolve()
            dst.symlink_to(rel)
        except OSError:
            shutil.copy2(src, dst)
    else:
        raise ValueError("copy-mode must be 'copy' or 'symlink'")

def main():
    ap = argparse.ArgumentParser(description="Reorganize dataset into TractSeg's per-subject layout and stack per-bundle masks.")
    ap.add_argument("--out-root", required=True, type=Path, help="Output working directory root (will create OUT_ROOT/HCP/<SUBJECT_ID>/...)")
    ap.add_argument("--bundle-masks-root", required=True, type=Path, help="Root containing per-subject per-bundle mask NIfTIs (SUBJ subfolders).")
    ap.add_argument("--trk-root", type=Path, default=None, help="Root containing per-subject tract .trk files, e.g., <root>/<SUBJ>/tracts/*.trk (optional).")
    ap.add_argument("--peaks-root", type=Path, default=None, help="Root containing per-subject mrtrix_peaks.nii.gz, e.g., <root>/<SUBJ>/mrtrix_peaks.nii.gz (optional).")
    ap.add_argument("--bundle-order-file", type=Path, default=None, help="Text file with one bundle filename per line to fix channel order (optional).")
    ap.add_argument("--copy-mode", choices=["copy", "symlink"], default="symlink", help="Whether to copy or symlink aux files into the output (default: symlink).")
    ap.add_argument("--overwrite", action="store_true", help="Rebuild bundle_masks.nii.gz even if it already exists.")
    args = ap.parse_args()

    out_root = args.out_root
    out_hcp = out_root / "HCP"
    out_hcp.mkdir(parents=True, exist_ok=True)

    # Discover subjects primarily from bundle masks root (required), but union with other roots if present
    subs = discover_subjects_from_root(args.bundle_masks_root)
    if args.trk_root:
        subs |= discover_subjects_from_root(args.trk_root)
    if args.peaks_root:
        subs |= discover_subjects_from_root(args.peaks_root)

    if not subs:
        print("No subjects discovered. Check your --bundle-masks-root / --trk-root / --peaks-root paths.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(subs)} subjects.")
    resolved_order_written = False
    order_outfile = out_root / "bundle_order_resolved.txt"

    for subj in sorted(subs):
        print(f"\n== Subject: {subj} ==")
        subj_out = out_hcp / subj
        subj_out.mkdir(parents=True, exist_ok=True)

        # 1) STACK BUNDLE MASKS
        subj_bundle_dir = args.bundle_masks_root / subj
        if subj_bundle_dir.exists():
            try:
                bundle_names, bundle_paths = load_bundle_order(subj_bundle_dir, args.bundle_order_file)
                expected_channels = len(bundle_paths)
                out_bundle = subj_out / "bundle_masks.nii.gz"
                if out_bundle.exists() and not args.overwrite:
                    try:
                        ex = nib.load(str(out_bundle))
                        if ex.shape[:3] == nib.load(str(bundle_paths[0])).shape[:3] and ex.shape[3] == expected_channels:
                            print(f"  [OK] bundle_masks.nii.gz exists with {expected_channels} channels → keeping")
                        else:
                            print(f"  [WARN] Existing bundle_masks.nii.gz has shape {ex.shape}, expected 4th dim={expected_channels}. Rebuilding...")
                            raise Exception("rebuild")
                    except Exception:
                        img = stack_bundle_masks(bundle_paths)
                        nib.save(img, str(out_bundle))
                        print(f"  [WRITE] bundle_masks.nii.gz with {expected_channels} channels ({out_bundle})")
                else:
                    img = stack_bundle_masks(bundle_paths)
                    nib.save(img, str(out_bundle))
                    print(f"  [WRITE] bundle_masks.nii.gz with {expected_channels} channels ({out_bundle})")

                # Write resolved order once (first subject) for reproducibility
                if not resolved_order_written:
                    order_outfile.write_text("\n".join(bundle_names) + "\n")
                    print(f"  [INFO] Wrote channel order to {order_outfile}")
                    resolved_order_written = True

            except Exception as e:
                print(f"  [SKIP] Could not stack bundle masks for {subj}: {e}")
        else:
            print(f"  [SKIP] No bundle mask dir for {subj}: {subj_bundle_dir}")

        # 2) COPY/SYMLINK PEAKS if present
        if args.peaks_root:
            peaks_candidate = args.peaks_root / subj / "mrtrix_peaks.nii.gz"
            if peaks_candidate.exists():
                dst = subj_out / "mrtrix_peaks.nii.gz"
                if dst.exists() and not args.overwrite:
                    print("  [OK] mrtrix_peaks.nii.gz already present → keeping")
                else:
                    copy_or_link(peaks_candidate, dst, args.copy_mode)
                    print(f"  [LINK] mrtrix_peaks.nii.gz → {dst.name}")
            else:
                # Try alternate naming (e.g., peaks.nii.gz)
                alt = list((args.peaks_root / subj).glob("*peaks*.nii*"))
                if alt:
                    dst = subj_out / "mrtrix_peaks.nii.gz"
                    if dst.exists() and not args.overwrite:
                        print("  [OK] mrtrix_peaks.nii.gz already present → keeping")
                    else:
                        copy_or_link(alt[0], dst, args.copy_mode)
                        print(f"  [LINK] {alt[0].name} → mrtrix_peaks.nii.gz")
                else:
                    print("  [INFO] No peaks found for this subject (skipping)")

        # 3) COPY/SYMLINK .TRK if present
        if args.trk_root:
            # common layout: <trk_root>/<subj>/tracts/*.trk
            trk_dir = args.trk_root / subj / "tracts"
            if not trk_dir.exists():
                # fallback: directly under <subj>
                trk_dir = args.trk_root / subj
            if trk_dir.exists():
                out_trk_dir = subj_out / "tracts"
                out_trk_dir.mkdir(exist_ok=True)
                trk_files = sorted(list(trk_dir.glob("*.trk")))
                if not trk_files:
                    print("  [INFO] No .trk files found for this subject")
                else:
                    copied = 0
                    for f in trk_files:
                        dst = out_trk_dir / f.name
                        if dst.exists() and not args.overwrite:
                            continue
                        copy_or_link(f, dst, args.copy_mode)
                        copied += 1
                    print(f"  [LINK] {copied} .trk files into {out_trk_dir}")
            else:
                print("  [INFO] No .trk dir for this subject (skipping)")

    print("\nDone. Verify one subject: OUT_ROOT/HCP/<SUBJECT_ID>/ contains 'bundle_masks.nii.gz' (and peaks/tracts if provided).")

if __name__ == "__main__":
    main()