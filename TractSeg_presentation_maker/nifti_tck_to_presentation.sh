#!/bin/bash

# Convert NIfTI and TCK files to a presentation using Python scripts

TRACTSEG_DIR="${1:?Usage: $0 TRACTSEG_DIR BG_NIFTI}"
BG="${2:?Usage: $0 TRACTSEG_DIR BG_NIFTI}"

# Normalize to absolute paths
TRACTSEG_DIR="$(realpath "$TRACTSEG_DIR")"
BG="$(realpath "$BG")"

NIFTI_DIR="${TRACTSEG_DIR}/tractseg_output/bundle_segmentations"
TCK_DIR="${TRACTSEG_DIR}/tractseg_output/TOM_trackings"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/nifti_to_pngs.py" \
  "${NIFTI_DIR}" \
  --out-dir "${TRACTSEG_DIR}/tractseg_output/nifti_rendered" \
  --mode iso \
  --solid-color 255,60,60 \
  --surface-opacity 0.85 \
  --surface-smooth-iters 20 \
  --bg-nifti "${BG}" \
  --bg-mode volume \
  --bg-opacity 0.0075

python "${SCRIPT_DIR}/nifti_to_pngs.py" \
  "${NIFTI_DIR}" \
  --out-dir "${TRACTSEG_DIR}/tractseg_output/nifti_rendered_no_bg" \
  --mode iso \
  --solid-color 255,60,60 \
  --surface-opacity 0.85 \
  --surface-smooth-iters 20

python "${SCRIPT_DIR}/tck_to_pngs.py" \
  "${TCK_DIR}" \
  --out-dir "${TRACTSEG_DIR}/tractseg_output/tck_rendered" \
  --render vtk \
  --bg-nifti "${BG}" \
  --bg-mode volume \
  --bg-opacity 0.06 \
  --coloring axis \
  --tract-style unlit \
  --dec-gain 0.85 \
  --tube-radius 0 \
  --line-width 1.5

python "${SCRIPT_DIR}/tck_to_pngs.py" \
  "${TCK_DIR}" \
  --out-dir "${TRACTSEG_DIR}/tractseg_output/tck_rendered_no_bg" \
  --render vtk \
  --coloring axis \
  --tract-style unlit \
  --dec-gain 0.85 \
  --tube-radius 0 \
  --line-width 1.5

python "${SCRIPT_DIR}/png_to_presentation.py" \
  "${TRACTSEG_DIR}/tractseg_output/nifti_rendered" \
  "${TRACTSEG_DIR}/tractseg_output/nifti_rendered_no_bg" \
  "${TRACTSEG_DIR}/tractseg_output/tck_rendered" \
  "${TRACTSEG_DIR}/tractseg_output/tck_rendered_no_bg" \
  "${TRACTSEG_DIR}/tractseg_output/rendered_images.pptx"