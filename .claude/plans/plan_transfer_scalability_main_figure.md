# Plan: 15t zero-shot transfer main figure
Date: 2026-10-27
Status: done

## What
- Add `scripts/plot_transfer_scalability.py`.
- Read the existing `output/9_4/scalability/summary_ontime.csv`, retain only
  `learned` and `flat`, and write an explicit transfer-data CSV beside the figure.
- Render a caption-free PNG showing arrival on-time rate for the fixed 15t
  hierarchical and flat policies evaluated without target-scale fine-tuning at
  10/15/20/25/30 trains.

## Why
The existing summary is a zero-shot 15t-to-scale evaluation, but its filename
and the broader scalability directory do not make the transfer protocol obvious.
A dedicated data extract and plot provide a traceable main figure without
misrepresenting the separate 25t from-scratch convergence trace as transfer.

## Alignment check
- [x] No hardcoded IDs: OK; only scale values and CSV method labels are used.
- [x] Observation 74-dim: N/A; no model or observation change.
- [x] Action mask enforces bounds: N/A.
- [x] Local critic 444-dim: N/A.
- [x] Fallback re-verifies bounds: N/A.
- [x] Config-only expansion: N/A; analysis-only artifact.

## Pass criteria
- A reproducible CSV preserves the five scales, two methods, on-time values,
  source-policy metadata, and evaluation protocol metadata.
- The PNG has no caption embedded in the image and clearly distinguishes
  hierarchical MAPPO from flat MAPPO.
- Re-running the script recreates both artifacts from the source summary CSV.
