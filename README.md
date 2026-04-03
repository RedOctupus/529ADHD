[README.md](https://github.com/user-attachments/files/26453694/README.md)
# 529 code package

This folder is the narrow WiDS package you asked for. It contains only the code paths needed to recover:

- ADHD preprocessing + BrainNetCNN baseline training
- Sex preprocessing + manual-tuned BrainNetCNN training
- ROI permutation method
- ROI-permuted training with the same manual-tuned Sex architecture

## Folder map

### `wids_brainnetcnn_baseline`
- `run_wids_brainnetcnn_baseline.py`
  - Rebuilt from the original tutorial notebook.
  - Reproduces the original WiDS BrainNetCNN baseline line for both `ADHD` and `Sex`.
  - Includes end-to-end preprocessing:
    - FC CSV loading
    - label / metadata merge
    - Fisher z transform
    - edge-vector to `ROI x ROI` matrix reconstruction
    - original BrainNetCNN architecture
    - 5-fold CV training and output export
- `baseline_config.json`
  - Exact baseline hyperparameters from the original notebook line.
- `brainnetcnn_summary_table.csv`
  - Existing baseline summary table.

### `sex_manual_tuned`
- `run_sex_tuning_compare.py`
  - Manual-tuned Sex training script that produced the report result.
- `sex_cfg.json`
  - Saved manual-tuned configuration.
- `sex_tuning_comparison.csv`
  - Existing comparison summary.

### `roi_permutation_manual`
- `run_roi_structure_sanity.py`
  - ROI permutation sanity-check script.
  - Uses the same manual-tuned Sex BrainNetCNN architecture as the report.
  - Contains both:
    - `global_fixed_permutation`
    - `subjectwise_random_permutation`
- `permutation_meta.json`
  - Saved permutation metadata.
- `roi_structure_comparison.csv`
  - Existing summary for the three ROI-layout settings.

## Quick index

- ADHD baseline:
  - `wids_brainnetcnn_baseline/run_wids_brainnetcnn_baseline.py`
- Sex manual-tuned:
  - `sex_manual_tuned/run_sex_tuning_compare.py`
  - use `config_name = manual_tuned`
- ROI permutation proof:
  - `roi_permutation_manual/run_roi_structure_sanity.py`

## Notes

- All scripts resolve paths from the project root and are runnable from this copied location.
- This folder is intentionally narrow. It is the quickest path back to the WiDS code line.
