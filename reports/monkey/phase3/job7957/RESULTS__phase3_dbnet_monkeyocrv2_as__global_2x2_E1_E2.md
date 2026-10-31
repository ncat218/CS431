# Phase 3 Text Detection experiment comparison

All results use the same source-held-out test split and frozen Phase 2 polygon evaluator.
Checkpoint/merge selection used validation only. E1/E2 have equal optimizer-update and effective-batch budgets.

| Experiment | Clean P (%) | Clean R (%) | Clean Hmean (%) | Degraded P (%) | Degraded R (%) | Degraded Hmean (%) | Gap (pp) | Degraded small-text R (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0_original_checkpoint__global | 78.376 | 66.461 | 71.928 | 67.997 | 24.064 | 35.548 | -36.380 | 10.398 |
| E0_original_checkpoint__global_plus_2x2 | 59.852 | 72.813 | 65.699 | 38.749 | 25.630 | 30.853 | -34.847 | 11.950 |
| E1_clean_only__global | 69.080 | 72.928 | 70.952 | 54.657 | 29.987 | 38.727 | -32.225 | 14.741 |
| E2_clean_and_degraded_balanced__global | 64.090 | 72.018 | 67.823 | 51.336 | 35.151 | 41.729 | -26.094 | 19.312 |

## Scope and interpretation

Small-text means original GT bounding-box height <16 px. Improvement must increase degraded Hmean, not merely reduce the gap by lowering clean accuracy. Check precision, recall and both clean/degraded results.
Method difference CIs are paired source bootstrap in the JSON report; these are distinct from clean/degraded gap CIs.
No E3 consistency loss, restoration, recognition, Qwen or regeneration of degraded input images was run.

## Existing data and timing limitations

- Synthetic realistic_mix L2 with image_relative_fallback.
- Phase 2 master-set results were previously inspected; Phase 3 test is held out from optimization only.
- TextOCR train was part of Monkey pretraining sources; not an unseen-data claim.
- E1 and E2 independently initialize from the same official checkpoint.
- Whole suite uses one normal H200 MIG allocation; no measured T4/H200 speed ratio is assumed.
- Tiled latency may combine cached E0 global timing with newly measured tile timing; inspect inference summaries.
- Training targets approximate fragmented valid polygons by convex hull and mask outside the common ROI; the conversion audit records counts. Exact evaluation geometries remain unchanged.
