# Phase 2: shared full-master baseline

| Model | Condition | Sources | Variants | P (%) | R (%) | Hmean (%) | Coverage (%) | Mean latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| paddle | clean | 9898 | 0 | 60.466 | 23.432 | 33.775 | 72.074 | 247.02 |
| paddle | degraded | 9898 | 9898 | 57.564 | 9.646 | 16.524 | 35.081 | 256.12 |
| monkey | clean | 9898 | 0 | 77.408 | 65.148 | 70.751 | 75.255 | 786.12 |
| monkey | degraded | 9898 | 9898 | 67.535 | 22.343 | 33.578 | 25.669 | 797.27 |

Gap = degraded − clean, percentage points; paired source-weighted micro.

| Model | Metric | Gap (pp) | Paired bootstrap 95% CI (pp) |
|---|---|---:|---|
| paddle | precision | -2.902 | [-3.949, -1.925] |
| paddle | recall | -13.785 | [-14.443, -13.116] |
| paddle | hmean | -17.251 | [-18.003, -16.517] |
| paddle | word_coverage_recall | -36.993 | [-38.642, -35.325] |
| monkey | precision | -9.873 | [-11.178, -8.549] |
| monkey | recall | -42.805 | [-44.209, -41.362] |
| monkey | hmean | -37.173 | [-38.820, -35.522] |
| monkey | word_coverage_recall | -49.586 | [-51.361, -47.846] |

Clean is evaluated once per source. Coverage is diagnostic. Synthetic degradation; word-versus-line GT and prior pretraining exposure limit interpretation.

Latency is recorded per run/hardware; do not claim an A100/T4/H200 speed ranking from these runs.

Monkey baseline hardware: two Tesla T4 workers. Paddle hardware: see phase2_provenance.json and inference/gpu*/identity.json. Latency includes decode/preprocess/postprocess.

Paddle may retry GPU-OOM images on CPU without resizing. Check execution_device per prediction and cpu_fallback_images in inference/timing.json. Mixed latency includes CPU startup, queue and retries; do not interpret it as GPU-only speed. CPU/GPU numerical equivalence is not assumed.

Manifest SHA256: 94e3a2ce989ce6a298d7ea30e7ce70f22ac676092b29df808bc3d33b426bf242
