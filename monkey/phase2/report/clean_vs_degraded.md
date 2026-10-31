# Clean versus degraded — checkpoint baseline

External TextVQA images + TextOCR word GT; not paper-official Hmean. No fine-tuning.

| Metric | Clean paired (%) | Degraded paired (%) | Delta (pp) | 95% cluster-bootstrap CI (pp) |
|---|---:|---:|---:|---|
| precision | 77.408 | 67.535 | -9.873 | [-11.178372147215516, -8.549094694277056] |
| recall | 65.148 | 22.343 | -42.805 | [-44.20926310694678, -41.36246061504841] |
| hmean | 70.751 | 33.578 | -37.173 | [-38.819587770840215, -35.52232223182352] |
| word_coverage_recall | 75.255 | 25.669 | -49.586 | [-51.360712016579676, -47.84568986752603] |

Sources: 9898; degraded variants: 9898.
Manifest SHA256: `94e3a2ce989ce6a298d7ea30e7ce70f22ac676092b29df808bc3d33b426bf242`.

Delta = degraded - clean. CI excluding zero indicates a directional difference under this protocol;
effect size in percentage points determines practical magnitude. No preset conclusion of a large drop.
Review word-vs-line granularity, giant boxes, FP/FN and geometric masks alongside these scores.
