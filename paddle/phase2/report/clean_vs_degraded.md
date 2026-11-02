# Clean versus degraded — checkpoint baseline

External TextVQA images + TextOCR word GT; not paper-official Hmean. No fine-tuning.

| Metric | Clean paired (%) | Degraded paired (%) | Delta (pp) | 95% cluster-bootstrap CI (pp) |
|---|---:|---:|---:|---|
| precision | 60.466 | 57.564 | -2.902 | [-3.9487416367358716, -1.9253953287835672] |
| recall | 23.432 | 9.646 | -13.785 | [-14.442728262904112, -13.116250760820893] |
| hmean | 33.775 | 16.524 | -17.251 | [-18.002920619175892, -16.517407840598892] |
| word_coverage_recall | 72.074 | 35.081 | -36.993 | [-38.641777281962476, -35.32540372044525] |

Sources: 9898; degraded variants: 9898.
Manifest SHA256: `94e3a2ce989ce6a298d7ea30e7ce70f22ac676092b29df808bc3d33b426bf242`.

Delta = degraded - clean. CI excluding zero indicates a directional difference under this protocol;
effect size in percentage points determines practical magnitude. No preset conclusion of a large drop.
Review word-vs-line granularity, giant boxes, FP/FN and geometric masks alongside these scores.
