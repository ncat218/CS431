# Repository validation — 31/10/2026

Local CPU verification used Windows, Python 3.11.9, NumPy 2.3.5, Pillow 11.3.0,
Shapely 2.2.0 and MMEngine 0.10.7 with its lightweight Config dependencies.

- 20 CS431 tests passed: local review gate, changed artifacts/identity, another clone,
  full ICDAR test requirements, parser/evaluator, two-worker orchestration with an
  explicit fake worker, interrupted JSONL handling, shared hashes, paired source IDs,
  notebooks and incompatible comparison rejection.
- The end-to-end Phase 1 controller test uses 500 tiny synthetic fixture images and
  an explicitly mocked detector. It checks real parsing/evaluation/rendering/evidence
  generation/review mechanics. It is not an ICDAR reproduction or Paddle GPU result.
- 49 existing Monkey snapshot CPU tests passed, including actual MMEngine Config
  serialization, geometry, source-safe splits, resume and the fake-GPU whole-suite test.
- Frozen manifest decompresses to SHA256
  `94e3a2ce989ce6a298d7ea30e7ce70f22ac676092b29df808bc3d33b426bf242`.
  All 9.898 sources/variants and paired valid annotation ID sets were checked.
- A clean checkout from Git's staged files preserved the locked byte hashes and
  unpacked the same manifest; fresh-checkout Paddle status and Phase 2 both blocked.
- Notebook code cells compile and contain no saved execution outputs. Git whitespace
  checks pass; local environments, receipts, dataset pixels and weights are excluded.

The official Paddle inference graph/config were fetched from an immutable official
HF revision. The weights SHA256 was obtained from the revision's HF LFS metadata;
the large weight binary was not independently downloaded during preparation.

**Paddle GPU setup/inference and real ICDAR2015 evaluation have not been executed
in this local session.** Kaggle T4 ×2 or Colab A100 execution is still required before
the operator can accept Phase 1 and open the local Phase 2 gate. No Paddle result or
review receipt is published as already completed. GitHub Actions provides additional
Linux CPU checks when the pushed workflow is enabled.

## Separate model folders and Monkey workspace

After adopting the group's proposed root `monkey/`, `paddle/`, `comparison/`
layout, 26 repo CPU tests and all 49 Monkey snapshot tests passed. Added checks
cover relocated model tools/config/report paths, the root Monkey dispatcher,
the editor workspace, runtime isolation, a real Phase 3 plan without creating
runtime/submitting jobs, and rejection of Paddle predictions by Monkey Phase 2.
Shared locked hashes still match. No GPU workload was launched for this reorganization.
