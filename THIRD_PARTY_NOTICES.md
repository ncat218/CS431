# Third-party sources and research artifacts

Paddle ICDAR IoU evaluator is copied unchanged from PaddlePaddle/PaddleOCR v3.0.3,
commit `a00f633cef33a8192f853fdaf8decf24ce2240b0`, `ppocr/metrics/eval_det_iou.py`; the file is
byte-identical at v3.3.3 (commit `c0522b89e2218741f174ec8b45588dec93fa66b3`), which
`prepare_paddle_model.py` verifies. PaddleOCR is Apache-2.0; its license is included at
`paddle/vendor/LICENSE.PaddleOCR`. The evaluator retains its original
reference to MhLiao/DB. The PaddleOCR v3.3.3 source and the official training checkpoint
are fetched at run time (pinned commit/URL/SHA256 in `paddle/configs/paddle_v5.json`);
no weights are redistributed in this git repository.

Monkey Phase 3 is a snapshot of the user's ncat218/MonKeyOCR- repository at the
commit recorded in configs/source_provenance.json. Its existing third-party
notices are preserved in `monkey/phase3/THIRD_PARTY_NOTICES.md`.
External training frameworks, pretrained weights and full datasets are not vendored.

The shared manifest notebook, shared code, frozen manifest and reports were
provided/generated in the CS431 project. The manifest is a paired annotation
subset with geometry and source/image hashes. It contains neither image pixels
nor OCR transcripts; underlying TextVQA/TextOCR dataset terms continue to apply.
Original dataset links are in README.md. No claim is made to relicense the
underlying datasets, external source code or model weights.
