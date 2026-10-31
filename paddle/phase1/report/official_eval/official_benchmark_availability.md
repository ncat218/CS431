# PP-OCRv5_server_det — số công bố và khả năng reproduce

## Số công bố
- Docs Text Detection: Hmean **83,8** trên "general evaluation set" — không mô tả số ảnh/nguồn, **không công khai**.
  (https://www.paddleocr.ai/main/en/version3.x/module_usage/text_detection.html)
- Docs PP-OCRv5: bảng 12 nhóm trên "internal complex evaluation sets", trung bình **0,827**, General Scenario 0,797 — **không công khai**.
  (https://www.paddleocr.ai/latest/en/version3.x/algorithm/PP-OCRv5/PP-OCRv5.html)
- Tech report PaddleOCR 3.0 (arXiv 2507.05595) và PP-OCRv5 (arXiv 2603.24373): **không có bảng detection nào**.
- Bộ 2.677 ảnh trên trang docs là bộ **đo tốc độ**, không gắn với Hmean 83,8 (handoff ghi nhầm).

## Trạng thái
- Exact reproduction 83,8: **NOT VERIFIED** — không có tập dữ liệu gốc.
- Đã làm giống hệt bản chính thức: code đánh giá `tools/eval.py` (tag v3.3.3, commit c0522b89e221), config
  `configs/det/PP-OCRv5/PP-OCRv5_server_det.yml` không sửa, weights `PP-OCRv5_server_det_pretrained.pdparams` (sha256 2802f7d4748ea592…),
  nhãn `test_icdar2015_label.txt` chính thức, thước đo `DetMetric`.
- Kết quả trên ICDAR2015 test (tập Eval mặc định trong config chính thức):
  **P 58.52 · R 47.28 · Hmean 52.30**.

## Đã kiểm tra (ngày 2026-10-31)
- [x] Trang docs Text Detection — không có link tải tập đánh giá.
- [x] Trang docs PP-OCRv5 — chỉ ghi "internal", không có link.
- [x] arXiv 2507.05595 — không có số detection.
- [x] arXiv 2603.24373 — không có số detection.
- [ ] GitHub issues/discussions PaddleOCR (điền nếu đã hỏi)
