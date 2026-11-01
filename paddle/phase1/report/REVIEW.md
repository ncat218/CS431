# Phase 1 — PP-OCRv5_server_det (PaddleOCR)

Ngày chạy: 31/10/2026, Kaggle Tesla T4. Notebook: [official_eval.ipynb](../notebooks/official_eval.ipynb).
Bằng chứng: [official_eval/](official_eval/).

Quyết định 31/10/2026: lấy kết quả của notebook này làm kết quả Phase 1 chính thức của Paddle.

## Kết quả

| | Precision (%) | Recall (%) | Hmean (%) | Tập đánh giá |
|---|---:|---:|---:|---|
| Công bố (docs Text Detection) | — | — | 83,8 | "general evaluation set", không mô tả, không công khai |
| **Lần chạy này** | **58,52** | **47,28** | **52,30** | ICDAR2015 test, 500 ảnh, 2.077 box care |

Exact reproduction 83,8: **NOT VERIFIED**. Tập đánh giá cho ra 83,8 là bộ nội bộ của Paddle,
không công khai. Chênh lệch −31,5 điểm phần trăm là do khác tập dữ liệu, không phải sai số
tái hiện. Đây là kiểm chứng checkpoint chính thức trên benchmark công khai (external validation).

Các số công bố khác, đều trên bộ nội bộ không công khai:

- Docs PP-OCRv5: trung bình 12 nhóm 82,7, nhóm General Scenario 79,7.
- Paper PP-OCRv6 (arXiv 2606.13108), bảng 4: PP-OCRv5 server det trung bình 81,6 trên 16 nhóm.

Hai tech report PaddleOCR 3.0 (arXiv 2507.05595) và PP-OCRv5 (arXiv 2603.24373) không có số
detection. Chi tiết: [official_benchmark_availability.md](official_eval/official_benchmark_availability.md).

## Cách chạy

Giữ nguyên quy trình đánh giá chính thức của PaddleOCR, không sửa code, config hay ngưỡng:

- Script: `tools/eval.py` của repo PaddleOCR, tag `v3.3.3`, commit
  `c0522b89e2218741f174ec8b45588dec93fa66b3`, clone trực tiếp, không sửa file.
- Config: `configs/det/PP-OCRv5/PP-OCRv5_server_det.yml`, không sửa. Tập Eval mặc định của
  config này là ICDAR2015 test; `DetResizeForTest` mặc định (cạnh ngắn 736); `DetMetric`.
  Chỉ override đường dẫn checkpoint, data_dir và label file.
- Weights: `PP-OCRv5_server_det_pretrained.pdparams` chính thức, SHA256
  `2802f7d4748ea592819ae4550c195c5bdb43755dfdb5ebd25e01bb4d885aebc9`.
- Nhãn: `test_icdar2015_label.txt` chính thức của PaddleOCR, SHA256
  `07a7dafd27fe0b4e10ba37568ccaa524b716efca3b123ebc4f7cf87e418aeb96`.
- Ảnh: ICDAR2015 test `ch4_test_images` từ Kaggle dataset `bestofbests9/icdar2015`.
- Môi trường: Python 3.13, `paddlepaddle-gpu 3.3.1` (CUDA 12.6), Tesla T4. Thời gian chấm 130 s.

## Bằng chứng đã kiểm tra

- Log nạp weights có dòng `resume from`, **0** cảnh báo thiếu key và **0** cảnh báo lệch shape.
  Notebook dừng nếu có cảnh báo, vì `load_model` của PaddleOCR chỉ cảnh báo chứ không báo lỗi.
- Nhãn chính thức khớp GT gốc RRC: 500/500 ảnh cùng ID, 5.230 box (3.153 `###`, 2.077 care),
  **0** ảnh lệch ([label_audit.json](official_eval/label_audit.json)).
- Repo PaddleOCR sạch, đúng commit của tag ([git_commit.txt](official_eval/git_commit.txt)).
- P/R/Hmean đọc trực tiếp từ log `tools/eval.py` ([official_eval.log](official_eval/official_eval.log)).
- Hash của mọi file bằng chứng: [artifacts_sha256.txt](official_eval/artifacts_sha256.txt).

## Phân tích phụ

Chạy qua module `paddleocr.TextDetection` ([module_run.ipynb](../notebooks/module_run.ipynb),
resize cạnh dài 960, inference model) cho P 58,33 / R 42,47 / Hmean 49,15. Thấp hơn 3,15 điểm
so với cách chạy chính thức ở trên, chủ yếu do khác cách resize.

Notebook [diag_line_vs_word.ipynb](../notebooks/diag_line_vs_word.ipynb) phân tích predictions
của lần chạy module này ([diag_report.md](diag_line_vs_word/diag_report.md)):

- 63,3% FN là từ nằm trong một box dòng; 45,9% FP là box gộp từ hai từ GT trở lên.
- Bỏ sót thật chỉ 15,5% từ care, chủ yếu chữ dọc hoặc xoay.
- Word Coverage Recall 84,55%; 91,01% box nằm trên chữ.

PP-OCRv5 trả box theo dòng, còn GT ICDAR2015 gán theo từng từ. Đây là nguyên nhân chính khiến
Hmean theo giao thức IoU-0,5 thấp, không phải do model bỏ sót chữ.

## Giới hạn

- Không phải reproduction 83,8 và không phải submission chính thức lên RRC.
- Ảnh overlay không được commit vì chứa pixel dataset.
- Stack này khác pin trong [paddle_v5.json](../../configs/paddle_v5.json): paddle 3.3.1 / repo
  PaddleOCR v3.3.3 / resize 736 theo cạnh ngắn, so với pin 3.0.0 / paddleocr 3.0.3 / resize 960
  theo cạnh dài. Lý do: Kaggle hiện chạy Python 3.13, và paddle 3.0.0 cùng numpy 1.26.4 không
  có wheel cho 3.13.
- Kết quả này **không mở gate Phase 2** của repo. Code gate không thay đổi, nên `run.py paddle
  phase2` vẫn cần `run.py paddle phase1` cùng `run.py paddle review` chạy trong cùng clone.
  Cách xử lý gate và stack pin cho Phase 2 chưa được chốt.
- *Cập nhật 01/11/2026 (Trí, Phase 2):* đã chốt dùng đúng stack của lần chạy này cho repo. `paddle_v5.json`
  giờ pin PaddleOCR v3.3.3 / config / `.pdparams` / paddle 3.3.1 như trên; `cs431/ppocr_detector.py` chạy y như
  `tools/eval.py`. Gate `run.py paddle phase1` vì vậy phải ra lại ≈ 52,30 trước khi sang Phase 2.

## Đánh giá của người thực hiện

- Người thực hiện: <tên>
- Ngày: 31/10/2026
- Nhận xét:
  - Đã chạy đúng quy trình đánh giá chính thức của PaddleOCR (`tools/eval.py`, tag v3.3.3,
    config không sửa, weights và nhãn ICDAR2015 chính thức). Kết quả P 58,52 / R 47,28 /
    Hmean 52,30 trên 500 ảnh ICDAR2015 test.
  - Các kiểm tra toàn vẹn đều đạt: nạp weights 0 cảnh báo thiếu key/lệch shape, nhãn chính
    thức khớp GT gốc 500/500 ảnh, repo PaddleOCR sạch đúng commit.
  - Không tái hiện được con số 83,8 vì tập đánh giá gốc là bộ nội bộ, không công khai;
    ghi NOT VERIFIED. Chênh lệch là do khác tập dữ liệu, không phải lỗi chạy.
  - Hmean thấp chủ yếu do PP-OCRv5 trả box theo dòng trong khi GT ICDAR2015 theo từ
    (63% FN nằm trong box dòng; Word Coverage Recall 84,55%), không phải do bỏ sót chữ.
  - Chấp nhận kết quả này làm kết quả Phase 1 của Paddle (external validation trên benchmark
    công khai). Gate Phase 2 của repo và vấn đề Python 3.13 chưa xử lý.
- Nhóm xác nhận:
