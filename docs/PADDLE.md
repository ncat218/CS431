# PaddleOCR Phase 1 → Phase 2

Model: **PP-OCRv5_server_det**, detector only. **Cùng detector với Phase 1 chính thức**
(`paddle/phase1/report/REVIEW.md`, `tools/eval.py`, Hmean 52,30): repo PaddleOCR **v3.3.3**
(commit `c0522b89…`, không sửa), config `configs/det/PP-OCRv5/PP-OCRv5_server_det.yml`,
weights `PP-OCRv5_server_det_pretrained.pdparams`, `paddlepaddle-gpu 3.3.1` (cu126), Python 3.13 Kaggle.
Hashes/pins ở `paddle/configs/paddle_v5.json`; package versions trong `paddle/requirements.txt`
lấy đúng theo `official_eval/env_versions.txt`.

*Đổi ngày 01/11/2026:* trước đây repo pin `paddleocr.TextDetection` 3.0.3 + inference model,
resize cạnh dài 960 (Hmean ICDAR2015 49,15), khác cấu hình Phase 1 chính thức. Nhóm chốt dùng
cấu hình của Phase 1 chính thức cho cả gate Phase 1 lẫn Phase 2.

Inference (`cs431/ppocr_detector.py`) làm đúng như `tools/eval.py` / `tools/infer_det.py`:
`tools.program.load_config` → transforms Eval của config bỏ label ops (DecodeImage BGR,
`DetResizeForTest` mặc định = cạnh ngắn 736, NormalizeImage ImageNet, ToCHWImage) →
`build_model` + nạp `.pdparams` **strict** (thiếu key / lệch shape là lỗi) → `model.eval()`,
`no_grad`, batch 1, FP32 → `DBPostProcess` của config (0,3 / 0,6 / 1000 / 1,5). Score lấy từ cùng
`boxes_from_bitmap` mà DBPostProcess dùng, box phải trùng hệt output chính thức. Không tune trên test.

Nguồn chính thức đã kiểm tra:

- [PaddleOCR v3.3.3 tools/eval.py](https://github.com/PaddlePaddle/PaddleOCR/blob/c0522b89e2218741f174ec8b45588dec93fa66b3/tools/eval.py)
  và [tools/infer_det.py](https://github.com/PaddlePaddle/PaddleOCR/blob/c0522b89e2218741f174ec8b45588dec93fa66b3/tools/infer_det.py)
- [Config PP-OCRv5_server_det.yml](https://github.com/PaddlePaddle/PaddleOCR/blob/c0522b89e2218741f174ec8b45588dec93fa66b3/configs/det/PP-OCRv5/PP-OCRv5_server_det.yml)
- [Paddle evaluator vendored unchanged](https://github.com/PaddlePaddle/PaddleOCR/blob/c0522b89e2218741f174ec8b45588dec93fa66b3/ppocr/metrics/eval_det_iou.py) (giống từng byte bản a00f633 cũ)

## Môi trường Kaggle/Colab

`setup_paddle.py` tạo venv Linux Python 3.11–3.13 (Kaggle hiện 3.13), cài `paddlepaddle-gpu==3.3.1`
cu126 rồi `paddle/requirements.txt`; venv tách khỏi torch của Kaggle (Phase 1 từng gặp lỗi NCCL
`ncclCommShrink` khi cài vào môi trường gốc). Nếu driver chỉ hỗ trợ CUDA 11.8, dùng `--cuda cu118`
**trước** Phase 1. Không cài driver, không sudo. `nvidia-smi` và GPU matmul probe phải thành công.
`prepare_paddle_model.py` clone PaddleOCR v3.3.3 (kiểm commit, cây sạch, SHA config, evaluator
vendor = upstream) và tải weights chính thức (kiểm SHA256). Ghi pip freeze và package identity.

T4 ×2: `--gpus 0,1`, hai worker độc lập, không model parallel/DDP.
A100: `--gpus 0`, một worker. Không coi latency hai phần cứng khác nhau là so sánh model.
Batch 1, FP32, không TensorRT/HPI/AMP. Nếu OOM/lỗi schema/CUDA thì dừng và lưu logs;
không tự đổi resize/threshold rồi tiếp tục ghép predictions cũ.

## ICDAR2015 test

Người chạy cung cấp test images và localization/transcription GT đã giải nén từ
nguồn hợp lệ của ICDAR2015. Không tự tải một mirror không rõ nguồn. Khai báo nguồn thật
qua `--dataset-source`. Cần đúng `img_1` … `img_500` và `gt_img_1.txt` … `gt_img_500.txt`;
code phát hiện duplicate/missing/extra, parse transcript có dấu phẩy và BOM, kiểm
decode/EXIF và ghi SHA256. Tên/count phù hợp không tự chứng minh nguồn official;
reviewer phải xem provenance của dataset đã attach.

Evaluator là `DetectionIoUEvaluator` của PaddleOCR (= `DetMetric` trong `tools/eval.py`), IoU >0,5 và
ignore-overlap >0,5 theo code upstream, giữ nguyên matching của nó. Đây là pipeline đánh giá
chính thức trên public benchmark, không tuyên bố submission RRC hoặc tái hiện 83,8 (bộ nội bộ).
Không dùng shared TextOCR evaluator thay evaluator Phase 1. Gate yêu cầu đủ results và review.

Kiểm tra tái tạo: cùng code/config/weights/packages với Phase 1 chính thức nên gate repo phải ra
lại ≈ P 58,52 / R 47,28 / Hmean 52,30 (`REVIEW_ME.md` in chênh lệch). Lệch đáng kể nghĩa là adapter
không tái tạo pipeline chính thức → điều tra, không review. Đây không phải ngưỡng chất lượng.

Kiểm tra 20 overlay (green GT/gray ignored/red prediction), logs, counts/P/R/Hmean,
dataset provenance và config. Nếu lỗi nghiêm trọng, sửa và chạy lượt mới rồi review;
không ghi nhận chấp nhận chỉ để mở Phase 2.

## Gate/resume

`paddle review` ghi quyết định local, không publish hoặc sửa trạng thái Monkey.
Phase 2 kiểm tra toàn bộ artifact hashes, đủ 500 prediction/per-image IDs, counts,
clone ID và current model/code/environment. Thay packages/config/checkpoint/code
yêu cầu Phase 1/review tương ứng. Receipt không phải số liệu nghiên cứu và không
được commit. Fresh clone luôn cần Phase 1, kể cả người khác đã có kết quả trên GitHub.

Giữ output và clone nếu cần resume. Cùng lệnh/cùng topology giữ predictions từng
ảnh; fragment cuối bị ngắt được lưu rồi sửa, lỗi giữa file làm dừng. Không chạy hai
controller đồng thời vào cùng `--out`. Chọn `--out` mới khi thay cấu hình/GPU.

Sau Phase 2, xem `paddle/phase2/report/metrics/clean_vs_degraded.md`, JSON, paired
source CSV, overlays/gallery.html và provenance. `run.py compare` ghép với baseline
Monkey đã review, chỉ khi cả ba hash và số source/variant khớp.
