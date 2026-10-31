# PaddleOCR Phase 1 → Phase 2

Model: **PP-OCRv5_server_det**, detector only. PaddleOCR/PaddleX 3.0.3,
PaddlePaddle GPU 3.0.0. Config/HF revision/file hashes ở `configs/paddle_v5.json`.
Môi trường Paddle pin NumPy 1.26.4 để giữ tương thích stack CV cũ; môi trường CPU
kiểm thử/audit dùng NumPy 2.3.5. Không đổi package versions giữa hai phase Paddle.
Default official inference model: resize long side 960/max, DB thresh 0,3,
box threshold 0,6, unclip 1,5. Giữ giống nhau giữa hai phase, không tune trên test.

Nguồn chính thức đã kiểm tra:

- [PaddleOCR 3.0.3 TextDetection](https://github.com/PaddlePaddle/PaddleOCR/blob/a00f633cef33a8192f853fdaf8decf24ce2240b0/paddleocr/_models/text_detection.py)
- [Official Paddle model, pinned revision](https://huggingface.co/PaddlePaddle/PP-OCRv5_server_det/tree/ca867c897ecbca8873081573a802ad70d499cb94)
- [PaddleOCR 3.0.3 installation](https://www.paddleocr.ai/v3.0.3/en/version3.x/installation.html)
- [Paddle evaluator vendored unchanged](https://github.com/PaddlePaddle/PaddleOCR/blob/a00f633cef33a8192f853fdaf8decf24ce2240b0/ppocr/metrics/eval_det_iou.py)

## Môi trường Kaggle/Colab

Notebook tạo venv Linux Python 3.11/3.12, không sửa môi trường Monkey. Setup cài
wheel chính thức cu126 mặc định; nếu driver chỉ phù hợp CUDA 11.8, chủ động dùng
`tools/setup_paddle.py --cuda cu118` trước Phase 1. Không tự downgrade/đổi giữa phase.
Không cài driver, không cần sudo. `nvidia-smi` và Paddle GPU matmul probe phải thành công.
Nếu phiên bản Python runtime không phù hợp, chọn runtime Python phù hợp trước khi chạy.

Chỉ cài PaddleX OCR extras; package PaddleOCR cài `--no-deps` sau khi đã có các phụ
thuộc detector để tránh kéo thêm IE/multimodal không cần thiết. Ghi toàn bộ pip freeze
và environment package identity khi chạy. Đây là stack đã pin để kiểm chứng trên GPU,
chưa có real Paddle GPU execution trong lúc dựng repo.

T4 ×2: `--gpus 0,1`, hai worker độc lập, không model parallel/DDP.
A100: `--gpus 0`, một worker. Không coi latency hai phần cứng khác nhau là so sánh model.
Batch 1, FP32, không TensorRT/HPI. Nếu OOM/lỗi schema/CUDA thì dừng và lưu logs;
không tự đổi resize/threshold rồi tiếp tục ghép predictions cũ.

## ICDAR2015 test

Người chạy cung cấp test images và localization/transcription GT đã giải nén từ
nguồn hợp lệ của ICDAR2015. Không tự tải một mirror không rõ nguồn. Khai báo nguồn thật
qua `--dataset-source`. Cần đúng `img_1` … `img_500` và `gt_img_1.txt` … `gt_img_500.txt`;
code phát hiện duplicate/missing/extra, parse transcript có dấu phẩy và BOM, kiểm
decode/EXIF và ghi SHA256. Tên/count phù hợp không tự chứng minh nguồn official;
reviewer phải xem provenance của dataset đã attach.

Evaluator là `DetectionIoUEvaluator` từ PaddleOCR 3.0.3, mặc định IoU >0,5 và
ignore-overlap >0,5 theo code upstream, giữ nguyên matching của nó. Đây là external
validation với checkpoint official trên public benchmark, không tuyên bố submission
chính thức RRC hoặc tái hiện benchmark Paddle riêng. Không dùng shared TextOCR evaluator
thay evaluator Phase 1. Gate yêu cầu đủ results và review, không tự gán target ICDAR2015.

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

Sau Phase 2, xem `outputs/paddle/phase2/metrics/clean_vs_degraded.md`, JSON, paired
source CSV, overlays/gallery.html và provenance. `run.py compare` ghép với baseline
Monkey đã review, chỉ khi cả ba hash và số source/variant khớp.
