# Paddle — Phase 1 → review gate → Phase 2

Detector: **PP-OCRv5_server_det**, Kaggle T4 ×2 hoặc Colab A100. Không recognition.
Config tại `configs/paddle_v5.json`; dependencies GPU tại `requirements.txt` ở đây.
Exact reproduction con số official Paddle 83,8%: **NOT VERIFIED**.

Repo dùng checkpoint official để kiểm chứng trên **ICDAR2015 test** bằng evaluator
upstream đã pin trong `vendor/`. Đây là external benchmark validation, không phải
reproduction benchmark riêng Paddle.

**Kết quả Phase 1 chính thức:** ICDAR2015 test, P 58,52 / R 47,28 / **Hmean 52,30**.
Chạy bằng `tools/eval.py` của PaddleOCR v3.3.3, đúng config, weights và nhãn chính thức.
Xem [phase1/report/REVIEW.md](phase1/report/REVIEW.md).

## Notebooks

Phase 1 chính thức, chạy độc lập trên Kaggle (Python 3.13, không clone CS431):

- `paddle/phase1/notebooks/official_eval.ipynb`: kết quả chính thức 52,30.
- `paddle/phase1/notebooks/module_run.ipynb`: chạy qua `paddleocr.TextDetection`
  (Hmean 49,15). Predictions của nó là đầu vào cho notebook diag.
- `paddle/phase1/notebooks/diag_line_vs_word.ipynb`: phân tích box dòng so với GT theo từ.

Pipeline repo, dùng cho gate Phase 2:

- `paddle/phase1/notebooks/repo_pipeline_phase1.ipynb`: `run.py paddle phase1` + review.
- `paddle/phase1/notebooks/pipeline_phase1_phase2.ipynb`: toàn bộ workflow, dừng tại
  review cho đến khi người thực hiện điền đánh giá; sau đó mới có Phase 2.

Hai notebook pipeline do CS431 tổ chức, không phải notebook nguyên bản của tác giả.
Kaggle bật Internet + T4 ×2, Colab dùng một GPU nếu được cấp A100.

**Từ 01/11/2026 pipeline repo dùng đúng detector của Phase 1 chính thức** (PaddleOCR v3.3.3,
config/weights `.pdparams` chính thức, paddle 3.3.1, Python 3.13; `cs431/ppocr_detector.py`
chạy y như `tools/eval.py`). Gate Phase 1 của repo vì vậy phải ra lại ≈ Hmean 52,30.
Bản cũ (`paddleocr.TextDetection` 3.0.3, resize 960, Hmean 49,15) không còn dùng.

Phase 2 trên Kaggle (khuyến nghị):

- `paddle/phase2/notebooks/paddle_phase2_kaggle.ipynb`: venv riêng từ Python 3.13 của Kaggle →
  `paddle phase1` (gate, so với 52,30) → người thực hiện xem overlays và tự điền `paddle review` →
  `paddle phase2` → `compare` → zip để review.

```bash
python paddle/tools/setup_paddle.py --venv /kaggle/working/cs431-paddle
/kaggle/working/cs431-paddle/bin/python paddle/tools/prepare_paddle_model.py
python run.py paddle status
```

Setup tạo môi trường riêng; model cache mặc định ở `paddle/models/`. `--gpus 0,1`
cho hai T4, `--gpus 0` cho một A100. Không cài dependencies Paddle vào env Monkey.

## Phase 1

Kết quả chính thức đã có, xem phần đầu. Phần dưới đây là pipeline repo để mở gate Phase 2.

Attach đủ 500 test images và localization/transcription GT ICDAR2015, ghi nguồn thật.
Chạy `python run.py paddle phase1 --images ... --gt ... --dataset-source ...` bằng
Python env Paddle. Code chặn missing/extra/duplicate IDs, ghi hashes và raw counts.
Outputs mặc định vào `paddle/phase1/report/`: metrics, per-image results, predictions,
logs, provenance, pip freeze, 20 overlays và `REVIEW_ME.md`.

Sau khi kiểm tra, người thực hiện dùng `python run.py paddle review --reviewer ...
--notes ...`. Không có auto-PASS theo 83,8% vì đó là benchmark khác. Clone mới chưa
có local receipt luôn bị chặn Phase 2. Report hiện có hoặc report Monkey không mở gate.

## Phase 2

### Colab A100 (một GPU)

Mở [notebook Colab A100](https://colab.research.google.com/github/ncat218/CS431/blob/paddle-phase2/paddle/phase2/notebooks/paddle_phase2_colab_a100.ipynb).
Chọn A100 và High-RAM nếu có, chạy từng cell từ trên xuống. Notebook tải ba dataset
Kaggle; nếu cần xác thực, thêm Colab secret `KAGGLE_API_TOKEN` và bật notebook access.
Cell 1b tải TextVQA clean bằng từng file của Version 1 (đúng 9.898 ảnh trong
manifest), tránh ZIP Kaggle đang `awaiting compression`. Kiểm SHA256 từng ảnh;
chạy lại giữ ảnh đã tải đúng hash. ICDAR2015 và degraded vẫn tải bằng gói thông thường.
Mount Drive khi được yêu cầu. Sau Phase 1, xem metrics/overlays rồi tự nhập tên và
nhận định ở bước 4; tiếp tục bước 5–9. ZIP cuối nằm trong
`MyDrive/CS431_Paddle_Phase2/<run>/CS431_A100_COMPLETE.zip`.

Notebook pin runtime `f586e14ab8d63e5f63a12550ac447da4f04a79b2`, bản GPU-only
có sửa pip trước CPU-retry. Giữ nguyên detector Triệu, FP32/batch 1/736-min;
không fallback CPU (đường CPU trước đó đã chạm giới hạn RAM 30 GiB).
Probe ảnh TextVQA từng OOM trước Phase 1; probe thành công không bảo đảm toàn bộ run.
Clone `/content/CS431_A100`, outputs `outputs/paddle_colab_a100_v1`, GPU `0`.
Không trộn prediction T4 hoặc sửa receipt. Chưa xác nhận hoàn tất trên A100 thực tế.

Drive lưu snapshot INCOMPLETE mỗi 15 phút và khi cell Phase 2 dừng; bản này giữ
bằng chứng, không phải gói tự phục hồi runtime. Cùng runtime/GPU/clone thì chạy lại
bước 7 để resume. Nếu Colab xóa runtime, clone mới phải chạy/review lại Phase 1.
Snapshot có thể chứa dòng JSONL cuối chưa hoàn chỉnh. Bước 9 chỉ ghi COMPLETE sau
khi đủ 19.796 prediction và metrics đã sinh; kiểm SHA256 bản sao Drive.

Đọc notebook shared, protocol/audit trong `shared/` và `docs/PHASE2_SHARED_MANIFEST.md`
ở root trước khi thực hiện. Dùng đúng bytes manifest 9.898 cặp và chung evaluator.
Chỉ đổi local dataset roots. Không join/sample riêng hoặc tạo lại degradation.

```bash
python run.py paddle phase2 --locations /actual/dataset_locations.json --gpus 0,1
python run.py compare --paddle-metrics paddle/phase2/report/metrics/metrics.json
```

Dùng Python env Paddle cho các lệnh chạy model. Phase 2 output mặc định ở
`paddle/phase2/report/`; bảng chung tại `comparison/phase2.md`. Gate ràng buộc
clone, artifacts Phase 1, model/config/code/package identity. Thay layout/code từ
bản repo cũ làm identity khác; không sửa receipt để vượt gate, chạy/review lại
Phase 1 tương ứng nếu đã có receipt local trước lần cập nhật này.

Report folders là nơi lưu kết quả khi chạy, không tự upload vào GitHub. Khi nhóm
cần công bố báo cáo đã review, chọn summary/provenance nhỏ phù hợp để commit;
giữ raw predictions, weights và dữ liệu gốc ngoài git. Chi tiết stack/evaluator:
`../docs/PADDLE.md`.
