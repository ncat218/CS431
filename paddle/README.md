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
Chúng gọi detector-only API và evaluator Paddle đã pin; không tạo metric sẵn.
Kaggle bật Internet + T4 ×2, Colab dùng một GPU nếu được cấp A100. Python 3.11/3.12.
Kaggle hiện chạy Python 3.13 nên `setup_paddle.py` sẽ dừng; chưa xử lý.

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
