# CS431 — Robust Scene Text Detection

Repo chung của nhóm cho **PP-OCRv5_server_det (PaddleOCR)** và **DBNet + MonkeyOCRv2-AS**:
Phase 1 kiểm chứng checkpoint trên benchmark gốc/public; Phase 2 đánh giá clean/degraded
trên cùng TextVQA–TextOCR subset; Phase 3 lưu phương án cải tiến Monkey đã thực hiện.
Chỉ phát hiện vùng chữ; không chạy recognition hoặc VQA answers.

| Phương pháp | Phase 1 | Phase 2 | Phase 3 | Môi trường |
|---|---|---|---|---|
| Monkey | Có kết quả/review ICDAR2015 | Có baseline/review 9.898 cặp | Có code và báo cáo job 7957 | Repo H200 riêng |
| Paddle | Chưa chạy trong repo này; phải chạy và review theo clone | Bị chặn đến khi Phase 1 được đánh giá chấp nhận | Chưa triển khai | Kaggle T4 ×2 hoặc Colab A100 |

Repo vận hành Monkey trên H200 vẫn là [ncat218/MonKeyOCR-](https://github.com/ncat218/MonKeyOCR-).
CS431 lưu snapshot code đã pin và báo cáo; không thay remote hay sửa repo H200.

## Bắt đầu với Paddle

```bash
git clone https://github.com/ncat218/CS431.git
cd CS431
python run.py paddle status
```

Clone mới báo `BLOCKED` (exit code 2) là đúng. Git clone chỉ tải code, không tự
chạy inference. Gate được thực thi trong lệnh Phase 2; không có receipt đã duyệt
được commit sẵn. Không thể dùng kết quả Monkey để mở gate Paddle.

**Cách dùng trên Kaggle/Colab:** mở [notebooks/Paddle_Phase1_Phase2_Kaggle_Colab.ipynb](notebooks/Paddle_Phase1_Phase2_Kaggle_Colab.ipynb).
Notebook clone repo, tạo môi trường riêng, tải model đã pin, chạy Phase 1 và hiển thị
kết quả/overlays. Người chạy ghi nhận đánh giá rồi mới chạy các cell Phase 2.
Kaggle cần Internet và GPU T4 ×2; Colab cần runtime GPU, dùng `GPUS="0"` nếu có A100.
Python đích 3.11/3.12; notebook không đảm bảo tài khoản sẽ được cấp A100.

Luồng dòng lệnh trong Linux GPU:

```bash
python tools/setup_paddle.py --venv /kaggle/working/cs431-paddle
/kaggle/working/cs431-paddle/bin/python tools/prepare_paddle_model.py
/kaggle/working/cs431-paddle/bin/python run.py paddle phase1 \
  --images /actual/icdar2015/ch4_test_images \
  --gt /actual/icdar2015/ch4_test_localization_transcription_gt \
  --dataset-source 'Nguồn URL và phiên bản ICDAR2015 test thực tế đã attach' \
  --gpus 0,1
```

Xem `outputs/paddle/phase1/REVIEW_ME.md`, metrics, logs, per-image results và
20 overlays. Khi người thực hiện đã kiểm tra và chấp nhận kết quả:

```bash
/kaggle/working/cs431-paddle/bin/python run.py paddle review \
  --reviewer 'Tên người đánh giá' \
  --notes 'Ghi nhận định thực tế về P/R/Hmean, kiểm tra overlays và các giới hạn còn lại.'
```

Đây là **official-checkpoint external validation trên ICDAR2015**, không phải
reproduction con số Paddle 83,8% trên benchmark riêng. Chưa có mục tiêu ICDAR2015
đã xác minh cho checkpoint này để tự so ngưỡng PASS; review có giải thích là bắt buộc.
Chi tiết: [docs/PADDLE.md](docs/PADDLE.md).

## Phase 2 dùng đúng shared manifest

Trước khi làm Phase 2, đọc [notebook shared](shared/Shared_Manifest_TextVQA_TextOCR_Public.ipynb),
[protocol](shared/frozen/evaluation_protocol.json) và [quy tắc Phase 2](docs/PHASE2_SHARED_MANIFEST.md).
Manifest nén đã có trong repo, không cần bên Paddle build lại hoặc chép từ máy Monkey.
Lệnh `shared unpack` khôi phục đúng bytes bản gốc, kiểm SHA256 trước khi dùng.

```bash
python run.py shared unpack
```

Tạo `outputs/dataset_locations.json` theo [configs/dataset_locations.example.json](configs/dataset_locations.example.json)
với đường dẫn thật trong session. Ba dataset đầu vào:

- [TextVQA clean](https://www.kaggle.com/datasets/anhthu128/text-vqa-train)
- [TextVQA degraded MAIN 10K đã tạo](https://www.kaggle.com/datasets/anhthu128/textvqa-suy-thoi-10k-nh)
- [TextOCR train JSON](https://www.kaggle.com/datasets/anhthu128/textocr-0-1-train-json), phục vụ audit nguồn; GT subset đã có trong manifest.

```bash
/kaggle/working/cs431-paddle/bin/python run.py paddle phase2 \
  --locations outputs/dataset_locations.json --gpus 0,1
/kaggle/working/cs431-paddle/bin/python run.py compare \
  --paddle-metrics outputs/paddle/phase2/metrics/metrics.json
```

Gate kiểm tra receipt local, đủ bằng chứng Phase 1 và model/config/code/environment
khớp lượt đã review. Shared manifest/protocol/evaluator phải đúng hash. Mỗi worker
kiểm hash từng ảnh trước inference. Không tự bù ảnh, sample riêng hay tạo suy thoái.

## Cấu trúc

```text
AGENTS.md                    Luật bắt buộc cho trợ lý/người phát triển
run.py, cs431/               Paddle workflow, review gate, comparison
configs/                    Model/source pins, shared hashes, ví dụ local roots
notebooks/                  Paddle Kaggle/Colab, đi tuần tự Phase 1 → review → Phase 2
shared/                     Notebook manifest + code chung + frozen artifact
methods/monkey/phase1/       Notebook reproduction đã bỏ outputs
methods/monkey/phase2/       Notebook/code baseline lịch sử đã bỏ outputs
methods/monkey/phase3/       Snapshot code H200, gồm tests/configs/provenance
methods/paddle/              Dependencies và evaluator ICDAR upstream
reports/monkey/              Review + metrics có sẵn của Phase 1–3
docs/                       Workflow hiện tại và handoff lịch sử
outputs/, .local/, models/   Sinh local, không commit
```

Monkey Phase 2 gốc chấm full master 9.898 source; Phase 3 job 7957 chấm test split
1.486 source. Hai bảng có phạm vi khác nhau. Báo cáo Phase 3 hiện chỉ kiểm tính
nhất quán bảng tổng hợp, chưa tái chấm từ raw predictions; xem review kèm theo.

## Kiểm tra CPU

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py shared verify
```

CI chạy thêm bộ tests Monkey Phase 3. Kiểm thử CPU không xác nhận Paddle CUDA inference.
Paddle phải được chạy thực tế trên Kaggle/Colab rồi review; không có số liệu Paddle giả.
Không đưa dataset pixels, full raw TextOCR JSON, weights hoặc credentials lên GitHub.
Nguồn và licenses: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
