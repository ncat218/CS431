# CS431 — Robust Scene Text Detection

Repo chung cho **PP-OCRv5_server_det (PaddleOCR)** và **DBNet + MonkeyOCRv2-AS**.
Phase 1 kiểm chứng checkpoint; Phase 2 so sánh TextVQA clean/degraded với TextOCR
polygon GT; Phase 3 lưu code và kết quả cải tiến Monkey. Chỉ phát hiện vùng chữ.

| Phương pháp | Phase 1 | Phase 2 | Phase 3 | Hướng dẫn |
|---|---|---|---|---|
| Monkey | Có kết quả ICDAR2015 | Có baseline 9.898 cặp | Có code và report job 7957 | [monkey/README.md](monkey/README.md) |
| Paddle | Chờ chạy và review theo clone | Chờ gate Phase 1 | Chưa triển khai | [paddle/README.md](paddle/README.md) |

## Cấu trúc

```text
CS431/
  README.md, AGENTS.md, requirements.txt
  run.py                         Dispatcher: monkey / paddle / shared / compare
  shared/                        Notebook + frozen manifest/protocol/evaluator
  comparison/                    Bảng so sánh Phase 2 do run.py compare sinh
  monkey/
    README.md, AGENTS.md, run.py   Workspace và runtime Monkey riêng
    phase1/
      MonkeyOCRv2_Phase1_ICDAR2015.ipynb
      report/                    REVIEW + metrics + hashes + source lock
    phase2/
      MonkeyOCRv2_Phase2_TextVQA.ipynb, run_monkey.py
      report/                    REVIEW + metrics + clean_vs_degraded
    phase3/                      Snapshot code H200, configs, src, tests
      report/                    job7957 + review
  paddle/
    README.md, requirements.txt
    configs/paddle_v5.json
    vendor/                      eval_det_iou.py + LICENSE
    tools/                       setup_paddle.py, prepare_paddle_model.py
    phase1/
      notebooks/official_eval.ipynb
      notebooks/pipeline_phase1_phase2.ipynb
      report/                    Chờ kết quả chạy và đánh giá Phase 1
    phase2/report/               Chờ gate và kết quả Phase 2
  cs431/                         Package Python dùng chung, giữ ở root
  configs/                       Shared lock, source provenance, local-root example
  docs/                          WORKFLOW, shared protocol, reference handoff
  tests/                         Kiểm thử CPU workflow/shared/dispatcher
  Monkey.code-workspace           Mở riêng Monkey + shared trong editor
```

## Monkey: workspace riêng

Mở [Monkey.code-workspace](Monkey.code-workspace) trong VS Code hoặc mở folder
[monkey](monkey). Reports nằm trong từng phase. Repo vận hành H200 cũ vẫn ở
[ncat218/MonKeyOCR-](https://github.com/ncat218/MonKeyOCR-).

```bash
git clone https://github.com/ncat218/CS431.git
cd CS431
python run.py monkey status
python run.py monkey phase3 --plan-only
```

Có thể `cd monkey` rồi dùng `python run.py status` / `python run.py phase3 --plan-only`.
Runtime mặc định là `monkey/.runtime/`, hoặc `/datastore/<user>/_workspace_cs431_monkey`
trên H200 có datastore cá nhân. Môi trường, cache và outputs Monkey thuộc runtime đó.
Lệnh plan không submit job/chạy GPU. Chi tiết tại [README Monkey](monkey/README.md).

## Paddle: Kaggle T4 ×2 hoặc Colab A100

Bắt đầu bằng [official_eval.ipynb](paddle/phase1/notebooks/official_eval.ipynb) cho
Phase 1, hoặc [pipeline_phase1_phase2.ipynb](paddle/phase1/notebooks/pipeline_phase1_phase2.ipynb)
cho luồng cả hai phase. `official_eval.ipynb` là notebook do CS431 tổ chức, dùng
upstream evaluator đã pin; không phải notebook do tác giả paper cung cấp.

```bash
python run.py paddle status
python paddle/tools/setup_paddle.py --venv /kaggle/working/cs431-paddle
/kaggle/working/cs431-paddle/bin/python paddle/tools/prepare_paddle_model.py
/kaggle/working/cs431-paddle/bin/python run.py paddle phase1 \
  --images /actual/icdar2015/ch4_test_images \
  --gt /actual/icdar2015/ch4_test_localization_transcription_gt \
  --dataset-source 'Nguồn URL/phiên bản ICDAR2015 test thực tế' --gpus 0,1
```

Clone mới báo `BLOCKED` là đúng. Cần đủ 500 ảnh ICDAR2015 test, xem metrics/logs/
20 overlays tại `paddle/phase1/report`, rồi ghi đánh giá chấp nhận bằng `paddle review`:

```bash
/kaggle/working/cs431-paddle/bin/python run.py paddle review \
  --reviewer 'Tên người đánh giá' \
  --notes 'Nhận định thực tế về P/R/Hmean, overlays, nguồn dữ liệu và giới hạn.'
```

Đây là **official-checkpoint external validation trên ICDAR2015**. Exact reproduction
Paddle Hmean 83,8% trên benchmark riêng: **NOT VERIFIED**. Không dùng 83,8% hoặc score
Monkey làm target ICDAR2015 cho Paddle. Receipt ở `.local/`, không commit và không
kế thừa qua clone mới. Hướng dẫn đầy đủ: [paddle/README.md](paddle/README.md).

## Phase 2: cùng shared benchmark

Trước Phase 2, đọc [notebook shared](shared/Shared_Manifest_TextVQA_TextOCR_Public.ipynb),
[protocol](shared/frozen/evaluation_protocol.json), audit và [quy tắc shared](docs/PHASE2_SHARED_MANIFEST.md).
Repo giữ đúng manifest nén 9.898 cặp đã freeze. Không join lại hoặc tạo lại degradation.

```bash
python run.py shared unpack
/kaggle/working/cs431-paddle/bin/python run.py paddle phase2 \
  --locations /actual/dataset_locations.json --gpus 0,1
/kaggle/working/cs431-paddle/bin/python run.py compare \
  --paddle-metrics paddle/phase2/report/metrics/metrics.json
```

Điền local roots theo [configs/dataset_locations.example.json](configs/dataset_locations.example.json).
Nguồn ảnh: [TextVQA clean](https://www.kaggle.com/datasets/anhthu128/text-vqa-train),
[degraded MAIN 10K](https://www.kaggle.com/datasets/anhthu128/textvqa-suy-thoi-10k-nh),
[TextOCR train JSON](https://www.kaggle.com/datasets/anhthu128/textocr-0-1-train-json).
GT subset đã có trong manifest; giữ metadata/JSON nguồn để audit.

Paddle Phase 2 xuất `paddle/phase2/report/`; bảng chung xuất `comparison/phase2.md`.
Dispatcher kiểm shared hashes/counts trước so sánh. Monkey baseline đọc từ
`monkey/phase2/report/metrics.json`. Không ghép Phase 3 test split 1.486 source vào
bảng full-master 9.898 source. Latency ghi kèm phần cứng từng run.

## Kiểm thử

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py shared verify
```

requirements.txt tại root chỉ có phụ thuộc CPU; môi trường GPU cài riêng từng model.
CI chạy thêm tests snapshot Monkey ở `monkey/phase3`. Paddle GPU chưa được chạy trong
lúc tổ chức repo. Không commit pixels dataset, weights, credentials hoặc raw predictions
lớn; output mới tại Paddle reports/comparison được gitignore. Xem [notices](THIRD_PARTY_NOTICES.md).
