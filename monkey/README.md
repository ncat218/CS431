# Monkey — workspace riêng trong CS431

Workspace cho **DBNet + MonkeyOCRv2-AS**, gồm Phase 1 reproduction, Phase 2
baseline và Phase 3 E0/tiling/E1/E2. Mở `../Monkey.code-workspace` trong VS Code,
hoặc mở riêng folder `monkey` trong editor. Terminal làm việc tại đây.

```bash
git clone https://github.com/ncat218/CS431.git
cd CS431/monkey
python run.py status
python run.py phase3 --plan-only
```

Hai lệnh cuối chỉ xem cấu hình, không tải model, submit Slurm hoặc chạy GPU.
Không cần cài Paddle. Python launcher dùng thư viện chuẩn; model Phase 3 tự có
môi trường riêng khi chạy thật. Repo [MonKeyOCR- H200](https://github.com/ncat218/MonKeyOCR-)
vẫn tồn tại độc lập cho các lượt chạy HPC cũ.

## Thư mục và runtime

```text
monkey/
  AGENTS.md, README.md         Hướng dẫn và phạm vi Monkey
  run.py                      Entry point riêng
  workspace.example.json      Mẫu cấu hình local
  phase1/                     Notebook ICDAR2015
  phase2/                     Notebook baseline và adapter inference đã có
  phase3/                     Snapshot runner H200, configs, src, tests
  .runtime/                   Local runtime (gitignored)
  .local/                     Local paths/state (gitignored)
  .venv*/                     Môi trường local riêng (gitignored)
```

Trên H200, nếu có `/datastore/<user>`, runtime mặc định là
`/datastore/<user>/_workspace_cs431_monkey`; local là `.runtime/` tại workspace.
Bên trong runtime: `shared/` chứa bản giải nén đúng hash, `phase2/` chứa predictions
và metrics baseline, `phase3/` chứa môi trường/model cache/logs và các suite runs.
Không ghi vào cache/output Paddle hoặc workspace mặc định của repo H200 cũ.

Muốn đổi runtime/data/interpreter, copy `workspace.example.json` thành
`workspace.local.json`, điền absolute paths thật. File local không commit:

```json
{
  "runtime_root": "/datastore/YOUR_USER/_workspace_cs431_monkey",
  "python": "/absolute/path/to/verified-monkey-environment/bin/python",
  "dataset_parent": "/datastore/dungmt/textvqa/data/benchmark_v1/raw/"
}
```

Chỉ đặt `python` khi chạy lại Phase 2 bằng môi trường Monkey đã xác minh.
Phase 3 tự bootstrap Python/model stack của nó vào runtime, không dùng trường đó.
Có thể override runtime qua `CS431_MONKEY_WORKSPACE` hoặc `--runtime-root` trước
tên subcommand. Không chuyển state đang chạy giữa hai repo hoặc đổi path giữa run.

## Phase 1 và kết quả hiện có

`phase1/MonkeyOCRv2_Phase1_ICDAR2015.ipynb` là notebook reproduction đã bỏ outputs.
Kết quả/review Phase 1–3 ở `phase1/report`, `phase2/report` và `phase3/report`, xuất hiện riêng trong workspace
editor. Không reset hoặc tự chạy lại GPU khi chỉ mở workspace. Snapshot Phase 3
giữ nguyên code tại commit H200 `0afec7c1d63f8cfc8577fab6a5c79a0d4ee88048`.

## Phase 2 — cùng shared benchmark

Trước khi làm Phase 2 đọc:

- `../shared/Shared_Manifest_TextVQA_TextOCR_Public.ipynb`
- `../shared/frozen/evaluation_protocol.json` và `data_audit.md`
- `../docs/PHASE2_SHARED_MANIFEST.md`

```bash
python run.py shared verify
python run.py shared unpack
```

Giải nén vào runtime Monkey riêng, giữ đúng SHA256 reference 9.898 cặp để so
Paddle. Không build/join lại, không regenerate degradation. Tạo
`.local/dataset_locations.json` theo mẫu ở repo root với clean_root/degraded_root thật.
Trong GPU allocation/môi trường Monkey đã xác minh, có thể chạy baseline:

```bash
python run.py --python /absolute/monkey-env/bin/python phase2 infer \
  --locations .local/dataset_locations.json \
  --config /absolute/verified-phase1-config.py \
  --checkpoint /absolute/dbnet_mkv2vitae_icdar2015.pth --gpus 0
python run.py --python /absolute/monkey-env/bin/python phase2 evaluate
```

Adapter/evaluator giữ checkpoint/config Phase 1 và ba shared hashes. Tasks đi
GPU chỉ có paths/IDs/image hashes, không có GT. Output ở `<runtime>/phase2/`.
Notebook Phase 2 cũ được bảo tồn để đọc lại thí nghiệm; không dùng nó tạo một
manifest độc lập rồi tự coi là tương đương với Paddle.

## Phase 3 — H200, runtime riêng

```bash
python run.py phase3 --plan-only
python run.py phase3
```

Lệnh chạy thật chuyển vào runner gốc nhưng luôn truyền workspace
`<runtime>/phase3`, tách khỏi state của repo H200 cũ. Runner giữ preflight,
auto-submit một job, bootstrap, resume và toàn bộ E0/tiling/E1/E2 như snapshot.
Có thể thêm `--dataset-parent /actual/path` hoặc `--config /actual/config.json`.
Xem `phase3/README.md` cho chi tiết model/Slurm/training; các ví dụ workspace mặc
định trong đó thuộc runner nguyên bản, launcher mới ưu tiên runtime riêng ở trên.

Kết quả job 7957 chỉ có báo cáo tổng hợp trong gói được cung cấp, chưa có raw
prediction/training artifacts để audit độc lập đầy đủ. Phase 2 full-master 9.898
source và Phase 3 test 1.486 source là hai phạm vi khác nhau.

## Kiểm thử CPU

Tại `phase3/`, dùng môi trường CPU riêng để chạy bộ tests snapshot:

```bash
cd phase3
python -m pip install -r requirements-controller.txt -r requirements-config-tests.txt
python -m pip install --no-deps mmengine==0.10.7
python -m unittest discover -s tests -v
```

Các test launcher ở `../tests` tính từ workspace Monkey chạy cùng bộ tests repo
root. Kiểm thử CPU không xác nhận một lượt GPU mới đã được thực hiện.
