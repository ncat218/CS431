# Monkey workspace

Đọc thêm AGENTS.md, README.md và docs/WORKFLOW.md tại repo root (`..`).

- Đây là workspace riêng cho DBNet + MonkeyOCRv2-AS: notebooks Phase 1–2,
  runner Phase 3, môi trường/cache/output Monkey. Không sửa workflow Paddle khi
  tác vụ chỉ yêu cầu Monkey.
- Mở `../Monkey.code-workspace`, hoặc mở riêng folder này trong editor.
  Thư mục shared và reports vẫn thuộc cùng repo CS431.
- Dùng `python run.py status`, `python run.py phase3 --plan-only` để kiểm tra
  setup mà không submit job hoặc chạy GPU.
- Mọi lệnh chạy mới dùng launcher `run.py` tại workspace này để tách runtime.
  Mặc định runtime là `.runtime/` tại đây, hoặc
  `/datastore/<user>/_workspace_cs431_monkey` trên server có datastore cá nhân.
  Không dùng cache/output Paddle hoặc mặc định workspace của repo H200 cũ.
- Repo MonKeyOCR- riêng tiếp tục phục vụ vận hành H200 cũ. Không chỉnh repo đó
  hoặc trộn state cũ vào workspace CS431 mới khi chưa audit identity.
- Trước Phase 2, đọc notebook shared, protocol, audit và
  `../docs/PHASE2_SHARED_MANIFEST.md`. Manifest và evaluator phải đúng hash.
  Không tạo bản manifest riêng theo model, không sửa snapshot evaluator.
- Phase 1–3 đã có kết quả tại `phase1/report`, `phase2/report` và `phase3/report`; không reset lịch sử hoặc
  tự chạy lại inference/training khi chỉ tổ chức workspace.
- Không cài Paddle vào môi trường Monkey. `.runtime/`, `.local/`, `.venv*/`,
  `workspace.local.json`, checkpoints và dataset pixels không được commit.

Kiểm tra: tests ở repo root và `python -m unittest discover -s tests -v`
trong `phase3/`. Chỉ báo CPU/GPU đã kiểm chứng đúng với việc thực tế đã chạy.
