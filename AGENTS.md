# CS431 — quy tắc làm việc

Yêu cầu trực tiếp mới nhất của người dùng có ưu tiên cao hơn tài liệu lịch sử.
Đọc README.md và docs/WORKFLOW.md trước khi thay đổi workflow.

## Workspace Monkey riêng

- Monkey làm việc trong `monkey/`, mở `Monkey.code-workspace` hoặc
  folder đó; đọc thêm AGENTS.md của workspace.
- Dùng launcher `monkey/run.py` để cache/output/env thuộc runtime
  Monkey riêng. Không đặt artifacts Monkey vào outputs/models/.local của Paddle.
- Shared manifest/protocol/evaluator và reports vẫn là tài nguyên của repo chung.
- Repo H200 cũ được giữ riêng, không tự thay đổi hoặc dùng lại state đang chạy.

## Paddle: gate theo từng clone

- Môi trường đích: Kaggle T4 ×2; thay thế: Colab A100, một GPU.
- Clone mới luôn bắt đầu Phase 1. Không đưa `.local/` hoặc receipt mở gate vào git.
- Chỉ mở Phase 2 sau khi chạy đủ ICDAR2015 test, kiểm tra kết quả/overlays,
  và ghi nhận đánh giá bằng `python run.py paddle review`.
- Cho phép official-checkpoint external validation trên ICDAR2015. Không gọi
  kết quả này là reproduction Hmean 83,8% trên bộ benchmark riêng Paddle.
- Không tạo receipt giả, không tự approve thay người thực hiện, không giảm
  số ảnh test, sửa gate hoặc mượn kết quả Monkey để mở gate Paddle.
- Thay checkpoint, config, inference code hoặc package versions làm mất gate.

## Bắt buộc trước mọi công việc Phase 2

Đọc cả ba nguồn này trước khi sửa/chạy code Phase 2:
1. `shared/Shared_Manifest_TextVQA_TextOCR_Public.ipynb`.
2. `shared/frozen/evaluation_protocol.json` và `shared/frozen/data_audit.md`.
3. `docs/PHASE2_SHARED_MANIFEST.md`.

- Dùng đúng bytes manifest reference đã freeze và đúng evaluator/protocol hash
  trong `configs/shared_lock.json`; không tự join/sample lại cho mỗi model.
- Notebook shared là nguồn code audit/tạo manifest. Phase 2 thông thường chỉ
  giải nén manifest đã freeze, không rebuild và không tạo lại degradation.
- Có 9.898 source/cặp hợp lệ từ bộ MAIN 10K; không bù ảnh để đủ 10K.
- Không thay đổi ignored policy, ROI, homography hoặc score threshold theo test.
- Một evaluator dùng chung; không CER/WER, recognition, restoration trong Phase 1–2.
- Không trộn full-master Phase 2 với test split Phase 3 đã fine-tune.
- File handoff trong docs/reference có trạng thái cũ. Monkey đã có Phase 1–3;
  xem monkey/phase1/report, monkey/phase2/report, monkey/phase3/report.
  Repo H200 riêng vẫn là nơi vận hành HPC.

## Kiểm tra và công bố

Chạy `python -m unittest discover -s tests -v` và kiểm tra shared hashes.
Khi sửa Monkey Phase 3, chạy thêm bộ tests trong monkey/phase3.
Không báo đã chạy GPU nếu chỉ kiểm thử CPU. Không commit ảnh dataset, raw full
TextOCR JSON, weights, credentials hoặc prediction files lớn. Manifest nén
đã freeze là annotation subset chung có chủ đích, không chứa pixels/transcripts.
