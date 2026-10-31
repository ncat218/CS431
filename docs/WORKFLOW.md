# Workflow hiện tại — quyết định 31/10/2026

Người dùng đã chốt: đưa toàn bộ phạm vi Monkey vào CS431, giữ repo Monkey cũ để
setup/chạy H200; Paddle chạy Kaggle T4 ×2 hoặc Colab A100; cho phép kiểm chứng
checkpoint Paddle trên ICDAR2015 trước Phase 2.

Cập nhật theo feedback: Monkey có workspace độc lập tại `monkey/` và
file `Monkey.code-workspace`. Launcher tại đó dùng `.runtime/` hoặc datastore
`_workspace_cs431_monkey`; Phase 3 nằm dưới `<runtime>/phase3`. Tất cả môi trường,
model cache, logs và outputs mới của Monkey thuộc runtime riêng này. Shared frozen
manifest/evaluator vẫn đọc từ repo root, bản giải nén local giữ đúng bytes.
Không dùng output/state Paddle hoặc workspace `_workspace_monkeyocr_text_detection_phase3`
của repo H200 cũ. Việc chuyển folder không thay code/tham số thí nghiệm của snapshot.

Layout được chốt theo cây thư mục nhóm cung cấp: `monkey/` và `paddle/` ở root;
report đặt trong từng phase; tools/config/vendor Paddle ở `paddle/`; `cs431/`
vẫn ở root để giữ import. `run.py` dispatch monkey/paddle/shared/compare. Bảng
so sánh chung sinh tại `comparison/phase2.md`. Shared frozen artifacts giữ nguyên.

## Phase 1

Monkey: lưu bằng chứng/review đã có; không reset hoặc tự chạy lại GPU. Báo cáo gốc
vẫn giữ trạng thái group-review tại thời điểm tạo, không sửa lịch sử thành PASS.

Paddle: mọi clone mới phải chạy đủ 500 ảnh ICDAR2015 test. Code kiểm tra tập ID,
đọc GT quadrilateral/transcript ignore, ghi hashes ảnh/GT và model/runtime/code,
chạy inference giữ postprocessing đã pin, chấm bằng evaluator PaddleOCR 3.0.3.
Kết quả, 20 overlays, logs và counts được lưu trước bước review.
Paddle reports mới mặc định vào `paddle/phase1/report` và `paddle/phase2/report`.
Config/code identity đổi do chuyển đường dẫn; không sửa receipt local cũ để bỏ
qua kiểm tra. Nếu đã có run trước khi đổi layout, chạy/review lại Phase 1 tương ứng.

`paddle review` chỉ thực hiện khi người chạy đã xem kết quả và quyết định chấp nhận
external validation. Nhập tên và nhận định thực tế. Không có threshold so với 83,8%
vì đây là benchmark khác; không lấy kết quả Monkey làm target cho Paddle. Gate chặn
metric không hữu hạn/degenerate, thiếu ảnh, artifact bị sửa hoặc receipt của clone khác.

Receipt ở `.local/`, gắn với clone ID, result hash và identity model/config/code/packages.
Fresh clone không được kế thừa gate từ git. Nếu mất môi trường/output Phase 1, cần
khôi phục đúng clone và artifacts, hoặc chạy/review lại trong clone mới. Đây là quy tắc
workflow chống nhầm lẫn, không phải hệ thống bảo mật chống người cố ý sửa mã nguồn.

## Phase 2

Đọc notebook shared và protocol trước, giải nén đúng bản reference đã review, chỉ đổi
local dataset roots. Gate được kiểm tra trước khi load model/dữ liệu. Không có cờ
`--skip-phase1`, `--force` hoặc `--accept-any-manifest`.

Kaggle chia task xen kẽ cho hai process, mỗi process thấy đúng một GPU. Colab dùng
một process GPU 0. Cùng checkpoint, config, fixed postprocessing, cùng raw prediction
schema cho evaluator. Worker chỉ có paths/IDs/image hashes; không nhận GT.

Resume giữ từng JSONL đã ghi, ràng buộc task/model/manifest/sharding/GPU. Chỉ sửa
fragment cuối bị ngắt và lưu lại fragment; lỗi giữa file dừng. Không trộn output giữa
hai lượt. Khi thay GPU topology, chọn thư mục `--out` mới. Không chạy đồng thời hai
controller vào cùng output directory.

Sau inference đủ IDs mới chấm shared evaluator, tạo paired overlays và bootstrap
1.000 lần theo source. `run.py compare` từ chối metrics khác manifest/protocol/evaluator
hoặc khác số source. Chỉ dùng Monkey full-master baseline trong bảng Phase 2 này.

## Phase 3 Monkey

Snapshot commit `0afec7c1d63f8cfc8577fab6a5c79a0d4ee88048` từ repo H200 nằm trong
`monkey/phase3`. README trong đó mô tả runner HPC nguyên bản; chạy từ thư
mục đó nếu cần kiểm thử code snapshot. Các thay đổi vận hành HPC tiếp tục ở repo cũ.
Chạy suite từ workspace CS431 bằng `python run.py phase3` tại `monkey`;
launcher truyền workspace riêng vào runner gốc. `--plan-only` không submit hoặc
bootstrap. Repo H200 cũ tiếp tục có runner riêng cho các lượt vận hành ở đó.
CS431 không tự đồng bộ/push thay đổi sang repo H200. Khi cập nhật snapshot sau này,
ghi commit mới và cập nhật provenance rõ ràng.

Giữ Phase 3 E0/tiling/E1/E2 cùng split và ngân sách đã ghi. Không gộp fine-tuned
Monkey với checkpoint Paddle nguyên bản để gọi là cùng baseline. Không dùng full
master đã phân tích như một test hoàn toàn chưa từng được xem.

Handoff trong `docs/reference` là tài liệu lịch sử, gồm trạng thái và script minh họa
cũ. Notebook shared + frozen artifacts + quyết định người dùng trong tài liệu này
là nguồn hiện hành khi có khác biệt.
