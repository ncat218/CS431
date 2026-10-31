# Monkey trong CS431

Phase 1: notebook ICDAR2015, đã bỏ execution outputs; kết quả/review lưu trong
`reports/monkey/phase1`. Phase 2: notebook baseline và runner lịch sử; bản shared
manifest/evaluator dùng chung hiện nằm tại `shared/` ở repo root.

Không chạy lại notebook Phase 2 cũ rồi tự coi manifest mới là cùng bản. Mốc để
so Paddle là reference frozen 9.898 cặp trong CS431, cùng ba hash đã khóa.
Notebook cũ có cells tự chứa code để bảo tồn thí nghiệm, không phải nguồn mới
để sửa shared protocol. Quy tắc repo root có ưu tiên hơn hướng dẫn lịch sử.

Phase 3: snapshot tracked files của repo H200 tại commit
`0afec7c1d63f8cfc8577fab6a5c79a0d4ee88048`. Không chứa nested `.git` hay caches.
Chạy/setup HPC tiếp tục tại [repo H200](https://github.com/ncat218/MonKeyOCR-).
CS431 giữ toàn bộ code và reports của phương pháp; không thay repo vận hành.

Kết quả job 7957 được lưu cùng review. Chỉ có báo cáo tổng hợp trong gói Phase 3
được cung cấp, chưa có raw prediction/training artifacts để audit lại độc lập.
Full-master Phase 2 và Phase 3 test split là hai phạm vi đánh giá khác nhau.
