# So sánh Phase 2 — Paddle và Monkey

Chạy từ root sau khi có Paddle metrics thực tế:

```bash
python run.py compare --paddle-metrics paddle/phase2/report/metrics/metrics.json
```

Mặc định đọc baseline Monkey tại `monkey/phase2/report/metrics.json`, kiểm cùng
manifest/protocol/evaluator và source counts, rồi xuất `comparison/phase2.md`.
Bảng gồm clean/degraded P/R/Hmean/coverage/latency và paired gaps/bootstrap CI.
Chỉ so full master 9.898 cặp; Phase 3 test split không thuộc bảng này.

Chưa có Paddle metrics nên chưa có bảng so sánh hoàn chỉnh. Output tự sinh được
gitignore; nhóm có thể chọn commit bảng đã review khi có kết quả cả hai model.
