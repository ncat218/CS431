# Paddle Phase 1 report

Kết quả Phase 1 chính thức của Paddle: [REVIEW.md](REVIEW.md).
Exact official 83,8% reproduction: NOT VERIFIED.

| Folder | Nội dung | Notebook sinh ra |
|---|---|---|
| `official_eval/` | Bằng chứng chạy `tools/eval.py` chính thức: metrics, log, label audit, hashes, môi trường | `../notebooks/official_eval.ipynb` |
| `diag_line_vs_word/` | Phân tích phụ: box dòng so với GT theo từ | `../notebooks/diag_line_vs_word.ipynb` |

Các file khác sinh tại đây khi chạy `run.py paddle phase1` (pipeline repo, dùng cho
gate Phase 2) vẫn được gitignore. Không có receipt mở gate được commit.
