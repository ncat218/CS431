# Paddle Phase 2 report

Báo cáo kết quả để nhóm review: [REVIEW.md](REVIEW.md) — Kaggle T4 ×2, 02/11/2026, 19.796 lượt inference trên shared manifest 9.898 cặp.

| File | Nội dung |
|---|---|
| `REVIEW.md` | Kết quả, so sánh Monkey, phân tích lỗi, ngoại lệ 4 ảnh 4608 px, bằng chứng, giới hạn |
| `metrics.json`, `clean_vs_degraded.md` | Output evaluator đã freeze (nguyên bản từ lượt chạy) |
| `phase2_provenance.json`, `inference_timing.json`, `phase1_gate_metrics.json` | Provenance, review Phase 1, thiết bị từng ảnh |
| `precomputed_oom_predictions.json` | Prediction CPU local cho 4 ảnh vượt VRAM T4/RAM Kaggle |
| `recheck_verification.json` | Chấm lại độc lập từ raw predictions: khớp |

Raw predictions, per-image metrics và overlays không commit (kích thước/pixels); giữ ngoài git.
Notebook: `paddle/phase2/notebooks/` (Kaggle T4 ×2 batch dùng cho lượt này sinh từ cùng code, commit `a6d73b8`).
