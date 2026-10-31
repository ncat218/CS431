# PP-OCRv5_server_det trên ICDAR2015 test — chẩn đoán lệch dòng/từ (DIAGNOSTIC)

Predictions: `paddle_icdar2015_test_predictions.jsonl` (sha256 `4baef5ddb9c09bcb…`). Chấm lại bằng đúng evaluator Phase 1:
**P 58.33 · R 42.47 · Hmean 49.15** (khớp `metrics.json`).

## Phân rã lỗi (chỉ tính box/từ "care")
| Loại | Số lượng | Tỷ lệ |
|---|---:|---:|
| FP tổng | 630 | 100% FP |
| FP_merged_line (box chứa ≥ 2 từ GT) | 289 | 45.87% |
| FP_localization (chứa 1 từ, IoU ≤ 0,5) | 109 | 17.30% |
| FP_no_care_word | 232 | 36.83% |
| FN tổng | 1195 | 100% FN |
| FN_inside_merged_line | 756 | 63.26% |
| FN_localization | 118 | 9.87% |
| FN_missed (không box nào phủ ≥ 50%) | 321 | 26.86% |

## Chỉ số coverage (diagnostic)
- Word Coverage Recall: **84.55%** (từ GT được hợp các box phủ ≥ 50%)
- Box nằm trên chữ: **91.01%** (≥ 50% diện tích box nằm trong polygon GT, kể cả ###)
- Coverage Hmean: 87.66%
- Cận trên recall nếu mọi box dòng được tách đúng thành từ: 78.86%

## Recall theo chiều cao chữ
| size_bin | n_words | iou_recall | coverage_recall | missed | inside_merged_line |
|---|---|---|---|---|---|
| 0-10px | 17.0 | 0.1176 | 0.4706 | 0.5294 | 0.2941 |
| 10-20px | 609.0 | 0.3005 | 0.8128 | 0.1872 | 0.4516 |
| 20-40px | 989.0 | 0.4328 | 0.8625 | 0.1375 | 0.364 |
| 40-infpx | 462.0 | 0.5823 | 0.8658 | 0.1342 | 0.2511 |

## Giới hạn
- Ngưỡng 0,5 của coverage là quy ước của notebook này, không phải giao thức benchmark nào.
- Box to có thể làm coverage recall tăng ảo; vì vậy luôn đọc kèm tỷ lệ "box nằm trên chữ".
- Không thay số chính thức: Phase 1 vẫn báo P/R/Hmean theo giao thức ICDAR2015.
