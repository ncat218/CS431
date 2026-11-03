# Đánh giá Phase 2 — PP-OCRv5_server_det trên TextVQA clean/degraded

Người thực hiện: Trí. Chạy ngày 02/11/2026 trên Kaggle **Tesla T4 ×2** (notebook batch `ojanek/cs431-paddle-phase2-t4x2`,
lượt B), repo commit `a6d73b8` (nhánh `paddle-phase2`; sau khi lịch sử được viết lại, cây code giống hệt `ca9444f` trên `main`). Bản phân tích này do trợ lý soạn từ artifacts; người thực hiện đọc và xác nhận.

Detector **giống hệt Phase 1 chính thức** của Triệu: PaddleOCR v3.3.3 (commit `c0522b89…`, không sửa), config
`PP-OCRv5_server_det.yml`, checkpoint `.pdparams` SHA256 `2802f7d4…`, paddle-gpu 3.3.1, tiền/hậu xử lý y như `tools/eval.py`
(`DetResizeForTest` cạnh ngắn 736, DBPostProcess 0,3 / 0,6 / 1,5), FP32, batch 1. Không train, không đổi ngưỡng, không thêm score filter.
Cùng manifest/protocol/evaluator với Monkey Phase 2: **9.898 nguồn clean + 9.898 biến thể degraded = 19.796 lượt inference**.

## Gate Phase 1 trong cùng lượt

ICDAR2015 test 500 ảnh: **P 58,522 / R 47,280 / Hmean 52,304** (GT 2077, pred 1678, matched 982) — trùng từng chữ số với
`tools/eval.py` của Triệu (Δ = 0,000 pp) ở cả lượt A lẫn lượt B. 553/553 tham số, 0 thiếu, 0 lệch shape. Người thực hiện đã xem
20 overlays và ghi review (receipt `ACCEPTED_EXTERNAL_ICDAR2015_VALIDATION`, nhận xét trong `phase2_provenance.json`).
Khớp 52,30 chỉ chứng minh cùng pipeline; không có nghĩa model phát hiện tốt mọi trường hợp, và 83,8 vẫn **NOT VERIFIED**.

## Kết quả chính — polygon IoU ≥ 0,5, cùng protocol

| Metric | Clean (%) | Degraded (%) | Degraded − clean (điểm %) | 95% CI (cluster bootstrap 1.000, seed 42) |
|---|---:|---:|---:|---|
| Precision | 60,466 | 57,564 | −2,902 | [−3,949; −1,925] |
| Recall | 23,432 | 9,646 | −13,785 | [−14,443; −13,116] |
| Hmean | 33,775 | 16,524 | −17,251 | [−18,003; −16,517] |
| Word coverage recall (diagnostic) | 72,074 | 35,081 | −36,993 | [−38,642; −35,325] |

Mỗi nguồn có đúng một biến thể nên paired source-weighted micro trùng clean/degraded micro. Macro theo nguồn: Hmean 44,92 → 24,74.
Giảm tương đối: Hmean −51,1 %, Recall −58,8 %, Coverage −51,3 %, Precision −4,8 %.

## So với Monkey Phase 2 (cùng 9.898 cặp, cùng evaluator)

| Model | Hmean clean | Hmean degraded | Δ Hmean (pp) | Coverage clean | Coverage degraded | Δ Coverage (pp) |
|---|---:|---:|---:|---:|---:|---:|
| PP-OCRv5_server_det | 33,775 | 16,524 | −17,251 | 72,074 | 35,081 | −36,993 |
| DBNet + MonkeyOCRv2-AS | 70,751 | 33,578 | −37,173 | 75,255 | 25,669 | −49,586 |

Bảng đầy đủ (P/R, CI cả hai model): `comparison/phase2.md`.

Đọc kết quả cần tách hai thước đo:

- **Theo IoU-word (thước đo chính của protocol)**, Monkey cao hơn rõ ở cả clean lẫn degraded. Nhưng Paddle có Recall clean chỉ
  23,4 % trong khi Coverage clean 72,1 % (gần Monkey 75,3 %): phần lớn chữ đã nằm trong box Paddle nhưng box theo **dòng/cụm từ**,
  còn GT TextOCR theo **từ**, nên trượt IoU 0,5. Đây là cùng hiện tượng đã đo ở Phase 1 (`paddle/phase1/report/diag_line_vs_word/`).
  Số box được chấm của Paddle trên ảnh clean 123.422 so với 318.490 GT và 268.050 box của Monkey.
- **Mức suy giảm tương đối của Hmean gần như nhau** (Paddle −51,1 %, Monkey −52,5 %). Không so trực tiếp Δ tuyệt đối −17 với −37 pp
  vì hai model xuất phát từ mức clean khác nhau.
- **Theo Coverage (diagnostic)**, Paddle giữ được nhiều vùng chữ hơn khi suy thoái: 35,1 % so với 25,7 % (giảm tương đối −51 % so với
  −66 %). Coverage có thể tăng giả nếu box quá lớn. Đã đo tỉ lệ diện tích ROI bị box Paddle phủ: trung vị 4,1 % (clean) / 3,6 %
  (degraded), p95 31 % / 34 %; số ảnh có box phủ >50 % ROI là 116 / 189 trên 9.898. Không thấy box khổng lồ phổ biến, nhưng đuôi
  phân phối tăng nhẹ khi suy thoái; precision vẫn ~58 %. Coverage chỉ là tín hiệu phụ, không thay P/R/Hmean. (Chưa có số tương ứng
  của Monkey trong report đã commit để so cùng thước đo này.)

## Phân tích lỗi (từ artifacts)

- Recall theo nguồn: **6.970/9.898 giảm**, 638 tăng, 2.290 không đổi.
- Ảnh không có raw prediction nào: **171 (1,73 %) → 1.414 (14,29 %)**. Raw polygons 139.164 → 56.274; sau ROI/ignored policy còn
  123.422 → 53.372 box được chấm.
- Recall theo chiều cao bbox GT gốc: <16 px **11,92 → 1,84 %**; 16–32 px 24,78 → 8,80 %; ≥32 px 42,58 → 25,36 %. Chữ nhỏ gần như
  biến mất sau suy thoái (cùng xu hướng Monkey: 60,13 → 9,92 %). Đây là bin theo kích thước polygon trong ảnh gốc, không phải
  thí nghiệm kiểm soát.
- Theo perspective: 5.974 cặp không perspective Hmean 34,22 → 15,05 (−19,17 pp); 3.924 cặp có perspective 33,12 → 18,61 (−14,51 pp).
  Hai nhóm có thành phần phép suy thoái khác nhau; không kết luận perspective "bảo vệ" model.
- Đã xem trực tiếp cặp overlay `0002d070329eb0fc` (nhóm recall giảm mạnh nhất): ảnh clean 4/4 chữ "Heineken" được bao sát; ảnh degraded
  (chói sáng + perspective) GT vẫn bám đúng vị trí sau homography, nhưng Paddle gộp hai chữ và ngôi sao thành một box dài (trượt IoU),
  bỏ sót hai chữ ở xa và có một box sai trên kính xe. Đây là kiểm tra mẫu, không phải audit thị giác toàn bộ.

## Ngoại lệ thiết bị: 4 ảnh 4608 px

Hai nguồn `21d60e0822064acb` (4608×3456) và `ca6e0d79c7cc20bd` (3456×4608) — 2 ảnh clean + 2 ảnh degraded — không chạy được với
protocol cố định: resize "cạnh ngắn 736" không thu nhỏ ảnh lớn nên T4 15 GB hết VRAM, còn CPU cần **50,7 GB RAM** (đo thật, 735 s/ảnh)
vượt RAM Kaggle ~30 GB. Không resize, không bỏ ảnh: 4 ảnh được chạy **CPU trên máy local 64 GB** bằng đúng `OfficialDetector`
(cùng commit/config/weights, paddle 3.3.1 CPU, Python 3.13) và nạp qua `CS431_PRECOMPUTED_OOM` — chỉ dùng khi worker GPU báo OOM
cho ảnh có cùng SHA256; file này được gắn vào model identity (`precomputed_oom_predictions.json`).

- Kết quả lượt chạy: **19.792 ảnh GPU, 4 ảnh `cpu_local_precomputed`, 0 ảnh retry CPU Kaggle** (`inference_timing.json`).
- Tương đương CPU/GPU: trên 40 ảnh ICDAR2015, prediction CPU local và GPU Kaggle trùng toạ độ tuyệt đối 40/40, score lệch ≤ 5×10⁻⁶.
  Đây là bằng chứng trên tập mẫu, không phải chứng minh cho mọi ảnh.
- 4 ảnh chiếm 4/19.796 lượt. Chưa có đối chứng GPU cho chính 4 ảnh này nên chưa định lượng được ảnh hưởng của ngoại lệ CPU lên metric tổng.

## Bằng chứng kiểm tra

- Predictions đủ **19.796 sample_id**, không thiếu/trùng/thừa, đúng tập của manifest; một model identity; checkpoint/manifest hash đúng.
- Manifest `94e3a2ce…f242`, protocol `0530adc2…`, evaluator `ea25c8d9…` khớp `configs/shared_lock.json`; `run.py compare` chấp nhận.
- **Đã chấm lại toàn bộ 19.796 ảnh từ raw predictions** trên máy local (NumPy 2.3.5, Shapely 2.2.0) bằng evaluator đã freeze:
  tổng counts, P/R/Hmean, coverage, size bins, strata và CI bootstrap **khớp tuyệt đối**; 14 giá trị `predicted_roi_area_fraction`
  lệch 2,2×10⁻¹⁶ do làm tròn số thực (`recheck_verification.json`).
- Inference 2.624 s (~44 phút) trên 2×T4, chưa gồm setup/chấm điểm. Latency trung bình (gồm decode/pre/post) 247,0 ms clean /
  256,1 ms degraded mỗi ảnh trên một T4; 4 lượt precomputed được tính bằng thời gian lần thử GPU OOM. Không xếp hạng tốc độ với Monkey
  từ các lượt khác nhau.
- ZIP gốc `CS431_Paddle_Phase2_COMPLETE.zip` SHA256 `33b2ff82…09f5`; `predictions.jsonl` SHA256 `aad15863…ea4a`. Raw predictions,
  per-image và overlays không commit (chứa pixels/kích thước lớn); giữ ngoài git.

## Giới hạn

Toàn bộ MAIN là `realistic_mix` L2, `severity_basis = image_relative_fallback`; suy thoái tổng hợp, không phải ảnh chụp thực tế,
không suy ra đường cong theo severity. Metric là external TextOCR word-level polygon evaluation có ROI/ignored policy của đồ án,
không phải benchmark chính thức của Paddle; granularity dòng-vs-từ làm Hmean của Paddle thấp hơn khả năng định vị vùng chữ thực tế.
Đã xem kết quả toàn tập nên không gọi tập này là final test chưa từng phân tích; nếu cải tiến/fine-tune ở Phase 3 phải chia nguồn
train/dev/test riêng và chạy lại cả hai model trên cùng split.

## Bước tiếp theo

Trả lời RQ trong phạm vi tập/protocol này: cả hai checkpoint suy giảm mạnh trên ảnh suy thoái (Hmean giảm ~51–52 % tương đối),
nặng nhất ở recall và chữ nhỏ. Theo IoU-word Monkey tốt hơn tuyệt đối; theo coverage Paddle giữ vùng chữ tốt hơn khi suy thoái.
Các điểm yếu Paddle gồm chữ nhỏ <16 px, mất toàn bộ prediction trên 14 % ảnh degraded và granularity dòng/từ. Nếu Phase 3 cải tiến Monkey, cần phân tích bằng chứng lỗi riêng của Monkey trước khi chọn hướng; không mặc định Monkey có cùng lỗi với Paddle.

Manifest SHA256: `94e3a2ce989ce6a298d7ea30e7ce70f22ac676092b29df808bc3d33b426bf242`.
Protocol SHA256: `0530adc27e230e9dafa4e4019070ace1de3673f493ae4ea70b45a32e45f7af81`.
Evaluator SHA256: `ea25c8d9fc240ec10dc6eb45883b163c8e92952ee885423985e6d013f026fcdc`.
