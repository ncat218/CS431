# Đánh giá Phase 2 — DBNet + MonkeyOCRv2-AS trên TextVQA clean/degraded

Người thực hiện: Tuấn. Review ngày 30/10/2026, dựa trên `phase2_monkey_full.zip` và notebook đã chạy `phase-2-dbnet-monkeyocrv2-as.ipynb`.

Lượt chạy đã hoàn tất baseline đánh giá checkpoint trên toàn bộ giao dữ liệu có GT hợp lệ: **9.898 ảnh clean và 9.898 ảnh degraded, tổng cộng 19.796 lượt inference**. Đây là đánh giá robustness trên dữ liệu của đồ án; kết quả tái hiện benchmark gốc thuộc Phase 1. Không phải tái hiện toàn bộ paper, training hay fine-tuning.

**Dữ liệu thực tế và lý do không chấm đủ 10.000 cặp**

| Bước | Số ảnh nguồn |
|---|---:|
| MAIN của bộ suy thoái | 10.000 |
| Giao có TextOCR annotations trước kiểm tra GT hợp lệ | 9.915 |
| Cặp có ít nhất một GT chữ hợp lệ để chấm | 9.898 |

85 ảnh có lý do `not_in_textocr_or_no_gt`; 17 ảnh có lý do `no_usable_valid_gt`. Cả hai phía clean/degraded đều loại cùng source IDs. Không có loại ảnh vì lỗi file/geometry trong audit đã lưu. Chi tiết từng ID nằm trong `shared/excluded.jsonl`. Không thay các ảnh này bằng ảnh khác để đạt số lượng tròn.

Mỗi phía có **318.490 GT chữ hợp lệ** và **150.383 GT ignored** trong manifest cuối. Valid/ignored annotation IDs của từng cặp giống nhau. Polygon degraded được biến đổi theo homography, không giữ nguyên tọa độ khi ảnh có perspective. ROI chung loại vùng phản chiếu ngoài field of view. Việc loại 17 ảnh không có valid GT cũng có nghĩa bộ đánh giá này không đo false positives trên những ảnh đó.

Schema TextOCR thực tế dùng `imgs`, `anns`, **`imgToAnns`**. Parser mới đã đối chiếu index này với `anns[*].image_id`; không suy ID từ tên file và không thay polygon bằng bbox. Lỗi trước đây do parser chỉ chấp nhận tên `img2Anns`.

**Kết quả chính — polygon IoU ≥ 0,5, cùng protocol**

| Metric | Clean (%) | Degraded (%) | Degraded − clean (điểm %) | Khoảng tin cậy 95% của chênh lệch |
|---|---:|---:|---:|---|
| Precision | 77,408 | 67,535 | −9,873 | [−11,178; −8,549] |
| Recall | 65,148 | 22,343 | −42,805 | [−44,209; −41,362] |
| Hmean/F1 | 70,751 | 33,578 | −37,173 | [−38,820; −35,522] |
| Word coverage recall, diagnostic | 75,255 | 25,669 | −49,586 | [−51,361; −47,846] |

Khoảng tin cậy dùng 1.000 lần cluster bootstrap theo source ID, seed 42. Mỗi source hiện chỉ có một degraded variant nên paired source-weighted micro trùng với clean/degraded micro. Đây là suy luận theo phép resampling của tập này, không đại diện tự động cho mọi ảnh suy thoái ngoài thực tế.

Suy thoái làm chất lượng giảm rõ rệt trong cấu hình đang kiểm tra, với vấn đề nổi bật là **bỏ sót chữ**. True positives giảm từ 207.491 xuống 71.161; false negatives tăng từ 110.999 lên 247.329. Hmean giảm 37,17 điểm phần trăm, tương đương giảm tương đối khoảng 52,54% so với clean; cần ghi rõ hai đơn vị khác nhau.

**Phân tích lỗi đã xác nhận từ artifacts**

- Có 7.793/9.898 nguồn bị giảm recall, 1.792 không đổi và 313 tăng. Không phải mọi ảnh đều giảm.
- 139.291 GT đã match trên clean bị mất match trên degraded; chiều ngược lại có 2.961 GT mới match được. Chênh lệch ròng là 136.330 true positives.
- Ảnh không có bất kỳ raw prediction nào tăng từ 167 (1,69%) lên 1.845 (18,64%). Raw polygons giảm từ 310.268 xuống 110.326. Sau các quy tắc ignored/ROI cố định, số predictions được chấm là 268.050 và 105.369; không có polygon malformed bị tính FP trong lượt này.
- Với GT có chiều cao bounding box gốc <16 px, recall giảm từ 60,13% xuống 9,92%. Nhóm 16–32 px giảm 72,71% xuống 28,78%; nhóm ≥32 px giảm 62,75% xuống 35,34%. Đây là bin theo độ cao bounding box của polygon trong ảnh gốc, chưa phải kích thước glyph đo độc lập hay thí nghiệm kiểm soát các yếu tố khác.
- Trong 5.974 cặp không perspective, Hmean vẫn giảm 70,58% xuống 30,89%. Vì vậy, suy giảm không chỉ xuất hiện trong nhóm cần warp GT. Nhóm 3.924 cặp có perspective giảm 71,00% xuống 37,30%. Hai nhóm có thành phần phép suy thoái khác nhau; không kết luận perspective có tác dụng bảo vệ mô hình.
- Nhóm hỗn hợp `defocus_blur + lowlight_sensor_noise` gồm 287 nguồn có Hmean clean 71,22%, degraded 6,24%. Đây là tín hiệu mô tả để ưu tiên phân tích, chưa chứng minh tác động riêng của blur hoặc lowlight. Bảng toàn bộ 43 hỗn hợp nằm trong `audit_validation.json`, trường `operation_mixtures`.

Đã xem trực tiếp ba cặp overlays: hai cặp lỗi detection và một cặp GT có perspective. Ví dụ chữ viết tay bị tối/nhiễu và biển EXIT bị mờ cho thấy vùng GT vẫn hiện diện nhưng detector bỏ sót. Cặp perspective được xem có GT bám vị trí chữ sau biến đổi. Đây là kiểm tra mẫu có chủ đích, không phải audit thị giác toàn bộ 9.898 cặp.

**Bằng chứng kiểm tra**

- 147 file trong inventory khớp SHA256 và kích thước. 13 code cells của notebook đều có execution count; không có output kiểu `error`.
- Predictions đủ đúng 19.796 sample IDs, không thiếu/trùng/thừa. Hai shards khớp file merge; metadata của mỗi prediction khớp source ID, condition, manifest, checkpoint và config.
- Checkpoint hash, config hash và source revisions khớp Phase 1. Strict checkpoint load thành công. Đã đối chiếu resolved model và test pipeline với cấu hình gốc đã lưu ở Phase 1, có chuẩn hóa tuple/list do JSON serialization.
- Đối chiếu toàn bộ valid/ignored GT và ROI qua phép chiếu độc lập từ homography lưu trong manifest: sai khác tọa độ tối đa 0 px. Điều này xác nhận tính nhất quán của artifacts, không thay thế đối chiếu lại toàn bộ metadata với ảnh nguồn bên ngoài ZIP.
- Đã cộng lại TP/GT/pred/covered từ toàn bộ per-image results, đối chiếu matched/missed IDs và tổng metric: khớp.
- **Đã chấm lại toàn bộ 19.796 ảnh từ raw predictions bằng evaluator đã lưu**: per-image results, tổng metrics và khoảng tin cậy bootstrap 1.000 lần đều khớp kết quả Kaggle. Bản chấm lại dùng NumPy 2.3.5, Shapely 2.2.0; xem `recomputed_metrics/verification.json`. Script `recheck_streaming.py` đọc manifest từng dòng và index predictions bằng SQLite để giới hạn RAM; không chạy lại model.

Notebook ghi nhận hai Tesla T4. Full inference mất **8.319 giây, khoảng 2 giờ 18 phút 39 giây**, chưa gồm setup, audit và evaluation. Toàn notebook từ code cell đầu đến cell cuối khoảng 3 giờ 9 phút. Peak allocated VRAM khoảng 10,1 GiB/GPU. Đây là thời gian của lượt chạy này, không phải cam kết cho mọi session Kaggle.

**Giới hạn khi viết báo cáo nghiên cứu**

Toàn bộ MAIN được chấm là **`realistic_mix`, mức `L2`**. Tất cả 9.898 nguồn ghi **`severity_basis = image_relative_fallback`**. Không gọi mức L2 này là đã chuẩn hóa theo kích thước chữ; không suy ra đường cong theo nhiều severity hoặc khả năng chịu từng phép suy thoái riêng. Dữ liệu là suy thoái tổng hợp, không phải ảnh suy thoái được chụp thực tế. Một số ảnh có thể mất thông tin chữ đến mức khó phục hồi; chưa thể quy mọi false negative thành lỗi có thể khắc phục của detector.

Metric ở đây là external TextOCR word-level polygon evaluation, có ROI/ignored policy của đồ án. Không so trực tiếp Hmean 70,75% với Hmean ICDAR2015 khoảng 88,99% ở Phase 1 để kết luận model bị thay đổi. Không sweep confidence threshold trên bộ này. Word coverage là diagnostic và có thể tăng giả nếu box quá lớn; kết quả chính vẫn phải xem Precision/Recall/Hmean và overlays cùng nhau.

TextOCR train thuộc nguồn dữ liệu pretraining đã được đối chiếu trước đây cho MonkeyOCRv2; không gọi đây là tập hoàn toàn unseen hay zero-shot. Lượt chạy hiện tại không fine-tune. Nếu phát triển cải tiến/fine-tune sau này, phải chốt nhóm source IDs cho train/development/final evaluation và giữ mọi clean/degraded variant của cùng nguồn trong cùng nhóm. Đã xem kết quả toàn tập nên cần công khai việc này, không mô tả toàn bộ tập như một final test chưa từng được phân tích.

Ảnh input gốc, full annotation JSON, checkpoint bytes và toàn bộ frozen degradation config không nằm trong ZIP review. Các kiểm tra nguồn đó dựa trên audit/hash và log Kaggle đã lưu; máy review không chạy lại GPU inference. Nếu cần phân tích mức suy thoái vật lý, cần dùng frozen config và metadata gốc đã có của Tuấn; không tạo lại dữ liệu.

**Bước tiếp theo**

Có đủ bằng chứng để trả lời RQ ban đầu trong phạm vi tập và protocol này: chất lượng detection của checkpoint giảm mạnh trên ảnh suy thoái, đặc biệt ở recall và chữ nhỏ. Chưa có kết quả Paddle để so sánh hai phương pháp, và chưa chọn cải tiến chỉ từ bảng điểm này.

Giữ nguyên baseline cùng manifest/protocol/evaluator. Bên Paddle cần dùng đúng các file đã freeze, không lấy mẫu lại. Tiếp tục phân tích lỗi theo hỗn hợp, chữ nhỏ, mức mất thông tin và mức gộp từ/dòng; từ đó mới chọn giả thuyết cải tiến và thiết kế ablation/split. Chưa cần chạy lại full Monkey hay fine-tune để xác nhận lượt baseline này.

Manifest SHA256: `94e3a2ce989ce6a298d7ea30e7ce70f22ac676092b29df808bc3d33b426bf242`.

Protocol SHA256: `0530adc27e230e9dafa4e4019070ace1de3673f493ae4ea70b45a32e45f7af81`.

Evaluator SHA256: `ea25c8d9fc240ec10dc6eb45883b163c8e92952ee885423985e6d013f026fcdc`.
