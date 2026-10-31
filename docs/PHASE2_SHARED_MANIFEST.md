# Quy tắc shared manifest Phase 2

Bắt buộc đọc `shared/Shared_Manifest_TextVQA_TextOCR_Public.ipynb`, protocol và data
audit. Không chỉ dựa vào đoạn code minh họa trong handoff lịch sử.

Manifest reference đã chạy/review Monkey: `94e3a2ce989ce6a298d7ea30e7ce70f22ac676092b29df808bc3d33b426bf242`.
Protocol: `0530adc27e230e9dafa4e4019070ace1de3673f493ae4ea70b45a32e45f7af81`.
Evaluator: `ea25c8d9fc240ec10dc6eb45883b163c8e92952ee885423985e6d013f026fcdc`.

## Nguồn dữ liệu và bytes dùng chung

Lấy giao TextVQA clean ∩ degraded MAIN có mapping metadata ∩ TextOCR train theo
source image ID. MAIN 10.000 source → giao có GT 9.915 → giữ 9.898 cặp có valid GT.
85 source không có TextOCR/GT và 17 source không có usable valid GT bị loại, có audit.
Không tạo lại degradation, không suy ID từ filename, không lấy question ID làm image ID.

Repo có gzip của đúng manifest reference, không Git LFS. Giải nén bằng
`python run.py shared unpack` và kiểm hash. Nó chứa GT polygon subset/ROI/homography,
relative paths và image hashes; không chứa pixels hay transcripts. Không cần parse lại
toàn bộ TextOCR JSON để chấm thông thường. Giữ raw JSON/metadata để audit nguồn khi cần.

`dataset_locations.json` là cấu hình local, không thuộc manifest identity. `clean_root`
phải là root sao cho `root / clean_relpath` trỏ đúng file; tương tự degraded_root. Không
sửa relative paths hoặc IDs để chữa đường dẫn. Worker hash từng file trước khi chạy.

## Geometry và evaluator khóa

Perspective dùng H từ corners đã lưu. Chỉ chấm common field of view qua các variants;
mask reflected borders. Từ có <80% diện tích còn thấy được ignore ở cả hai condition.
Valid annotation IDs clean/degraded giống nhau. Giữ chính xác Polygon/MultiPolygon.

Ignore transcript `.`, `###`, rỗng và cờ ignore. Prediction có <50% diện tích trong ROI
hoặc >50% overlap ignored GT được bỏ theo protocol. Polygon IoU ≥0,5, maximum-cardinality
matching. Word coverage ≥0,5 chỉ là diagnostic, xem cùng P/R/Hmean và predicted area fraction.
Không score sweep, không thêm score filter; giữ fixed postprocessing riêng mỗi detector.

Một clean inference/source; degraded inference/variant. Gap = degraded − clean,
đơn vị điểm phần trăm. Báo unique sources và variants, counts, paired bootstrap,
small-text bins và paired overlays. Dữ liệu hiện là realistic_mix L2 synthetic;
không suy thành nhiều severity hoặc captured real-world degradation.

## Tương thích Monkey Phase 3

Repo H200 đã có một biến thể số thực được review riêng với hash `ad43fa08...`.
Paddle Phase 2 trong CS431 chỉ chấp nhận reference `94e3a2ce...`, dùng để so đúng
Monkey Phase 2 gốc. Không tự chấp nhận hash gần giống/rounding hoặc ghép score
test split Phase 3 vào bảng full-master Phase 2. Muốn benchmark hai model trên split
Phase 3, freeze một protocol/bộ artifacts riêng và chạy cả hai trên đúng split đó.

Nếu cần rebuild do thay dữ liệu: đây là revision mới cần audit/review và kết quả
cho cả hai model; không ghi đè lock hiện tại để vượt kiểm tra.
