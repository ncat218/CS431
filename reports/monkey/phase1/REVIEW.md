# Đánh giá Phase 1 — Tuấn / DBNet + MonkeyOCRv2-AS

Ngày kiểm tra: 30/10/2026. Đầu vào: `phase1_monkey_artifacts.zip` do Tuấn gửi sau khi chạy notebook trên Kaggle.

## Kết luận

**Đủ bằng chứng để đánh giá đã hoàn thành phần official-checkpoint evaluation reproduction của DBNet + MonkeyOCRv2-AS trên ICDAR2015 test.** Kết quả rất gần bảng công bố chính thức. Không phải reproduce toàn bộ pretraining/training, không phải đánh giá robustness trên TextVQA và không tự phê duyệt Phase 2.

| Metric | Official (%) | Run của Tuấn (%) | Run − official (điểm phần trăm) |
|---|---:|---:|---:|
| Precision | 91,300 | 91,291 | −0,009 |
| Recall | 86,700 | 86,808 | +0,108 |
| Hmean | 88,900 | 88,993 | +0,093 |

Nguồn đối chiếu: [MonkeyOCRv2 detection model card — ICDAR2015](https://huggingface.co/HB16888/MonkeyOCRv2_det#icdar2015). Số đo thực nghiệm nằm trong [official_metrics.json](official_metrics.json) và [log evaluation](official_eval.log).

Hmean đo chính xác là 0.8899308983218164. Chênh lệch so với mốc 0.889 là +0.093089832 điểm phần trăm, nhỏ hơn mức 0.5 điểm phần trăm được đề xuất trong handoff. Quy ước 0.5 chưa được nhóm phê duyệt; đây không phải tolerance do tác giả paper công bố. Không kết luận kết quả trùng tuyệt đối hoặc quy nguyên nhân sai khác cho rounding/precision khi chưa có bằng chứng.

## Bằng chứng đã kiểm tra

- Log full test kết thúc ở `Epoch(test) [500/500]`; kết quả log khớp raw metrics và JSON tổng hợp.
- `full/predictions.jsonl` có đúng **500 image paths duy nhất**, khớp đầy đủ tập ảnh trong dataset audit và hook output, không thiếu/trùng. Có **3.872 polygons**; các tọa độ hữu hạn và số scores khớp polygons. Đây là số dự đoán xuất thô, không phải số TP sau lọc của evaluator.
- Dataset audit ghi **2.077 valid GT** và **3.153 ignored GT**; có hash annotation và hash từng ảnh. Preparer sử dụng các nguồn ICDAR2015 chính thức trong dataset-zoo config.
- [Strict checkpoint load](logs/09_strict_checkpoint_load.log) thành công, không chấp nhận detection head thiếu weights. Checkpoint đúng tên `dbnet_mkv2vitae_icdar2015.pth`.
- Đối chiếu AST giữa resolved official config và full config: `model`, `test_pipeline`, `test_evaluator`, `test_cfg` bằng nhau; full dataset không có subset indices. Overrides phục vụ đường dẫn dữ liệu, batch size 1, workers và hook xuất artifact.
- Evaluator là **HmeanIOUMetric**, strategy `vanilla`, `match_iou_thr=0.5`, `ignore_precision_thr=0.5`. Score sweep giữ nguyên evaluator; log có các mức hiển thị 0.30–0.90, Hmean cao nhất ở 0.60. Không dùng evaluator TextOCR/Phase 2 thay cho metric này.
- Preflight xác nhận CUDA toolkit **12.8**, không dùng nhánh mismatch. Core stack khớp hướng dẫn: Python 3.11, torch 2.9.0+cu128, MMCV 2.0.1, MMEngine 0.10.7, MMDetection 3.1.0, MMOCR 1.0.1, Transformers 4.57.1. CUDA NMS và modulated deformable convolution đã chạy được.
- Máy có 2 Tesla T4; tiến trình evaluation chỉ thấy GPU 0, phù hợp baseline single-GPU.
- Đối chiếu **112 file** với inventory: không thiếu file và không lệch size/SHA256. Inventory không phải chữ ký xác thực độc lập từ tác giả.
- Có **20 full overlays**. Đã xem trực tiếp 4 ảnh `img_147`, `img_192`, `img_38`, `img_430`: các vùng GT/prediction nhìn chung cùng hệ tọa độ, không thấy sai lệch toàn ảnh do resize/inverse transform. Có vùng bỏ sót/dự đoán thừa, phù hợp detection chưa hoàn hảo; không dùng bốn ảnh để suy rộng về robustness.

## Revisions và hashes cần giữ

| Artifact | Giá trị |
|---|---|
| MonkeyOCRv2 commit | `6cc0c0bb4864f2541fa6bc3a8f976cffaac8f359` |
| MMOCR commit | `1dcd6fa6958de22bcb997319833f0ac19c180ec7` |
| MMCV commit | `fc038a386a42d1f36d9ec49169d5817a52c70b8e` |
| Detection HF revision | `58c8bd712a1757a1d21b729ab69910573ecc325b` |
| Backbone HF revision | `51778e7738fdad36ed72288881071b4dcd04b127` |
| Detector SHA256, được ghi khi chạy | `4b13e26ad5c2e1b13fef45ac608298254d083062d2b9cafba706064bd8e2cf82` |
| Converted test annotation SHA256 | `fd1a9d52dbd2b3ed86a16fef27cda35e4c44624d07c4f1c25338ea2ca660f718` |

## Phạm vi và giới hạn của kiểm tra này

Đây là audit artifact của run đã hoàn thành, **không chạy lại model/evaluator trên máy local**. ZIP không chứa checkpoint binary, toàn bộ ảnh test hay polygons GT gốc nên chưa thể tính lại metric hoàn toàn độc lập hoặc tự tính lại hash checkpoint từ binary. Hash checkpoint ở trên là bằng chứng được notebook ghi tại thời điểm chạy; không gọi đó là đối chiếu hash độc lập với tác giả.

473 giây là wall time full evaluation **có export predictions/overlays và evaluator**, không phải phép đo thuần model latency. GPU memory khoảng 2.167 MB trong log không thay thế một benchmark tài nguyên riêng.

Trạng thái `EVALUATED_REQUIRES_GROUP_REVIEW` / `group_tolerance_approved=false` trong JSON là gate review chưa được người dùng phê duyệt, không biểu thị evaluation lỗi. Không sửa các artifact gốc để đổi trạng thái sang PASS.

## Cách ghi trong báo cáo

> Nhóm tái hiện phần evaluation của DBNet sử dụng backbone MonkeyOCRv2-AS trên 500 ảnh ICDAR2015 test bằng checkpoint chính thức và HmeanIOUMetric của MMOCR. Kết quả Precision 91,291%, Recall 86,808%, Hmean 88,993%, so với số công bố tương ứng 91,3%, 86,7%, 88,9%. Nhóm không tái huấn luyện mô hình từ đầu.

Phase 1 của Monkey có thể đề nghị nghiệm thu về mặt kỹ thuật. Phase 1 của Paddle và quyền chuyển Phase 2 cần được nhóm xác nhận riêng. Không tự lấy score threshold 0.60 tối ưu trên ICDAR2015 làm threshold chung cho hai mô hình trên TextVQA; Phase 2 phải khóa protocol riêng theo kế hoạch.
