# Đánh giá Phase 3: DBNet + MonkeyOCRv2-AS trên TextVQA clean/degraded

Người thực hiện: Tuấn. Job HPC: 7957. Run: `seed42_v2`.
Slurm báo COMPLETED, ExitCode 0:0, thời gian toàn suite 02:17:02.
Nguồn phân tích: ZIP kết quả tổng hợp do Tuấn gửi ngày 31/10/2026.

## 1. Kết luận chính

E2 (fine-tune clean + degraded) đạt Hmean degraded cao nhất: **41,729%**, tăng
**6,182 điểm phần trăm (pp)** so với checkpoint gốc 35,548%. Đây là tăng khoảng
17,39% tương đối, không phải tăng 17,39 pp. Đổi lại, clean Hmean giảm 4,105 pp.

E1 (fine-tune clean) đạt degraded Hmean 38,727%, tăng 3,180 pp; clean Hmean
giảm nhẹ hơn, 0,976 pp. Nếu coi clean/degraded quan trọng ngang nhau và lấy
trung bình hai Hmean, E1 và E2 gần như ngang nhau, E1 nhỉnh 0,063 pp.

Tiling global + overlapping 2×2 không cải thiện Hmean ở cấu hình đã thử.
Recall tăng nhưng precision giảm mạnh; clean và degraded Hmean đều giảm.
Đây là kết quả âm của cấu hình này, không chứng minh mọi cách tiling đều thất bại.

## 2. Phạm vi và những gì đã kiểm tra

- Master có 9.898 source hợp lệ sau giao TextVQA, dữ liệu degraded và TextOCR.
- Split theo source ID và source bytes SHA256, seed 42: train 6.928,
  validation 1.484, test 1.486 source. Mỗi source test có clean và degraded:
  tổng 2.972 trường hợp ảnh, 46.883 GT valid mỗi condition.
- Vì E1/E2 đã dùng train để fine-tune, bảng này đánh giá test 1.486 source,
  không phải toàn bộ 10.000 ảnh gốc hoặc toàn master 9.898 source.
- Cả bốn mục trong JSON ghi cùng test manifest SHA256
  `7b7a32769caf6eeb3af8437fc1719a2e9ad2ab8713483b906a8e157920e23bdb`.
- Cùng protocol SHA256
  `0530adc27e230e9dafa4e4019070ace1de3673f493ae4ea70b45a32e45f7af81`
  và evaluator SHA256
  `ea25c8d9fc240ec10dc6eb45883b163c8e92952ee885423985e6d013f026fcdc`.
- Đã tính lại P = TP/pred, R = TP/GT, Hmean = 2TP/(pred+GT) từ counts trong
  JSON; tất cả khớp metric JSON và CSV. Đã kiểm tra chênh lệch Hmean với E0.
- Chưa có raw predictions, per-image metrics, training/validation logs và
  checkpoint-selection artifacts trong ZIP 8,8 KB này. Vì vậy đây là kiểm tra
  tính nhất quán báo cáo, **chưa phải tái chấm độc lập từ prediction polygons**
  hay kiểm tra đầy đủ lịch sử training.

Không so trực tiếp số Hmean E0 ở bảng này với Phase 2 trên toàn master để
kết luận model đã tăng/giảm: hai tập đánh giá có thành phần nguồn khác nhau.

## 3. Bảng kết quả chính

Các P/R/H dưới đây tính bằng %. Δ là chênh lệch so với E0 global, tính bằng pp.

| Phương án | Clean P | Clean R | Clean H | Degraded P | Degraded R | Degraded H | ΔH clean | ΔH degraded |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| E0 global | 78,376 | 66,461 | 71,928 | 67,997 | 24,064 | 35,548 | 0 | 0 |
| E0 global + 2×2 | 59,852 | 72,813 | 65,699 | 38,749 | 25,630 | 30,853 | −6,229 | −4,695 |
| E1 clean | 69,080 | 72,928 | 70,952 | 54,657 | 29,987 | 38,727 | −0,976 | +3,180 |
| E2 clean + degraded | 64,090 | 72,018 | 67,823 | 51,336 | 35,151 | 41,729 | −4,105 | +6,182 |

Paired source bootstrap, 1.000 lần lấy mẫu, CI 95% của ΔH so với E0:

| Phương án | ΔH clean và CI 95% (pp) | ΔH degraded và CI 95% (pp) |
|---|---|---|
| 2×2 | −6,229 [−7,625; −4,750] | −4,695 [−5,732; −3,736] |
| E1 | −0,976 [−1,729; −0,215] | +3,180 [+1,847; +4,728] |
| E2 | −4,105 [−5,328; −2,997] | +6,182 [+4,255; +8,263] |

Các CI trong báo cáo không chứa 0. Chúng phản ánh biến thiên khi lấy lại mẫu
source test, không bao gồm biến thiên do đổi seed training. Hiện mới có một
seed. Không suy ra CI E2−E1 bằng cách trừ hai CI so với E0; báo cáo chưa có
paired bootstrap trực tiếp cho E2−E1.

## 4. Vì sao 2×2 không hiệu quả trong lần thử này?

Cấu hình đã thử: global view + 4 ô, overlap 20% chiều ô, seam margin 2 px;
polygon NMS 0,3 được chọn trong {0,3; 0,5} trên 200 source validation.
Test không được dùng để chọn NMS.

Trên degraded, recall tăng 24,064% → 25,630%, nhưng precision giảm
67,997% → 38,749%. Counts sau quy tắc ignore/ROI của evaluator:

| Phương án | TP | FP | FN |
|---|---:|---:|---:|
| E0 global | 11.282 | 5.310 | 35.601 |
| E0 global + 2×2 | 12.016 | 18.994 | 34.867 |
| E1 | 14.059 | 11.663 | 32.824 |
| E2 | 16.480 | 15.622 | 30.403 |

2×2 thêm ròng 734 TP nhưng thêm 13.684 FP. Trên clean cũng tương tự:
TP tăng 31.159 → 34.137, FP tăng 8.597 → 22.899. Tăng recall không đủ bù
tổn thất precision.

FP ở đây là prediction không match GT theo protocol, chưa đồng nghĩa tất cả
là nhiễu nền. Có thể gồm duplicate, polygon định vị kém, word/line mismatch
hoặc vùng khác; phải xem prediction overlays mới phân loại được nguyên nhân.

Latency degraded trung bình được báo: global 111,53 ms/ảnh, 2×2 507,61 ms/ảnh,
khoảng 4,55 lần. Tiled timing có cộng global timing đã cache và thời gian ô
đo mới; đây không phải phép đo end-to-end toàn job. Cấu hình tiling hiện tại
không có lợi về Hmean hoặc latency, dù có lợi về recall.

## 5. E1 và E2 thực sự cải thiện gì?

E1 chỉ dùng clean để training nhưng degraded Hmean vẫn tăng. Đây là bằng chứng
rằng thích nghi với miền ảnh/word GT hiện tại đã có ích cho ảnh degraded;
không được quy toàn bộ phần tăng của E2 cho việc dùng degraded training.

E2 so với E1: degraded Hmean cao hơn 3,002 pp, clean Hmean thấp hơn 3,129 pp.
Hai lượt có cùng optimizer-update/effective-batch budget nhưng phân bổ lượt
xem clean/degraded khác nhau. Cần paired per-image comparison để đánh giá
độ chắc chắn của chênh lệch trực tiếp này.

Cả E1/E2 tăng TP và recall nhưng cũng tăng FP. Riêng E2 trên clean,
recall tăng 66,461% → 72,018% trong khi precision giảm 78,376% → 64,090%.
Vì vậy chưa đủ bằng chứng để kết luận model đơn thuần “quên clean”: cần
phân tích precision/recall, confidence và loại lỗi định vị trước.

| Phương án | Trung bình H clean/degraded (%) | Mức giảm H do degraded (pp) |
|---|---:|---:|
| E0 global | 53,738 | 36,380 |
| E0 + 2×2 | 48,276 | 34,847 |
| E1 | 54,840 | 32,225 |
| E2 | 54,776 | 26,094 |

Gap của E2 thu hẹp 10,287 pp so với E0: gồm degraded tăng 6,182 pp và clean
giảm 4,105 pp. Không gọi toàn bộ 10,287 pp là cải thiện robustness.
2×2 minh họa rõ nguy cơ này: gap nhỏ hơn dù Hmean của cả hai condition đều tệ hơn.

E2 là lựa chọn tốt nhất nếu ưu tiên degraded Hmean trong bốn phương án đã thử.
E1 có đánh đổi clean/degraded nhẹ hơn; E1/E2 gần như ngang nhau về trung bình
hai Hmean. E0 vẫn tốt nhất trên clean Hmean. Chưa có một phương án thắng mọi tiêu chí.

## 6. Chữ nhỏ còn là điểm yếu

Chữ nhỏ được định nghĩa bằng chiều cao bbox GT gốc <16 px, không theo kích
thước polygon sau suy thoái.

| Phương án | Recall chữ nhỏ clean (%) | Recall chữ nhỏ degraded (%) |
|---|---:|---:|
| E0 global | 60,828 | 10,398 |
| E0 + 2×2 | 72,810 | 11,950 |
| E1 | 67,515 | 14,741 |
| E2 | 65,341 | 19,312 |

E2 tăng recall chữ nhỏ degraded 8,915 pp, từ 1.956 lên 3.633 TP trong
18.812 GT nhỏ. Tuy vậy vẫn bỏ sót khoảng 80,69% chữ nhỏ theo tiêu chí IoU.
Trên tất cả kích thước degraded, E2 còn 30.403 FN/46.883 GT, tức 64,85%.
Đã có cải thiện nhưng bài toán chưa được giải quyết triệt để.

## 7. Hướng xử lý tiếp theo, chưa triển khai thêm

1. Giữ nguyên kết quả E0/2×2/E1/E2 hiện tại làm bảng đối chứng. Lưu checkpoint
   E1/E2 và bản freeze manifest/protocol để có thể tiếp tục mà không train lại.
2. Thu thêm raw test predictions, per-image metrics, checkpoint-selection và
   validation logs. Tái chấm độc lập; kiểm tra selected iteration, checkpoint
   hashes, budget, input IDs và GT visibility/ignore nhất quán.
3. Phân tích FP mới xuất hiện và TP mất/được theo cùng source/annotation ID.
   Ưu tiên 2×2 và E2, đồng thời chia theo kích thước chữ và nhóm suy thoái.
4. Với E2, giả thuyết tiếp theo đáng kiểm chứng là cải thiện precision và giữ
   clean Hmean, thay vì chỉ đẩy recall. Các thay đổi score/postprocessing hoặc
   training balance chỉ được chọn bằng validation và phải công bố như thí nghiệm
   mới, không sửa hồi tố baseline. Chưa có bằng chứng hướng cụ thể nào sẽ thắng.
5. Nếu tiếp tục tiling, cần kiểm tra duplicate/seam/word-line mismatch và độ tin
   cậy của prediction từ crop trước. Chưa có lý do chạy lại nguyên cấu hình 2×2
   này chỉ để mong kết quả tốt hơn.

Ưu tiên hiện tại là audit và phân tích lỗi từ artifacts đã có, trước khi tiêu
thêm lượt H200. Không tự fine-tune thêm hoặc triển khai E3 từ báo cáo này.

## 8. Giới hạn khi viết báo cáo đồ án

- Đây là synthetic realistic_mix L2, severity_basis=image_relative_fallback;
  không suy ra kết quả cho mọi corruption, mọi mức độ hay ảnh degraded thực.
- Master baseline đã được xem ở Phase 2. Test Phase 3 được giữ khỏi optimization,
  nhưng không phải test chưa từng được phân tích trong lịch sử. Các hướng mới
  hình thành sau khi xem bảng này cần công bố là phát triển tiếp sau quan sát test.
- TextOCR train có trong nguồn pretraining Monkey; không tuyên bố unseen dataset.
- Một seed training và bootstrap theo source không tương đương nhiều seed độc lập.
- Không có phép đo tốc độ training T4 tương ứng để quy đổi thời gian H200/T4.

## 9. Đoạn kết luận có thể đưa vào báo cáo

“Trên 1.486 nguồn ảnh test với cặp clean/degraded và cùng ground truth TextOCR,
DBNet + MonkeyOCRv2-AS có Hmean giảm từ 71,93% trên clean xuống 35,55% trên
degraded. Fine-tune trên clean cải thiện degraded Hmean lên 38,73%, trong khi
fine-tune trên clean và degraded đạt 41,73%, cao hơn baseline 6,18 điểm phần
trăm. Tuy nhiên clean Hmean của phương án mixed giảm còn 67,82%, thể hiện đánh
đổi giữa khả năng phát hiện trên clean và degraded. Suy luận kết hợp global
và ô 2×2 tăng recall nhưng làm precision giảm mạnh, dẫn tới Hmean thấp hơn
baseline. Các kết quả cho thấy fine-tune có khả năng giảm tác động của suy
thoái trong thiết lập đã khảo sát; cải thiện precision và khả năng phát hiện
chữ nhỏ là những vấn đề còn cần nghiên cứu.”
