# HANDOFF — Reproduce 2 Text Detectors → Benchmark gốc → TextVQA Clean/Degraded (TextOCR GT)

**Ngày soạn:** 30/10/2026  
**Đề tài học phần:** *Robust Scene Text Detection under Realistic Image Degradation*  
**Mục tiêu ngay lúc này:** phát hiện đúng **vị trí/vùng chữ**, chưa nghiên cứu nhận dạng nội dung, chưa triển khai restoration, chưa chạy Qwen-VL.  
**Đối tượng thực nghiệm:** `PP-OCRv5_server_det` (thuộc PaddleOCR 3.0) và `DBNet + MonkeyOCRv2-AS` (MonkeyOCRv2).  
**Quan trọng:** Chỉ thực hiện Phase 2 sau khi Phase 1 được **xác minh và được trưởng nhóm phê duyệt**. Không tự kết luận reproduce thành công nếu không có đúng dữ liệu/evaluator của paper.

> [!IMPORTANT]
> **Cập nhật quyết định cuối cùng:** Phase 2 bắt buộc dùng **MỘT CODE TẠO MANIFEST CHUNG**, lấy **giao của ba dataset**: TextVQA clean ∩ TextVQA degraded 10K ∩ TextOCR v0.1 train JSON theo `image_id` gốc. Hai model phải chạy **chính xác cùng tập ảnh và cùng metric script**. Không tạo lại degradation: ~10.000 ảnh suy thoái **đã được nhóm tạo và upload lên Kaggle**. File handoff cũ chỉ dùng để hiểu thuật toán degradation, không dùng số lượng/manifest cũ làm trạng thái hiện tại.

## A. Nguồn dữ liệu, paper và phân công

### Dữ liệu Phase 2 — chỉ dùng ba nguồn này

| Vai trò | Kaggle dataset | Ghi chú |
|---|---|---|
| Ảnh clean | https://www.kaggle.com/datasets/anhthu128/text-vqa-train/ | TextVQA **không suy thoái** |
| Ảnh degraded | https://www.kaggle.com/datasets/anhthu128/textvqa-suy-thoi-10k-nh | Khoảng 10.000 ảnh tự tạo, không generate lại |
| Ground truth | https://www.kaggle.com/datasets/anhthu128/textocr-0-1-train-json | TextOCR **polygon của chữ**, không dùng VQA answer làm detection GT |

**Phải inspect thực tế `/kaggle/input/` trước khi viết đường dẫn.** Không suy đoán cấu trúc thư mục, tên file, hoặc cách parse `image_id` từ filename degraded. Nếu không tìm thấy mapping degraded → ảnh gốc/transform metadata, hỏi Tuấn trước khi lập manifest. Các URL có thể xem công khai nhưng không xác minh được sơ đồ file trực tiếp từ giao diện web ở thời điểm soạn.

### Paper và benchmark Phase 1

| Người | Mô hình | Benchmark reproduction ưu tiên | Kết quả paper/official để đối chiếu |
|---|---|---|---|
| **Tuấn** | **MonkeyOCRv2-AS + DBNet** | **ICDAR2015 test** (MMOCR) | **P 91,3% · R 86,7% · Hmean 88,9%** |
| **Trí hoặc Triệu** — hai bạn tự chọn một người | **PP-OCRv5 Server Text Detector** | **Bộ đánh giá PaddleOCR 3.0** nếu có đầy đủ ảnh + GT + evaluator chính thức | **Hmean 83,8%** trên bộ đánh giá riêng của PaddleOCR |
| **Người còn lại (không nhận PP-OCRv5 Phase 1)** | **Chủ trì Phase 2** | Shared manifest, unified evaluator, clean/degraded data audit, báo cáo | Chấm điểm cho **cả hai model** bằng cùng protocol |

Phân công chi tiết: Tuấn phụ trách môi trường/checkpoint/config MonkeyOCRv2 và xuất predictions; người phụ trách Paddle tương tự. Ở Phase 2, người còn lại **sở hữu duy nhất code manifest + evaluator**. Hai model owner hỗ trợ tích hợp inference trong môi trường riêng. Không có hai phiên bản manifest/evaluator do hai người tự viết khác nhau.

### Tài liệu chính thức

- MonkeyOCRv2 repo: https://github.com/Yuliang-Liu/MonkeyOCRv2
- MonkeyOCRv2 detection checkpoint + hướng dẫn: https://huggingface.co/HB16888/MonkeyOCRv2_det
- PaddleOCR repo: https://github.com/PaddlePaddle/PaddleOCR
- PP-OCRv5 Text Detection API: https://www.paddleocr.ai/main/en/version3.x/module_usage/text_detection.html
- TextOCR v0.1 format: https://textvqa.org/textocr/dataset/

---

## B. Sơ đồ workflow tổng thể / gates

```text
PHASE 0 — Preflight, khóa tiêu chí
   Kiểm tra GPU/Kaggle, commit, phiên bản môi trường, checkpoint
   Khóa model variants + preprocessing + benchmark protocol
              |
              v
PHASE 1 — REPRODUCE RIÊNG TỪNG PAPER
   MonkeyOCRv2: official ICDAR2015 test + checkpoint + MMOCR evaluator
   PaddleOCR: official PaddleOCR3.0 benchmark NẾU truy cập được
   (nếu không: public benchmark validation, ghi rõ KHÔNG phải reproduce 83.8%)
              |
              v
   Đối chiếu số liệu paper / audit logs / visual inspection
              |
        PASS? | ----- NO ---> sửa config/dataset/checkpoint/evaluator
              |                 hoặc đánh dấu BLOCKED, xin quyết định từ nhóm
              v
PHASE 2 — DATA AUDIT + SHARED MANIFEST (DUY NHẤT)
     TextVQA clean ∩ TextVQA degraded 10K ∩ TextOCR GT
     + join source_image_id + kiểm tra geometric transforms
     + khóa manifest v1 + SHA256 + overlay inspection
              |
              v
   PP-OCRv5 và MonkeyOCRv2 (GIỮ NGUYÊN checkpoint/config)
   SAME clean images / SAME degraded variants / SAME GT / SAME evaluator
              |
              v
   Clean vs Degraded P/R/Hmean + Word Coverage Recall + Error Cases
              |
              v
PHASE 3 — CHƯA TRIỂN KHAI
   Khi có gap thật, chọn 1 limitation -> giải pháp -> ablation
```

**Khóa phạm vi:** Trong Phase 1–2 không train/fine-tune, không thay thresholds dựa trên kết quả test, không restoration và không chạy Qwen-VL.

---

## C. Phase 1 — MonkeyOCRv2 (Tuấn)

### C1. Reproduce cái gì?

- **Chính xác mô hình:** DBNet với visual backbone `MonkeyOCRv2-AS / ViTAEv2-S`, không dùng parser 0.7B, không dùng OCR recognition.
- **Benchmark tham chiếu:** ICDAR2015 **test** với cấu hình `dbnet_mkv2vitae_1200e_icdar2015_2gpu_adamw.py`, checkpoint tương ứng.
- **Kết quả đối chiếu:** Precision **91,3%**, Recall **86,7%**, Hmean **88,9%** theo `HmeanIOUMetric` của MMOCR. Đây là **evaluation reproduction**, không phải reproduce toàn bộ pretraining/training.

### C2. Thiết lập (ưu tiên T4×2 nếu tương thích)

Nguồn chính thức tham chiếu môi trường Python 3.11, PyTorch 2.9.0/CUDA 12.8, MMEngine 0.10.7, MMCV 2.0.1 (build from source), MMDetection 3.1.0, MMOCR 1.0.1 và Transformers 4.57.1. **Kaggle có thể không có toolchain CUDA tương thích**: kiểm tra và điều chỉnh theo README, không `pip install` bừa làm hỏng môi trường.

```bash
# Preflight trong Kaggle Notebook (bật Accelerator GPU T4 x2)
nvidia-smi
python - <<'PY'
import torch
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('visible gpus:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()): print(i, torch.cuda.get_device_name(i))
PY

# Chỉ thực hiện khi môi trường đáp ứng / đã thống nhất giải pháp build:
git clone https://github.com/Yuliang-Liu/MonkeyOCRv2.git
cd MonkeyOCRv2/detection/mmocr
# Tuân thủ phần Installation trong official README.
# README có install.sh (clone MMOCR + patch), source build MMCV.
# KHÔNG chạy ngay khi chưa kiểm tra bản CUDA trong Kaggle.

# Sau khi install.sh và cài đủ dependencies:
# Tải checkpoint đúng mô hình (checkpoint đã được công bố):
hf download HB16888/MonkeyOCRv2_det --include 'dbnet_mkv2vitae_icdar2015.pth' --local-dir ./mmocr/model_weight
hf download zenosai/MonkeyOCRv2-AS --local-dir ./mmocr/pretrained/monkeyocrv2_as

# Dataset phải ở format chính thức dưới ./mmocr/data/icdar2015/
# Dùng dataset preparation instructions từ repo, kiểm tra test split/GT.
cd mmocr

# Baseline SINGLE GPU: đúng evaluator gốc; chạy trước khi thử 2 GPU
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  configs/textdet/dbnet/dbnet_mkv2vitae_1200e_icdar2015_2gpu_adamw.py \
  model_weight/dbnet_mkv2vitae_icdar2015.pth

# OPTION: multi-GPU official evaluation (nếu single GPU test đã OK)
# Chỉ dùng khi available CUDA devices = 2 và official tool hoạt động.
CUDA_VISIBLE_DEVICES=0,1 bash tools/dist_test.sh \
  configs/textdet/dbnet/dbnet_mkv2vitae_1200e_icdar2015_2gpu_adamw.py \
  model_weight/dbnet_mkv2vitae_icdar2015.pth 2
```

**Báo cáo:** model SHA256/size, checkpoint URL + hash, repo commit SHA, chính xác config, installed packages, dataset/version/test image count, `P/R/Hmean`, log evaluator, 10–20 ảnh detection visualization.

**Gate Monkey:** Đối chiếu P/R/Hmean trên **cùng ICDAR2015 test, evaluator, config**. Mức sai khác chấp nhận đề xuất ≤0,5 điểm phần trăm cho Hmean; vượt mức phải debug. Ngưỡng này là **quy ước nội bộ cần cả nhóm chấp thuận**, không phải ngưỡng do paper đưa ra. Nếu checkpoint/dataset không match, báo blocker thay vì tuyên bố PASS.

### C3. Cần lưu ý về Kaggle T4×2

- Chỉ **inference/evaluation** được ưu tiên, không pretrain 113M ảnh.
- Với benchmark gốc, chạy 1 GPU đã có thể đủ; `dist_test.sh` có thể phân phối hai GPU nhưng phải xác nhận không đổi evaluator/kết quả.
- Vướng build/MMCV/CUDA thì chuyển sang H200 hoặc môi trường/container phù hợp **sau khi báo rõ vấn đề**. H200 không tự giải quyết lỗi phiên bản.

---

## D. Phase 1 — PaddleOCR 3.0 / PP-OCRv5 Server (Trí hoặc Triệu)

### D1. Reproduce cái gì?

- **Chỉ detector:** `PP-OCRv5_server_det`, không OCR recognizer.
- Official documentation công bố **Hmean 83,8%** trên bộ đánh giá mới của PaddleOCR 3.0 (2.677 ảnh, nhiều nguồn).
- **Chỉ được khẳng định reproduce đúng 83,8% nếu có đúng ảnh/annotation/protocol đánh giá gốc.** Nếu không truy cập được, chạy pretrained model trên benchmark public được nhóm chốt (ưu tiên ICDAR2015), chấm bằng evaluator thích hợp và ghi rõ **official-checkpoint validation / external benchmark**, chưa phải exact official reproduction.

### D2. Bước thực hiện

1. Ghi repo commit, PaddlePaddle/PaddleOCR versions, checkpoint official và hash.
2. Kiểm tra có được full 2.677 images/GT + evaluator mà tác giả dùng không. Nếu không, lập báo cáo blocker và chuyển phương án công khai có phê duyệt.
3. Chạy official pretrained Server Detector, kiểm tra `dt_polys`, `dt_scores` trên 20 ảnh.
4. Nếu benchmark gốc khả dụng, chạy official eval theo docs; nếu không, kiểm chứng có định lượng trên ICDAR2015 bằng code GT/evaluator riêng và ghi nhãn external.
5. Bàn giao code model-inference adapter và 10–20 visual examples.

```python
# PaddleOCR 3.x detector-only quick smoke test
# Cài PaddlePaddle GPU đúng phiên bản và CUDA theo tài liệu trước.
from paddleocr import TextDetection

model = TextDetection(model_name='PP-OCRv5_server_det', device='gpu:0')
for result in model.predict(input='/kaggle/input/EXACT_DATASET_PATH/sample.jpg', batch_size=1):
    print(result.json)   # includes dt_polys + dt_scores
```

**Gate Paddle:** Có reproducible result trên benchmark chính thức => Phase 1 PASS. Nếu bộ đánh giá 2.677 ảnh không public thì cột "Exact Hmean 83.8 reproduction" phải là **NOT VERIFIED**; chuyển Phase 2 **chỉ khi nhóm đồng ý bằng văn bản rằng checkpoint + public-benchmark validation đủ cho phạm vi đồ án**.

---

## E. Phase 2 — SHARED MANIFEST / INTERSECTION, người thứ ba phụ trách

### E1. Quy tắc cứng: giao ba dataset theo ảnh gốc

Với `C` = tập ảnh clean có thể mở, `D` = tập degraded có mapping nguồn ảnh, `G` = TextOCR `imgs` có `img2Anns` hợp lệ:

```text
S = unique_source_image_ids(C) ∩ unique_source_image_ids(D) ∩ G
```

**Không union, không sampling độc lập theo từng model, không dùng question_id thay image_id.** Nếu ảnh clean có trong TextVQA nhưng không có degraded hoặc không có GT TextOCR → **BỎ**. Nếu ảnh degraded không truy về chính xác ảnh gốc → **BỎ/QUARANTINE**, không đoán qua fuzzy matching. Nếu 1 ảnh gốc có 3 biến thể suy thoái thì manifest có 3 dòng paired; Clean inference chỉ chạy 1 lần cho image_id đó. Chưa biết số `unique_source_image_ids` thực tế: phải audit trước, không mặc định 10.000 ảnh gốc.

**TextOCR JSON v0.1** sử dụng cấu trúc `imgs`, `anns`, `img2Anns` với trường `anns[*].points` là polygon; `utf8_string` không phải detection GT. Giữ polygon làm GT vị trí. Một số vùng text illegible có transcript `'.'`: vẫn có polygon; phải chốt chính sách ignore/difficult và báo cáo.

### E2. Tạo hai CSV chỉ mục trước khi join

**(i) `clean_index.csv`**:

```csv
image_id,path
OpenImages_ID_1,/kaggle/input/REAL_CLEAN_ROOT/path/to/ID_1.jpg
```

**(ii) `degraded_index.csv`**:

```csv
image_id,variant_id,path,geometry,homography_json
OpenImages_ID_1,unique_variant_001,/kaggle/input/REAL_DEG_ROOT/path/to/variant.jpg,identity,
```

- `image_id` phải xuất phát từ metadata/mapping **đã xác minh** (không đoán từ tên tệp).
- `variant_id` duy nhất trên toàn bộ dữ liệu.
- `geometry=identity`: chỉ photometric effects (blur, noise, light, JPEG, xử lý không đổi hệ tọa độ/kích thước).
- `geometry=scale`: ảnh output khác size do **resize thuần**, dùng tỉ lệ kích thước từ GT.
- `geometry=homography`: phép biến đổi phối cảnh/affine tổng hợp, cần ma trận `H` 3×3 biến đổi **GT original pixel coordinates → degraded output pixel coordinates**. Nếu sau perspective có resize thì `H` phải là **composite H cuối cùng**, không phải chỉ H ở một bước.
- Không có transform metadata đáng tin thì **không được dùng TextOCR GT gốc để chấm ảnh perspective**; đánh dấu blocked/quarantine cho đến khi cung cấp lại mapping hoặc warp GT chính xác.
- Chú ý EXIF orientation, các tệp bị rotate/resize. Overlay nhiều ảnh clean/degraded + GT trước khi chạy lớn.

**Nếu không có mapping metadata degraded:** Codex cần liệt kê file và đề xuất cách index, **hỏi Tuấn xác nhận**. Chỉ sau đó mới sinh CSV; không tự suy đoán regex filename.

### E3. Tạo manifest bằng code chung (đính kèm trong gói này)

Tập tin: `tools/build_shared_manifest.py`. Code đọc 2 CSV index + TextOCR JSON, kiểm tra file tồn tại, đúng `image_id`, nhãn hình học, GT, tạo **một manifest JSONL** và audit SHA256.

```bash
python tools/build_shared_manifest.py \
  --clean-index clean_index.csv \
  --degraded-index degraded_index.csv \
  --textocr-json /kaggle/input/EXACT_TEXT_OCR_PATH/TextOCR_0.1_train.json \
  --out outputs/manifest/shared_manifest_v1.jsonl

cat outputs/manifest/shared_manifest_v1.jsonl.audit.json
sha256sum outputs/manifest/shared_manifest_v1.jsonl
```

**Quan trọng:** Đường dẫn dataset trong lệnh chỉ là minh họa. Chạy `find /kaggle/input -maxdepth 3 -type f | head` để xác định thật. Không xóa hàng thiếu một cách im lặng: lưu số lượng excluded / reasons / examples trong audit. Kết quả phải được Tuấn và cả hai model owners kiểm tra và **freeze** trước inference. **Cả hai model dùng chính xác cùng bytes manifest (cùng SHA256)**.

### E4. Kiểm tra GT, granularity, leakage

- TextOCR thường chú thích theo **word**, trong khi detector có thể xuất **text-line polygon**. Không ngầm xem word-level IoU 0,5 là official detection metric của paper. Cùng báo cáo **polygon-IoU micro Precision/Recall/Hmean** theo evaluator thống nhất **và** `Word Coverage Recall` dạng diagnostic. Diễn giải giới hạn word-vs-line và thêm visual audit. Một polygon khổng lồ có thể đạt Coverage cao → luôn xem kèm precision, predicted image-area fraction và false positives.
- **Đánh giá Text Detection, không tính CER/WER** vì chưa chạy recognition. `TextOCR.utf8_string` chỉ dùng tham khảo, không dùng VQA answer để đo detection.
- Với GT polygon của textOCR, không giữ nguyên tọa độ khi ảnh bị perspective/crop/rotate/resize mà hệ tọa độ thay đổi.
- Dataset TextOCR train có thể từng được một checkpoint pretrained nhìn thấy; ghi rõ **external test on matched TextVQA–TextOCR subset**, không tuyên bố unseen/test-blind. Phase cải tiến sau này phải tạo train/val/test split theo source_image_id không chồng lắp.

### E5. Protocol đánh giá bắt buộc

**Cùng conditions / giữ checkpoint & config gốc:**

```text
Condition A: TextVQA CLEAN images in frozen intersection S
Condition B: matching TextVQA DEGRADED variants from 10K dataset
GT: polygons TextOCR, transform only when geometry changes
```

Bảng kết quả bắt buộc:

| Model | Condition | #unique source images | #variants | Precision | Recall | Hmean | Word Coverage Recall | latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| PP-OCRv5_server_det | clean | | | | | | | |
| PP-OCRv5_server_det | degraded | | | | | | | |
| DBNet + MonkeyOCRv2-AS | clean | | | | | | | |
| DBNet + MonkeyOCRv2-AS | degraded | | | | | | | |

- **Gap = metric_degraded − metric_clean**, luôn ghi rõ dấu; kèm absolute delta percentage points. Không mặc định gap sẽ âm.
- Nếu nhiều degraded variants / image: báo **clean một lần/unique image**, degraded trên từng variant; có thể báo thêm macro theo unique source image và cluster-bootstrap 95% CI theo `image_id` để tránh trọng số mất cân bằng.
- Phân tích theo type/severity (chỉ khi dataset có nhãn xác thực), kích thước chữ, clean/degraded paired error cases, FP/FN visual overlays.
- Với dữ liệu có corruption perspective, annotate transformed GT và kiểm tra trực quan trước chấm; **nếu không thể dựng GT tương ứng thì loại riêng các variant đó khỏi đánh giá và báo tỷ lệ loại**. Trong cả hai model phải dùng cùng tập cuối.

---

## F. Kaggle T4×2: code chạy CHIA 2 GPU AN TOÀN

Trong gói kèm 3 script:

```text
tools/run_sharded_inference.py  # tạo tasks, chia 2 shard, launch 2 subprocess
 tools/infer_worker.py         # 1 worker/1 GPU, adapter riêng Paddle/Monkey
 tools/evaluate_detection.py  # cùng GT, cùng evaluator cho cả hai model
```

**Chia dữ liệu (data parallel inference) chứ không chia model qua GPU:** GPU0 xử lý shard0, GPU1 xử lý shard1, dùng **cùng checkpoint/config/preprocess**, hợp nhất predictions theo ID và kiểm tra đủ số lượng trước evaluation.

```bash
# Trong Kaggle, Accelerator GPU T4 x2; xác nhận count == 2
nvidia-smi

# Chạy trên môi trường riêng của PADDLE (người nhận PP-OCRv5):
python tools/run_sharded_inference.py \
  --manifest outputs/manifest/shared_manifest_v1.jsonl \
  --model paddle \
  --output-dir outputs/paddle_phase2

# Chạy trên môi trường riêng của MONKEY (Tuấn):
# Tên đường dẫn config/checkpoint phải đúng với cây MMOCR sau install.sh.
python tools/run_sharded_inference.py \
  --manifest outputs/manifest/shared_manifest_v1.jsonl \
  --model monkey \
  --monkey-config /kaggle/working/MonkeyOCRv2/detection/mmocr/mmocr/configs/textdet/dbnet/dbnet_mkv2vitae_1200e_icdar2015_2gpu_adamw.py \
  --monkey-checkpoint /kaggle/working/MonkeyOCRv2/detection/mmocr/mmocr/model_weight/dbnet_mkv2vitae_icdar2015.pth \
  --output-dir outputs/monkey_phase2
```

Khi `run_sharded_inference.py` chạy, nó thực thi hai tiến trình độc lập có môi trường:

```python
# Physical GPU 0 -> worker 0; physical GPU 1 -> worker 1:
env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
# Mỗi tiến trình CHỈ thấy GPU được gán, dùng local device = cuda:0 / gpu:0
# tasks[i::2] phân phối theo sample ID; merged output kiểm tra thiếu/trùng.
```

**Lưu ý thực tiễn:** 2 T4 không tự làm inference nhanh gấp đôi nếu bottleneck là đọc file / giải mã ảnh / khởi tạo framework. Nếu GPU memory không đủ, batch size 1, sequential per worker, kiểm tra smoke test 20–50 samples trước. `torch.cuda.device_count()` trong notebook phải là 2; với Paddle hãy xác nhận framework nhận hai physical GPUs. Inference gốc Phase 1 không nhất thiết phải chạy song song: single-GPU official eval thường là lựa chọn chuẩn đầu tiên.

**Đầu ra chuẩn hóa của mỗi model:**

```json
{"sample_id":"deg::unique_variant_001","image_id":"OpenImages_ID_1","variant_id":"unique_variant_001","condition":"degraded","model":"monkey","polygons":[[10,20,30,20,30,45,10,45]],"scores":[0.91]}
```

`polygons` lưu tọa độ trong hệ tọa độ ảnh input ở độ phân giải thực sau preprocessing inverse; mỗi polygon list `[x1,y1,x2,y2,...]`. `sample_id` là khóa bắt buộc, không lấy đường dẫn làm ID.

**Eval chung:**

```bash
python tools/evaluate_detection.py \
  --manifest outputs/manifest/shared_manifest_v1.jsonl \
  --predictions outputs/paddle_phase2/paddle_all_predictions.jsonl \
  --textocr-json /kaggle/input/EXACT_TEXT_OCR_PATH/TextOCR_0.1_train.json \
  --out outputs/metrics/paddle_textvqa.json

python tools/evaluate_detection.py \
  --manifest outputs/manifest/shared_manifest_v1.jsonl \
  --predictions outputs/monkey_phase2/monkey_all_predictions.jsonl \
  --textocr-json /kaggle/input/EXACT_TEXT_OCR_PATH/TextOCR_0.1_train.json \
  --out outputs/metrics/monkey_textvqa.json
```

Cài các dependency evaluator trong môi trường riêng: `pip install numpy scipy shapely pillow`. **Evaluator cung cấp là shared external word-level protocol, KHÔNG được dùng để tái hiện các con số benchmark chính thức ở Phase 1.**

### F1. Chạy pilot và khóa chất lượng

```text
Pilot 1: 10-20 ảnh clean có overlay GT + boxes, xem orientation và ID
Pilot 2: 10-20 ảnh degraded có blur, low-res, perspective (nếu đủ H)
Pilot 3: 50-100 samples chạy 2 GPU: so sánh sample_id, số predictions, polygons hợp lệ
Pilot 4: 1-GPU vs 2-GPU trên 10 ảnh trùng nhau; kết quả phải gần tương đương
Full run: chỉ sau khi pass toàn bộ pilot + SHA manifest được cả nhóm xác nhận
```

Nếu adapter báo lỗi/không đọc đúng keys, sửa theo API/version đang chạy, lưu lỗi và diff; **không thay checkpoint hoặc ngưỡng detector để "đẹp" kết quả**.

---

## G. Gate checklist và deliverables

### Gate 1 — Phase 1 hoàn thành

- [ ] Monkey: đúng version, checkpoint, config, ICDAR2015 test, P/R/Hmean + log.
- [ ] Monkey: so với paper 91.3 / 86.7 / 88.9; sai khác giải thích được.
- [ ] Paddle: benchmark gốc có được không? Nếu không, ghi `NOT VERIFIED` cho 83.8 và đưa kết quả public benchmark riêng.
- [ ] 2 repo commits/weights hashes/env saved.
- [ ] 10–20 visual overlays cho từng detector.
- [ ] Tuấn xác nhận chuyển Phase 2 (hoặc chấp thuận ngoại lệ Paddle public validation).

### Gate 2 — Manifest dùng chung

- [ ] Trích xuất đúng mapping degraded → image_id từ metadata thực tế.
- [ ] Chỉ lấy `Clean ∩ Degraded ∩ TextOCR`; audit removed + unique counts.
- [ ] Đúng image orientation / GT polygons / geometric mapping.
- [ ] Đã review 20–30 ảnh overlays GT clean/degraded, gồm perspective.
- [ ] `shared_manifest_v1.jsonl` và `SHA256` được cả nhóm dùng cùng.
- [ ] Đã thống nhất difficult annotation, IoU, word-level granularity.

### Gate 3 — External evaluation

- [ ] 4 dòng metrics (2 models × clean/degraded) và số ảnh rõ ràng.
- [ ] 2×T4 runs merged đủ 100%, không trùng và không thiếu `sample_id`.
- [ ] P/R/Hmean, Word Coverage, gap clean-degraded.
- [ ] FN/FP visual gallery, theo type/severity/text size nếu metadata hỗ trợ.
- [ ] Không chỉnh degradation/GT/threshold sau khi thấy kết quả test.
- [ ] Chưa đề xuất một cải tiến "đã hiệu quả" nếu chưa có evidence.

### File cấu trúc nộp

```text
artifacts/
├── phase1_monkey/
│   ├── env_versions.txt
│   ├── git_commit.txt
│   ├── checkpoint_sha256.txt
│   ├── official_eval.log
│   ├── official_metrics.json
│   └── visual_examples/
├── phase1_paddle/
│   ├── env_versions.txt
│   ├── git_commit.txt
│   ├── official_benchmark_availability.md
│   ├── evaluation_type.txt  # official or external; never conflate
│   └── metrics.json
├── shared/
│   ├── clean_index.csv
│   ├── degraded_index.csv
│   ├── shared_manifest_v1.jsonl
│   ├── shared_manifest_v1.jsonl.audit.json
│   └── gt_overlays/
└── phase2/
    ├── monkey_predictions.jsonl
    ├── paddle_predictions.jsonl
    ├── monkey_metrics.json
    ├── paddle_metrics.json
    ├── paired_error_analysis.csv
    └── clean_vs_degraded_report.md
```

---

## H. Chỉ dẫn cho Codex khi bắt đầu

1. **Đọc nguyên tài liệu này** rồi tóm tắt mục tiêu/gates/phân công.
2. **Không tự download/re-generate 10K degraded images.** Dataset đã có sẵn; quan sát cấu trúc Kaggle trước.
3. Đầu tiên hỏi những dữ liệu chưa biết: đường dẫn thật, `source_image_id` mapping, saved homographies, original GT coordinate frame, paper benchmark access, và ai trong Trí/Triệu nhận Paddle.
4. **Ưu tiên Phase 1**: báo reproduction success/failure/blockers với bằng chứng, không chuyển thẳng sang TextVQA.
5. Khi Phase 1 đã được Tuấn xác nhận, triển khai **một shared manifest/evaluator**; không cho mỗi model một pipeline khác nhau.
6. Giữ artifact có thể audit (checkpoint/config/hashes, source commits, environment, predictions, metrics, logs, failures).
7. Chưa làm Phase 3 restoration/feature adapter cho tới khi có gap và Tuấn chốt phương án.

## I. Ghi chú về dữ liệu tạo suy thoái (đã có sẵn)

Hướng thiết kế từ handoff cũ: **R-Bench-inspired, BSRGAN/Real-ESRGAN-style**. Bao gồm motion camera shake, defocus blur, low-light/sensor noise, overexposure/glare, low resolution, perspective, JPEG compression và realistic mix có thứ tự gần với camera. Có severity theo kích thước chữ và nguyên tắc đánh giá answerability. **Không giả định rằng file Kaggle hiện tại mang đủ 7 loại/đủ metadata hay đúng manifest cũ**: phải inspect và báo cáo bằng chứng. Đặc biệt, phép perspective là **geometric degradation** nên GT cần transform; các phép blur/noise/compression giữ coordinates khi ảnh không đổi kích thước.

## J. Tài liệu tham khảo kiểm chứng

- MonkeyOCRv2 repo + detection directory: https://github.com/Yuliang-Liu/MonkeyOCRv2
- MonkeyOCRv2 detection results + install/train/test: https://huggingface.co/HB16888/MonkeyOCRv2_det
- PaddleOCR 3.0 detector official docs: https://www.paddleocr.ai/main/en/version3.x/module_usage/text_detection.html
- PaddleOCR 3.0 tech report: https://arxiv.org/abs/2507.05595
- MMOCR inferencer custom model: https://mmocr.readthedocs.io/en/latest/user_guides/inference.html
- TextOCR annotations + alignment to TextVQA: https://textvqa.org/textocr/dataset/


---

## K. Mã nguồn đầy đủ — dùng khi chỉ chuyển file .md cho Codex

Bốn script dưới đây cũng có file `.py` riêng trong gói. Nếu chỉ copy riêng file `.md`, Codex có thể tạo lại từng file theo code block và sau đó điều chỉnh đường dẫn thực tế. Đây là code khởi đầu đã qua kiểm tra cú pháp Python, **chưa được chạy trực tiếp trên dữ liệu Kaggle hoặc trên các model thực tế**.

### K — `tools/build_shared_manifest.py`

```python
"""Build one locked, paired TextVQA/TextOCR manifest for BOTH detectors.
Requires pre-audited CSVs: clean(image_id,path) and
 degraded(image_id,variant_id,path,geometry,homography_json).
No filename heuristics or silently inferred geometry.
"""
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path
from PIL import Image


def rows_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clean-index', required=True)
    p.add_argument('--degraded-index', required=True)
    p.add_argument('--textocr-json', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    clean_rows = rows_csv(args.clean_index)
    deg_rows = rows_csv(args.degraded_index)
    with open(args.textocr_json, encoding='utf-8') as f:
        gt = json.load(f)
    assert isinstance(gt.get('imgs'), dict) and isinstance(gt.get('img2Anns'), dict), 'Unexpected TextOCR JSON schema'
    clean = {}
    for r in clean_rows:
        image_id = r['image_id'].strip()
        path = r['path'].strip()
        if not image_id or image_id in clean: raise ValueError(f'Missing/duplicate clean ID: {image_id}')
        if not Path(path).is_file(): raise FileNotFoundError(path)
        clean[image_id] = path
    seen_variants, records = set(), []
    stats = Counter()
    for r in deg_rows:
        image_id = r['image_id'].strip()
        variant_id = r['variant_id'].strip()
        deg_path = r['path'].strip()
        geometry = r.get('geometry','').strip()
        h_raw = r.get('homography_json','').strip()
        if not image_id or not variant_id or variant_id in seen_variants: raise ValueError(f'Bad/duplicate variant_id: {variant_id}')
        seen_variants.add(variant_id)
        if image_id not in clean: stats['not_in_clean'] += 1; continue
        if image_id not in gt['imgs']: stats['not_in_textocr'] += 1; continue
        if not gt['img2Anns'].get(image_id): stats['no_text_annotations'] += 1; continue
        if not Path(deg_path).is_file(): raise FileNotFoundError(deg_path)
        if geometry not in {'identity', 'scale', 'homography'}:
            raise ValueError(f'{variant_id}: geometry must be identity/scale/homography; no guessing')
        if geometry == 'homography':
            try: H = json.loads(h_raw)
            except Exception as exc: raise ValueError(f'{variant_id}: invalid homography_json') from exc
            if not (isinstance(H, list) and len(H)==3 and all(isinstance(z,list) and len(z)==3 for z in H)):
                raise ValueError(f'{variant_id}: H must be a 3x3 matrix, ORIGINAL-GT pixels -> degraded pixels')
        else:
            H = None
        clean_path = clean[image_id]
        orig_meta = gt['imgs'][image_id]
        original_size = (int(orig_meta['width']), int(orig_meta['height']))
        with Image.open(clean_path) as img: clean_size = img.size
        with Image.open(deg_path) as img: degraded_size = img.size
        if clean_size != original_size:
            raise ValueError(f'{image_id}: clean size {clean_size} != TextOCR GT size {original_size}; investigate EXIF/resizing')
        if geometry == 'identity' and degraded_size != original_size:
            raise ValueError(f'{variant_id}: identity cannot change dimensions {degraded_size} vs GT {original_size}')
        records.append({
            'image_id': image_id, 'variant_id': variant_id,
            'clean_path': clean_path, 'degraded_path': deg_path,
            'geometry': geometry, 'homography': H,
            'gt_size': list(original_size), 'degraded_size': list(degraded_size),
            'gt_annotation_ids': gt['img2Anns'][image_id],
        })
    if not records: raise RuntimeError('Empty intersection. STOP and inspect dataset metadata.')
    records.sort(key=lambda r: (r['image_id'], r['variant_id']))
    output = Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False, sort_keys=True)+'\n')
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    summary = {
        'total_clean_index': len(clean), 'total_degraded_index': len(deg_rows),
        'intersection_variants': len(records),
        'intersection_unique_images': len({r['image_id'] for r in records}),
        'skips': dict(stats), 'sha256': digest,
        'WARNING': 'Review polygon overlay for clean AND geometric degraded images before evaluation.'
    }
    (output.parent / (output.name+'.audit.json')).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__': main()

```

### K — `tools/run_sharded_inference.py`

```python
"""Launch TWO independent inference workers, one visible T4 per process.
Each worker runs THE SAME fixed detector/checkpoint/config on different items.
Works in Kaggle notebook via subprocess; model frameworks remain separate.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path


def load_tasks(manifest):
    with open(manifest, encoding='utf-8') as f:
        rows=[json.loads(s) for s in f if s.strip()]
    unique_clean = {r['image_id']: r['clean_path'] for r in rows}
    tasks = [{'sample_id':f'clean::{k}', 'image_id':k, 'condition':'clean', 'path':v}
             for k,v in sorted(unique_clean.items())]
    tasks += [{'sample_id':f'deg::{r["variant_id"]}', 'image_id':r['image_id'],
               'variant_id':r['variant_id'], 'condition':'degraded', 'path':r['degraded_path']}
              for r in rows]
    ids=[t['sample_id'] for t in tasks]
    if len(ids) != len(set(ids)): raise ValueError('Duplicate sample_id')
    return tasks


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--model', required=True, choices=['paddle', 'monkey'])
    p.add_argument('--output-dir', required=True)
    p.add_argument('--monkey-config', default='')
    p.add_argument('--monkey-checkpoint', default='')
    args=p.parse_args()
    tasks=load_tasks(args.manifest)
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    procs=[]
    for i in (0,1):
        subset=tasks[i::2]
        shard=out/f'tasks_gpu{i}.jsonl'
        with shard.open('w',encoding='utf-8') as f:
            for task in subset: f.write(json.dumps(task,ensure_ascii=False)+'\n')
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(i)
        cmd=[sys.executable,str(Path(__file__).parent/'infer_worker.py'), '--tasks',str(shard),
             '--model',args.model, '--output',str(out/f'pred_gpu{i}.jsonl')]
        if args.model == 'monkey':
            cmd += ['--config',args.monkey_config,'--checkpoint',args.monkey_checkpoint]
        print(f'Start GPU {i}: {len(subset)} samples',flush=True)
        procs.append(subprocess.Popen(cmd,env=env,stdout=(out/f'gpu{i}.log').open('w'),stderr=subprocess.STDOUT))
    failures=[(i,pr.wait()) for i,pr in enumerate(procs) if pr.wait()!=0]
    if failures: raise RuntimeError(f'Workers failed {failures}; see logs. Do not score partial outputs.')
    got={}
    for i in (0,1):
        for line in (out/f'pred_gpu{i}.jsonl').read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            r=json.loads(line); key=r['sample_id']
            if key in got: raise RuntimeError(f'Duplicate prediction: {key}')
            got[key]=r
    expected={x['sample_id'] for x in tasks}
    if set(got)!=expected: raise RuntimeError(f'Missing/unexpected predictions: missing={len(expected-set(got))}, extra={len(set(got)-expected)}')
    merged=out/f'{args.model}_all_predictions.jsonl'
    with merged.open('w',encoding='utf-8') as f:
        for k in sorted(got): f.write(json.dumps(got[k],ensure_ascii=False)+'\n')
    print(f'OK: {len(got)} predictions in {merged}')

if __name__=='__main__': main()

```

### K — `tools/infer_worker.py`

```python
"""Single-GPU worker: launches under CUDA_VISIBLE_DEVICES=physical_gpu_id.
The process sees ONE GPU at local ordinal 0.
"""
import argparse,json
from pathlib import Path


def model_init(a):
    if a.model=='paddle':
        from paddleocr import TextDetection
        det=TextDetection(model_name='PP-OCRv5_server_det',device='gpu:0')
        def predict(path):
            output=list(det.predict(input=path,batch_size=1))
            assert len(output)==1
            raw=output[0].json
            if 'res' in raw: raw=raw['res']
            polys=raw['dt_polys']; scores=raw['dt_scores']
            return [[float(t) for point in poly for t in point] for poly in polys], [float(s) for s in scores]
        return predict
    if not a.config or not a.checkpoint:
        raise ValueError('Monkey requires exact official --config and --checkpoint')
    from mmocr.apis import TextDetInferencer
    det=TextDetInferencer(model=a.config,weights=a.checkpoint,device='cuda:0')
    def predict(path):
        result=det(path,return_vis=False,save_pred=False)
        pred=result['predictions'][0]
        return [[float(x) for x in poly] for poly in pred['polygons']], [float(s) for s in pred['scores']]
    return predict


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tasks',required=True);ap.add_argument('--model',choices=['paddle','monkey'],required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--config',default='');ap.add_argument('--checkpoint',default='')
    a=ap.parse_args()
    pred=model_init(a)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with open(a.tasks,encoding='utf-8') as f, out.open('w',encoding='utf-8') as o:
        for line in f:
            if not line.strip():continue
            task=json.loads(line)
            if not Path(task['path']).is_file():raise FileNotFoundError(task['path'])
            polygons,scores=pred(task['path'])
            if len(polygons)!=len(scores):raise ValueError(f'Polygon/score mismatch {task["sample_id"]}')
            r={k:task[k] for k in task if k!='path'}
            r.update(model=a.model,polygons=polygons,scores=scores)
            o.write(json.dumps(r,ensure_ascii=False)+'\n')
            o.flush()
if __name__=='__main__':main()

```

### K — `tools/evaluate_detection.py`

```python
"""A shared *external TextOCR word-level* evaluator, NOT paper-official Hmean.
Compute polygon IoU one-to-one micro P/R/F1, and coverage recall for text words.
Identical evaluator for both models/conditions. Annotation level may differ from detectors.
"""
import argparse, json, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from scipy.optimize import linear_sum_assignment


def read_jsonl(file):
    with open(file,encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def poly(raw):
    vals=np.asarray(raw,dtype=float)
    if vals.size<8 or vals.size%2: return None
    p=Polygon(vals.reshape(-1,2))
    if not p.is_valid: p=p.buffer(0)
    return p if p.is_valid and p.area>0 else None


def gt_polys(row, gt, degraded):
    ids=row['gt_annotation_ids']
    raw=[gt['anns'][x]['points'] for x in ids]
    if not degraded: return [z for x in raw if (z:=poly(x)) is not None]
    geometry=row['geometry']; W,H=row['gt_size']; w,h=row['degraded_size']
    result=[]
    for x in raw:
        arr=np.array(x,dtype=float).reshape(-1,2)
        if geometry=='scale':
            arr[:,0]*=w/W;arr[:,1]*=h/H
        elif geometry=='homography':
            mat=np.array(row['homography'],dtype=float)
            xy1=np.column_stack([arr,np.ones(len(arr))])@mat.T
            if np.any(np.abs(xy1[:,2])<1e-12):raise ValueError('Degenerate H')
            arr=xy1[:,:2]/xy1[:,2,None]
        elif geometry!='identity': raise ValueError('Unknown geometry')
        if (z:=poly(arr.reshape(-1))) is not None:result.append(z)
    return result


def match(gt, pr, iou_thr):
    if not gt or not pr: return 0
    ious=np.zeros((len(gt),len(pr)))
    for i,g in enumerate(gt):
        for j,p in enumerate(pr):
            inter=g.intersection(p).area
            if inter: ious[i,j]=inter / g.union(p).area
    # maximize number of matches above threshold (then IoU as tie-break)
    cost=np.where(ious>=iou_thr, -1 - ious/1000, 0)
    ri,ci=linear_sum_assignment(cost)
    return int(sum(bool(ious[i,j]>=iou_thr) for i,j in zip(ri,ci)))


def word_coverage(gt,pr,thresh):
    # Diagnostic only; giant boxes can game it. Always report with precision.
    if not gt or not pr:return 0
    union=unary_union(pr)
    return int(sum(bool(union.intersection(g).area / g.area >= thresh) for g in gt))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--predictions',required=True)
    ap.add_argument('--textocr-json',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--iou',type=float,default=.5)
    ap.add_argument('--coverage',type=float,default=.5)
    a=ap.parse_args()
    rows=read_jsonl(a.manifest)
    preds=read_jsonl(a.predictions)
    pred_map={p['sample_id']:p for p in preds}
    if len(pred_map)!=len(preds):raise ValueError('Duplicate prediction IDs')
    with open(a.textocr_json,encoding='utf-8') as f:gt=json.load(f)
    clean_by_id={}
    for row in rows:clean_by_id.setdefault(row['image_id'],row)
    tasks=[(f'clean::{k}',r,False) for k,r in clean_by_id.items()]
    tasks += [(f'deg::{r["variant_id"]}',r,True) for r in rows]
    expected={task[0] for task in tasks}
    if expected!=set(pred_map): raise ValueError(f'Missing/extraneous prediction IDs: {len(expected-set(pred_map))}/{len(set(pred_map)-expected)}')
    totals=defaultdict(lambda: dict(tp=0,gt=0,pred=0,covered=0,images=0))
    per_image=[]
    for key,row,degraded in tasks:
        gt_ps=gt_polys(row,gt,degraded)
        pr_ps=[z for x in pred_map[key]['polygons'] if (z:=poly(x)) is not None]
        matched=match(gt_ps,pr_ps,a.iou)
        covered=word_coverage(gt_ps,pr_ps,a.coverage)
        label='degraded' if degraded else 'clean'
        s=totals[label]
        s['tp']+=matched;s['gt']+=len(gt_ps);s['pred']+=len(pr_ps);s['covered']+=covered;s['images']+=1
        per_image.append(dict(sample_id=key,image_id=row['image_id'],condition=label,
                              tp=int(matched),gt=len(gt_ps),pred=len(pr_ps),covered=int(covered)))
    summary={}
    for label,s in totals.items():
        P=s['tp']/s['pred'] if s['pred'] else 0
        R=s['tp']/s['gt'] if s['gt'] else 0
        F=2*P*R/(P+R) if P+R else 0
        summary[label]={**s,'precision':P,'recall':R,'hmean':F,
                        'word_coverage_recall':s['covered']/s['gt'] if s['gt'] else 0}
    output=Path(a.out);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (output.parent / (output.stem+'_per_image.jsonl')).open('w',encoding='utf-8') as f:
        for item in per_image:f.write(json.dumps(item)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()

```
