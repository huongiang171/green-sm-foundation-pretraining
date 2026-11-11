# VRIC — Rail-hailing Foundation Model

Dự án nghiên cứu **User Foundation Model**: học biểu diễn người dùng từ hồ sơ và lịch sử hành vi, sau đó tái sử dụng cho **Food Recommendation** (gợi ý món ăn) và **Ads Optimization** (xếp hạng quảng cáo theo xác suất click).

Pipeline Green SM hiện có gồm sinh dữ liệu mô phỏng, kiểm định dữ liệu, chuẩn hóa đầu vào, pretraining Transformer, huấn luyện downstream với Foundation đóng băng và đánh giá bằng evaluator chung. Repository cũng lưu dữ liệu, metadata và công cụ xử lý của các hướng OmniBehavior, Meituan và Taobao.

> Green SM trong repository là dữ liệu **mô phỏng**. Kết quả trên corpus này phục vụ kiểm chứng pipeline và giả thuyết nghiên cứu; không phải bằng chứng hiệu quả trên dữ liệu vận hành thực tế.

## Mục lục

- [Phạm vi triển khai](#phạm-vi-triển-khai)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
- [Dữ liệu và hợp đồng đầu vào](#dữ-liệu-và-hợp-đồng-đầu-vào)
- [Chạy pipeline Green SM](#chạy-pipeline-green-sm)
- [Chạy trên Kaggle](#chạy-trên-kaggle)
- [Đánh giá và so sánh](#đánh-giá-và-so-sánh)
- [Kiểm tra và đóng góp](#kiểm-tra-và-đóng-góp)
- [Tài liệu trong repository](#tài-liệu-trong-repository)

## Phạm vi triển khai

Các thành phần đã có mã triển khai và test:

- **Green SM simulator:** sinh hai corpus `linked` và `control_paired`, cùng Food/Ads opportunities, manifest, checksum và báo cáo chất lượng/nhân quả.
- **Foundation input:** đọc Parquet theo manifest, kiểm tra cutoff/split, tokenize theo trường dữ liệu, fit vocabulary và chuẩn hóa số chỉ trên train, collate thành batch có numeric và geo tensors.
- **Foundation model:** adapter theo modality, `UserIdAdapter`, `GeoAdapter`, Transformer encoder và pooling. `FoundationModel.embed_users()` trả về `long_term` và `short_term`; frozen embedder của pretraining bổ sung biểu diễn hợp nhất `z_user`.
- **Pretraining:** masked reconstruction cho categorical/numeric/geo và next-action prediction; validation, checkpoint selection, lưu vocabulary, loss trace và resume state.
- **Downstream:** Food ranker với listwise loss và Ads CTR ranker với binary cross-entropy, dùng Foundation đóng băng và xuất prediction không chứa nhãn.
- **Evaluation:** xác minh artifact, tính metric Food/Ads và so sánh các report tương thích, chạy được trên CPU.

Một số phần vẫn là khung: `src/baselines/baseline.py`, `src/downstream/ranking.py`, `src/data/dataset.py` và `src/data/preprocess.py`. Chế độ `demo` của Food chưa được triển khai. Các design/plan trong `docs/superpowers/` là tài liệu thiết kế, không tự xác nhận tính năng đã hoàn tất hay có kết quả thực nghiệm.

## Cấu trúc thư mục

```text
configs/                          # JSON simulator/pretraining, YAML downstream
src/
  data/green_sm/                  # Sinh corpus mô phỏng và kiểm định readiness
  foundation_input/               # Manifest, loader, tokenizer, vocabulary, batch
  models/                         # Adapters, Transformer, pooling, objective heads
  pretraining/                    # CLI, config, masking, gates, runner, artifacts
  training/                       # Public loader cho Foundation checkpoint
  downstream/                     # Food Recommendation và Ads Optimization
  contracts/                      # OutputContractV1 cho prediction artifacts
  evaluation/                     # CLI evaluate/compare và metric policy
  baselines/                      # Khung baseline
notebooks/                        # Notebook pretrain/Food/Ads/evaluation trên Kaggle
data/
  pretrain_data/                  # OmniBehavior và Meituan theo storage schema v2
  meituan_data/                   # Dữ liệu/tài liệu Meituan gốc
  meituan_analysis/               # Script tiền xử lý, kiểm định, báo cáo Meituan
  downstream1_meituan_data/        # Dữ liệu downstream Meituan và candidate POIs
  processed/                     # Output Green SM cục bộ, được gitignore
  README*.md                     # Hướng dẫn nguồn dữ liệu và định dạng
  requirements.txt               # Dependency bổ sung cho phân tích dữ liệu
docs/                             # Contracts, runbook, protocol và thiết kế
tests/                            # Test data/input/model/pretraining/downstream/evaluation
pyproject.toml                    # Package metadata, dependency, pytest và Ruff
CONTRIBUTING.md                   # Quy ước branch, commit và Pull Request
```

Mã nguồn nằm trực tiếp trong các package dưới `src/`; không có thư mục `src/rail_hailing/`. Chạy các lệnh `python -m src...` bên dưới từ **thư mục gốc repository** để import nhất quán.

## Cài đặt

Yêu cầu **Python 3.10 trở lên**. Kiểm tra và evaluator chạy được trên CPU; các recipe huấn luyện Kaggle dùng GPU.

```bash
python -m venv .venv
```

Kích hoạt môi trường:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

Cài package và công cụ phát triển:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

`pyproject.toml` khai báo `pyarrow>=17.0.0` và `scikit-learn==1.7.2`; extra `dev` gồm pytest và Ruff. **PyTorch và PyYAML chưa được khai báo ở đây**, nhưng code model/pretraining/downstream cần chúng:

```bash
python -m pip install torch PyYAML
```

Nếu dùng GPU, chọn bản PyTorch tương thích CUDA của môi trường. Với các script phân tích Meituan/Taobao, cài thêm:

```bash
python -m pip install -r data/requirements.txt
```

Kiểm tra các entry point mà không chạy huấn luyện:

```bash
python -m src.data.green_sm --help
python -m src.pretraining --help
python -m src.downstream.food_recommendation --help
python -m src.downstream.ad_optimization --help
python -m src.evaluation --help
```

## Dữ liệu và hợp đồng đầu vào

### Các luồng dữ liệu

- **Green SM:** corpus mô phỏng có lịch sử Foundation và opportunities Food/Ads gắn với cùng target world. Đây là luồng của các CLI huấn luyện và đánh giá bên dưới.
- **OmniBehavior:** nguồn pretraining được mô tả trong [data/README_pretrain_data.md](data/README_pretrain_data.md), với metadata và script đọc ở `data/pretrain_data/OmniBehavior_data/`.
- **Meituan:** dữ liệu pretraining chuyển sang cùng storage schema `omnibehavior_pretraining_v2`, bộ downstream riêng và chuỗi script health check → sampling → build → manifest → verification trong `data/meituan_analysis/`.
- **Taobao:** tài liệu nguồn Ads ở [data/README_taobao.md](data/README_taobao.md). Recipe Ads Green SM hiện tại dùng native Green SM opportunities; không ghép Taobao vào recipe này.

Các script Meituan còn cấu hình đường dẫn tuyệt đối `d:/Meituan` trong [common/config.py](data/meituan_analysis/common/config.py); cần điều chỉnh trước khi tái tạo dữ liệu trên máy khác. Input adapter dùng một dataset root và một logical split mỗi lần, không tự nối user giữa các nguồn. CLI `src.pretraining` là runner chuyên cho Green SM, không phải CLI pretraining tổng quát cho mọi dataset root.

### Nguyên tắc chống rò rỉ dữ liệu

- Snapshot chỉ sử dụng lịch sử hợp lệ tại cutoff; split được kiểm tra theo thời gian.
- Vocabulary, numeric normalization và feature artifacts được fit trên **train**; validation/test chỉ transform. Giá trị ngoài vocabulary đi vào `UNK`.
- Tokenization giữ nhóm sự kiện theo `source_record_id`; truncation giữ nhóm hoàn chỉnh gần nhất. `max_history_events` và `max_tokens` là hai giới hạn khác nhau.
- Next-action target được tách khỏi input; chỉ tính loss trên những dòng có `next_action_valid_mask` hợp lệ.
- Numeric missing có mask riêng; giá trị `0` quan sát được không bị đồng nhất với missing. Geo cũng có value/missing tensors riêng.
- Manifest/checksum gắn dữ liệu, vocabulary, checkpoint và prediction với đúng phiên bản, corpus và lần chạy.

Storage schema v2 và geo batch contract v3 là hai lớp khác nhau. Green SM preflight yêu cầu geo contract v3 và `readiness_passed: true`; code hiện đã có `GeoAdapter` cùng các trường geo trong batch. Một số runbook cũ vẫn mô tả thời điểm geo còn là blocker, nên cần đối chiếu code/config hiện tại khi chạy.

## Chạy pipeline Green SM

```text
Simulator → linked / control_paired + Food/Ads opportunities
          → readiness reports + manifests/checksums
          → train-only vocabulary → Foundation pretraining
          → selected linked checkpoint + frozen embedder
          → Food / Ads ranker → label-free predictions
          → shared evaluator → compatible report comparison
```

`linked` giữ liên hệ hành vi với downstream targets; `control_paired` là đối chứng. Hai corpus dùng cùng request, candidate và label để phục vụ ablation. Mỗi corpus pretraining có vocabulary riêng; downstream recipes dùng checkpoint **linked** tương ứng.

### 1. Sinh corpus và kiểm tra readiness

Bắt đầu với pilot v2:

```bash
python -m src.data.green_sm --config configs/green_sm_sim_pilot_v2.json
```

Config pilot v2 định nghĩa 240 user, 42 ngày, split 28/7/7 ngày; 2.880 Foundation snapshots, 480 Food requests và 640 Ads requests, tính tổng train/validation/test.

Output nằm dưới `data/processed/green_sm_sim_pilot_v2/`:

```text
green_sm_sim_linked_v2/             # Corpus liên kết
green_sm_sim_control_paired_v2/     # Corpus đối chứng
```

Mỗi corpus chứa `foundation/`, `food_opportunities/`, `ads_opportunities/` và `reports/`. Kiểm tra `reports/data_quality_report.json`, `reports/causal_audit_report.json` và manifest trước khi huấn luyện. Builder từ chối ghi đè output root đã có nội dung; khi đổi dữ liệu, dùng phiên bản/root mới.

Recipe full nằm ở [configs/green_sm_sim_full.json](configs/green_sm_sim_full.json): 12.000 user, 144.000 Foundation snapshots, 24.000 Food requests và 32.000 Ads requests. Chỉ chuyển sang full sau khi xem xét readiness và kết quả pilot:

```bash
python -m src.data.green_sm --config configs/green_sm_sim_full.json
```

### 2. Pretraining Foundation

```bash
python -m src.pretraining --config configs/pretrain_green_sm_pilot_v2.json --device cpu
```

Mặc định CLI train cả `linked` và `control`. Với GPU, thay `--device cpu` bằng `--device cuda`. Có thể giới hạn bước cho smoke run hoặc chỉ chọn một corpus:

```bash
python -m src.pretraining --config configs/pretrain_green_sm_pilot_v2.json --device cpu --max-steps 10 --corpus linked --output-dir artifacts/green_sm_smoke
```

Dù chọn `--corpus linked`, preflight vẫn kiểm tra **cả hai root** linked/control và tính tương thích của chúng. Smoke run ngắn chỉ kiểm tra đường chạy, không đủ để kết luận chất lượng biểu diễn.

Pilot recipe dùng `d_model=128`, 4 Transformer layers, 4 attention heads, tối đa 256 tokens, batch size 32 và 1.000 bước mỗi corpus. Full recipe dùng batch size 64 và 20.000 bước:

```bash
python -m src.pretraining --config configs/pretrain_green_sm_full.json --device cuda --corpus linked
```

Artifact được tổ chức theo output directory, run ID và corpus, gồm:

- `vocabulary.json`, `loss_trace.json` và `run_manifest.json`;
- `last.pt`, `last_manifest.json`;
- `best.pt`, `best_manifest.json` khi có validation checkpoint được chọn;
- `checkpoint_manifest.json` mô tả checkpoint được chọn.

Checkpoint lưu model, fusion, objective heads, optimizer, bước chạy và RNG state. `--resume` yêu cầu chọn đúng một corpus và dùng checkpoint trong đúng run directory với config/identity tương thích. Không dùng lại run directory cho một experiment mới.

`configs/pretrain.yaml` và `configs/downstream.yaml` là cấu hình khung; CLI Green SM pretraining yêu cầu recipe JSON `pretrain_green_sm_*.json`.

### 3. Huấn luyện Food và Ads

Trước khi chạy, tạo bản config phù hợp môi trường từ:

- Food full: [configs/food_recommendation_full.yaml](configs/food_recommendation_full.yaml).
- Food pilot v2: [configs/food_recommendation.yaml](configs/food_recommendation.yaml).
- Ads full: [configs/ad_optimization.yaml](configs/ad_optimization.yaml).

Các YAML hiện trỏ vào mount Kaggle và chứa `replace-with-...`. Cần thay đường dẫn dataset/checkpoint/output, điền checksum dữ liệu và checkpoint thật, kiểm tra `data_version`/corpus khớp nhau, rồi chọn `training.device` phù hợp. Ads dùng selected Foundation checkpoint package qua `checkpoint.package_root`; Food chỉ định checkpoint, manifest và vocabulary riêng.

Validate và train Food, sau khi đã sửa config:

```bash
python -m src.downstream.food_recommendation --config configs/food_recommendation_full.yaml --mode validate
python -m src.downstream.food_recommendation --config configs/food_recommendation_full.yaml --mode train
```

Validate và train Ads:

```bash
python -m src.downstream.ad_optimization --config configs/ad_optimization.yaml --mode validate
python -m src.downstream.ad_optimization --config configs/ad_optimization.yaml --mode train
```

Sau training, điền `ranker.run_dir` bằng thư mục run vừa tạo và `ranker.expected_run_manifest_sha256` bằng SHA-256 của `run_manifest.json` trong thư mục đó. Sau đó xuất prediction:

```bash
python -m src.downstream.food_recommendation --config configs/food_recommendation_full.yaml --mode predict --split test
python -m src.downstream.ad_optimization --config configs/ad_optimization.yaml --mode predict --split test
```

Dùng `--split validation` để xuất tập validation. Run downstream lưu ranker checkpoints, feature artifacts, training metrics và run manifest; prediction artifacts được xuất theo `OutputContractV1`.

## Chạy trên Kaggle

- [pretrain.ipynb](notebooks/pretrain.ipynb): pretraining **linked full v1**, kiểm tra cả paired-control root, chọn và đóng gói Foundation checkpoint.
- [food_recommendation_kaggle.ipynb](notebooks/food_recommendation_kaggle.ipynb): huấn luyện Food ranker trên linked full v1, dùng selected full-v1 checkpoint và xuất prediction validation/test. Cần GPU và Internet theo hướng dẫn notebook.
- [ad_optimization_kaggle.ipynb](notebooks/ad_optimization_kaggle.ipynb): dùng corpus `green-sm-sim-full` cùng package `green-sm-full-v1-checkpoint`, train Ads ranker và xuất prediction.
- [downstream_evaluation_kaggle.ipynb](notebooks/downstream_evaluation_kaggle.ipynb): đánh giá prediction artifacts đã hoàn tất bằng CPU, dùng corpus linked tương ứng.

Kiểm tra dataset mount, nguồn code/branch, checkpoint package và output ở các cell cấu hình trước khi Run All. Notebook pretraining hiện clone branch `feat/foundation-encoder`; không mặc định coi checkout đó giống branch đang mở trên máy. Dùng run ID mới cho experiment mới và lưu commit SHA cùng checksum khi chia sẻ artifact.

## Đánh giá và so sánh

Evaluator đọc nhãn từ immutable opportunity dataset, không đọc nhãn từ prediction. Các đường dẫn `path/to/...` dưới đây là placeholder cần thay bằng artifact thực tế:

```bash
python -m src.evaluation evaluate --task food_recommendation --split test --prediction-dir path/to/food/predictions/test --opportunities-root path/to/linked/food_opportunities --output artifacts/food-report.json
python -m src.evaluation evaluate --task ad_optimization --split test --prediction-dir path/to/ads/predictions/test --opportunities-root path/to/linked/ads_opportunities --output artifacts/ads-report.json
```

Metric hiện được triển khai:

- **Food:** NDCG@5/@10, Recall@3/@10, Hit Rate@1 và MRR. Request không có positive được loại khỏi metric ranking và ghi nhận trong report.
- **Ads:** ROC-AUC, PR-AUC (average precision), log loss, Brier score, NDCG@3 và Recall@3. Request không có click vẫn được tính cho metric pointwise; metric ranking chỉ dùng request có positive. AUC trả về `null` cùng lý do khi chỉ có một lớp nhãn.

Food dùng score xếp hạng; Ads yêu cầu xác suất trong `[0, 1]`. Khi đồng điểm, evaluator sắp theo `candidate_id` tăng dần.

So sánh baseline và Foundation trên **cùng task, split, dữ liệu và metric policy**:

```bash
python -m src.evaluation compare --report artifacts/baseline-food-report.json --report artifacts/foundation-food-report.json --output artifacts/food-comparison.json
```

Không dùng compare để so Food với Ads hoặc trộn report từ các corpus khác nhau. Baseline bên ngoài phải xuất `predictions.<split>.parquet` không chứa nhãn, kèm `evaluation_input_v1.json` và provenance baseline; xem [Evaluation Input V1](docs/evaluation_input_v1.md). Foundation predictions dùng `OutputContractV1` trong [src/contracts/output.py](src/contracts/output.py).

## Kiểm tra và đóng góp

```bash
python -m pytest
python -m ruff check .
```

Có thể kiểm tra từng nhóm trong lúc phát triển:

```bash
python -m pytest tests/foundation_input tests/models tests/pretraining
python -m pytest tests/data tests/downstream tests/contracts tests/evaluation
```

Tests bao phủ contract, temporal validation, tokenization, masking/objective, checkpoint artifacts, downstream và evaluation. Chạy test không thay thế việc review readiness report hoặc chạy experiment thực tế trên dữ liệu đầy đủ.

Quy ước dự án là không thêm dataset lớn, checkpoint, secret và output experiment vào Git. Checkout hiện đã có nhiều Parquet trong các thư mục dữ liệu; không nên giả định `data/` chỉ chứa metadata. `.gitignore` chặn output mới dưới `data/processed/` và các định dạng checkpoint phổ biến; khi tạo thư mục artifact khác, kiểm tra `git status` trước khi commit.

Ghi dataset/preprocessing version, split, seed, candidate generation, negative sampling, downstream setup và metrics cho mỗi experiment theo [experiment protocol](docs/experiment_protocol.md). Tạo branch theo task và Pull Request theo [CONTRIBUTING.md](CONTRIBUTING.md).

## Tài liệu trong repository

- [Foundation input contract](docs/foundation_input_contract.md): canonical samples, tokenization, vocabulary và batch.
- [Data schema](docs/data_schema.md) và [quy ước dữ liệu](data/README.md).
- [OmniBehavior pretraining data](data/README_pretrain_data.md), [Meituan](data/README_Meituan.md) và [Taobao](data/README_taobao.md).
- [Green SM pilot runbook](docs/green_sm_pilot_runbook.md): quy trình readiness và archive; ví dụ dùng pilot v1.
- [Kaggle Foundation runbook](docs/kaggle_foundation_pretraining_runbook.md) và [Kaggle Ads runbook](docs/kaggle_ad_optimization_runbook.md).
- [Baseline evaluation contract](docs/evaluation_input_v1.md).
- [Design specs](docs/superpowers/specs/) và [implementation plans](docs/superpowers/plans/): lịch sử thiết kế và kế hoạch triển khai.
