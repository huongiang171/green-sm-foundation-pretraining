# VRIC — Rail-hailing Foundation Model

Repository chung cho dữ liệu, foundation model, baseline và các bài toán downstream
trong hệ sinh thái rail-hailing.

## Cấu trúc hiện tại

```text
configs/                  # Cấu hình train/evaluation
data/                     # Chỉ lưu hướng dẫn và metadata, không commit dataset lớn
docs/                     # Data schema và experiment protocol
src/rail_hailing/         # Source code dùng chung
tests/                    # Test tự động
```

## Cài đặt môi trường

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

## Kiểm tra nhanh

```bash
python -m pytest
```

## Quy tắc dữ liệu

Dataset, checkpoint, model weight và experiment artifact lớn không được commit vào
Git. Xem [data/README.md](data/README.md) và [docs/data_schema.md](docs/data_schema.md).

## Làm việc nhóm

Xem [CONTRIBUTING.md](CONTRIBUTING.md) trước khi tạo branch hoặc Pull Request.

## Downstream evaluation

Evaluate completed label-free Food or Ads predictions on the linked Green SM corpus without a GPU:

```bash
python -m src.evaluation evaluate --task food_recommendation --split test --prediction-dir <prediction-dir> --opportunities-root <food-opportunities-root> --output <food-report.json>
python -m src.evaluation compare --report <baseline-report.json> --report <foundation-report.json> --output <comparison.json>
```

Foundation Food/Ads artifacts already use `OutputContractV1`. A baseline must also provide
`evaluation_input_v1.json`; see [docs/evaluation_input_v1.md](docs/evaluation_input_v1.md).
The CPU-only Kaggle recipe is [notebooks/downstream_evaluation_kaggle.ipynb](notebooks/downstream_evaluation_kaggle.ipynb).
