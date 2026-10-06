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
