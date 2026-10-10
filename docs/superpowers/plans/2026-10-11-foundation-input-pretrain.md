# Foundation Input Adapter & Tokenization trên Pretrain Data hiện tại — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng input adapter/tokenization dùng được với bộ pretrain được docs chỉ định là OmniBehavior V2, đồng thời tương thích với bộ Meituan TRD đã convert cùng schema.

**Architecture:** Dùng một pipeline loader/normalizer/tokenizer chung cho hai thư mục Parquet v2, nhưng mỗi run chỉ nhận một `dataset_root` và giữ namespace source riêng. Run chuẩn để verify là `data/pretrain_data/OmniBehavior_data`; run compatibility là `data/pretrain_data/meituan_data`.

**Tech Stack:** Python, PyArrow/Parquet, PyTorch tensors, dataclasses/type hints, pytest và ruff; text processing phải deterministic và offline.

**Spec:** `docs/parallel_vibe_implementation_plan.md`, `data/README_pretrain_data.md`, và ý tưởng kiến trúc trong PDF technical report được user cung cấp.

## Goal

Xây dựng phần input adapter/tokenization của Người 1 để biến các sample trong pretrain data hiện tại thành `FoundationBatch` dùng được bởi encoder và pretraining objectives của các thành viên khác.

Phạm vi dữ liệu:

- **Nguồn chuẩn theo docs:** `data/pretrain_data/OmniBehavior_data`, vì `data/README_pretrain_data.md` chỉ định OmniBehavior User Pretraining Observations v2 cho User Foundation Model pretraining.
- **Nguồn compatibility theo project:** `data/pretrain_data/meituan_data`, vì pipeline Meituan TRD đã convert dữ liệu này sang cùng schema `omnibehavior_pretraining_v2`, và downstream Food dùng user subset của bộ này.
- **Không mặc định concatenate hai nguồn.** Chạy pretraining riêng theo từng dataset trước; chỉ làm mixed-source run sau khi đã có experiment config, namespace và ablation được phê duyệt.
- **Không gộp user giữa các nguồn.** Dataset/source phải nằm trong namespace của metadata và vocabulary; `user_id` từ hai nguồn không được coi là cùng một người.
- **Không đọc raw source để tự suy diễn thêm dữ liệu.** Adapter dùng output pretrain v2 đã được chuẩn hóa; các field thiếu hoặc không chắc chắn phải được giữ là missing/unknown.

Kết quả phải thực hiện đúng contract trong `docs/parallel_vibe_implementation_plan.md`:

```text
[CLS] [tabular profile tokens...] [SEP] [event tokens...] [PAD...]
```

và trả về `FoundationBatch` với các tensor `input_ids`, `key_ids`, `modality_ids`, `source_ids`, `position_ids`, `time_gap_ids`, `attention_mask`, `user_ids`, cùng target metadata cho masking/next-action.

## Data Decision

### Dataset nào dùng để pretrain?

Plan này không coi hai thư mục hiện tại là hai shard của cùng một population:

| Run | Dataset root | Vai trò | Split | Kết quả mong đợi |
|---|---|---|---|---|
| Canonical foundation run | `data/pretrain_data/OmniBehavior_data` | Bộ được README chỉ định cho User Foundation Model | `train`, `validation`, `test` | Kiểm tra adapter đa scenario và temporal split |
| Meituan compatibility run | `data/pretrain_data/meituan_data` | Meituan TRD được convert sang schema v2 | hiện có `train` | Kiểm tra transfer/use case Food Recommendation |

Không dùng `data/README_taobao.md` làm nguồn pretraining trong plan này: tài liệu đó mô tả Taobao Display Advertising CTR, có nhãn `clk` cho downstream CTR, nhưng không mô tả nó là input dataset cho User Foundation Model.

### Data shape thực tế cần adapter

Mỗi snapshot là một training example:

```text
sample_id, user_id, cutoff_time, split
profile_json
history_event_ids
    -> events/{user_id}.parquet
       event_time / eligibility_time / time_precision
       scenario / event_type
       context_json / action_json
```

`OmniBehavior_data` có nhiều scenario và time-based train/validation/test. `meituan_data` có một snapshot train cho mỗi user theo pipeline hiện tại, với các event Food Delivery/E-commerce được tạo từ Meituan TRD. Loader phải dùng `manifest.json` và `preprocessing_config.json` của từng root, không hard-code semantics của một dataset vào tokenizer chung.

## Architecture

### Pipeline

```text
pretrain v2 Parquet
    -> manifest/source loader
    -> canonical normalized sample
    -> temporal/as-of validation
    -> train-only vocabulary + numeric metadata
    -> KeyValueToken/event groups
    -> padded FoundationBatch
    -> Person 2 encoder / Person 3 objectives
```

### Canonical sample adapter

Tạo một internal representation trong `src/foundation_input` để tách format Parquet hiện tại khỏi tokenizer:

- sample identity: `sample_id`, `dataset_name`, `split`, `user_id`;
- temporal metadata: `cutoff_time`, `cutoff_inclusive`, `history_order_known`, `history_truncated`, `history_group_truncated`;
- profile/tabular fields: dictionary đã parse từ `profile_json`;
- event groups: `event_id`, `source_record_id`, `source_record_index`, `action_index`, `event_time`, `eligibility_time`, `time_precision`, `scenario`, `event_type`, parsed `context`, parsed `action`;
- provenance flags: source field, missingness, uncertain ordering, and whether a value is eligible for a target.

Một event có thể tạo nhiều token field-level. Các token cùng `source_record_id` phải giữ cùng event group để sau này masking có thể phối hợp giữa event type, context và action, không làm mất tính nhất quán của một event.

### Mapping dữ liệu hiện tại

**Meituan profile/tabular**

- `active_days_30d`, `event_count_30d` và các count theo scenario là numeric profile features.
- `user_id` là categorical identity token, nhưng phải namespace theo dataset.
- `as_of_time` dùng để kiểm tra leakage; không đưa trực tiếp vào input như một business value nếu chưa có quyết định rõ ràng.

**Meituan/OmniBehavior event**

- metadata thời gian: `event_time`, `eligibility_time`, `time_precision`;
- categorical: `scenario`, `event_type`, các category/service/product-source fields;
- numeric: price, amount, item count, click count và duration khi có thật;
- text/string: title, description, caption, OCR/ASR và các field tương tự chỉ đưa qua deterministic text adapter hoặc giữ ở dạng normalized string metadata;
- action fields: parse theo schema thực tế, không coi `action.type` là success label nếu source không bảo đảm điều đó;
- field allowlist lấy từ `preprocessing_config.json`, không tự đưa toàn bộ JSON metadata vào vocabulary.

**Các field không được bịa**

- không tạo tọa độ/location nếu data không có tọa độ thật;
- không biến `order_price_interval` thành exact average/order amount;
- không suy ra ad impression/click/conversion từ event không có semantics tương ứng;
- không làm tăng precision cho date-only hoặc timestamp không chắc chắn;
- không dùng `source_timestamp` thay cho `event_time` nếu semantics không tương đương.

### Token layout và giới hạn

- Special IDs cố định: `PAD=0`, `CLS=1`, `SEP=2`, `MASK=3`, `UNK=4`.
- Tabular tokens là unordered: hoán vị thứ tự profile columns không được đổi nội dung thực của sample.
- Event groups giữ thứ tự theo `eligibility_time`, `eligibility_inclusive`, `source_record_index`, `action_index`; timestamp chỉ được sử dụng trong phạm vi precision mà source xác nhận.
- `max_history_events=30` của data là **số event**, không phải số token. Vì một event có nhiều field token, phải có thêm `max_tokens` độc lập và quy tắc truncate field deterministic.
- Quy tắc ban đầu: giữ tối đa 30 event groups; trong mỗi group ưu tiên event type/scenario, fields trong allowlist, action fields, rồi mới tới text fields theo priority cấu hình. Không truncate giữa các field bắt buộc của cùng một event group nếu còn cách tránh được.
- Padding chỉ thêm token `PAD`; không làm thay đổi token thật, time-gap thật hoặc targets thật.

### Target interface

Current pretrain data không lưu target. Người 1 chỉ tạo target metadata ở collator, không cài model loss:

- masked token positions và original value metadata cho masked reconstruction;
- event-group/source-record identity để Person 3 áp dụng coordinated masking;
- next-action candidate/event index và label metadata khi bật objective;
- token/field type để objective chọn CE, MSE/Huber hoặc bỏ qua.

Việc chọn mask rate, tỷ lệ numeric/categorical masking và chính sách next-action là config-level decision; không hard-code thành giả định của source data. Next-action target phải loại event đích khỏi input và không được nhìn thấy thông tin sau cutoff.

## Tech Stack

- Python hiện có trong repository.
- PyTorch tensors theo contract `FoundationBatch`.
- PyArrow/Parquet reader theo schema của pretrain v2. `data/requirements.txt` đã nêu `pyarrow>=17.0.0`, nhưng `pyproject.toml` hiện chưa khai báo runtime dependency; đây là dependency gate cần integration owner xử lý, không tự ý sửa config của team trong branch Người 1.
- Dataclass/type hints cho canonical sample, token metadata và batch.
- `pytest` cho unit/integration tests; `ruff` cho lint.
- Không dùng network tokenizer/API; text processing phải deterministic và offline.

## Spec

### File map của Người 1

Các file mới nằm dưới `src/foundation_input/**`; không sửa `src/data/**`, `data/**`, `src/models/**`, `src/training/**`, `src/downstream/**`, `src/baselines/**`, `src/evaluation/**` hoặc config chung nếu chưa được integration owner giao.

```text
src/foundation_input/
  __init__.py
  types.py          # KeyValueToken, canonical sample, FoundationBatch, target metadata
  manifest.py       # dataset root/manifest/schema/checksum validation
  loader.py         # read-only bridge từ Parquet v2 -> canonical sample
  validation.py     # temporal, as-of, split, JSON/value safety checks
  vocab.py          # train-only vocab, numeric normalizer/buckets, serialization
  tokenizer.py      # canonical sample -> field/event tokens -> encoded IDs
  dataset.py        # FoundationPretrainDataset + collator
tests/foundation_input/
  fixtures/         # sample nhỏ, không copy dataset thật vào repository
  test_manifest.py
  test_loader_validation.py
  test_vocab.py
  test_tokenizer.py
  test_dataset.py
docs/foundation_input_contract.md
```

Nếu cần smoke utility, đặt trong `tests/foundation_input/` hoặc một module read-only dưới `src/foundation_input`; không tạo pipeline mới trong `data/**`.

### Interfaces giữa các task

Các tên dưới đây là interface phải được chốt trước khi implement để Người 2/3 không phải đọc Parquet trực tiếp:

```python
def load_manifest(dataset_root: Path) -> DatasetManifest: ...

def iter_normalized_samples(
    dataset_root: Path,
    split: str,
    *,
    validate: bool = True,
) -> Iterator[CanonicalSample]: ...

def fit_vocabulary(
    train_samples: Iterable[CanonicalSample],
    config: VocabularyConfig,
) -> VocabularyArtifacts: ...

def tokenize_sample(
    sample: CanonicalSample,
    artifacts: VocabularyArtifacts,
    config: TokenizerConfig,
) -> list[TokenRecord]: ...

class FoundationPretrainDataset:
    def __getitem__(self, index: int) -> EncodedSample: ...

def foundation_collate(samples: list[EncodedSample]) -> FoundationBatch: ...
```

`CanonicalSample` phải giữ `dataset_name`, `split`, cutoff metadata, profile, event groups và uncertainty flags. `TokenRecord` phải giữ `event_group_id/source_record_id`; `EncodedSample` phải giữ mapping token-to-field để target builder không mask nhầm special/padding token.

### Task 0 — Freeze contract và fixtures trước khi code

**Mục tiêu:** chốt mapping của pretrain v2 vào interface chung trước khi phụ thuộc vào model.

**Thực hiện:**

- [ ] Ghi rõ canonical sample, field namespaces, event-group semantics và source limitations trong `docs/foundation_input_contract.md`.
- [ ] Tạo fixture nhỏ cho:
   - một Meituan sample có profile, click và purchase;
   - một OmniBehavior sample có nhiều scenario;
   - event có date-only/uncertain order;
   - missing/null/unknown categorical và numeric fields;
   - sample profile được tạo sau cutoff để test leakage.
- [ ] Định nghĩa enum/string constants cho dataset source, modality, token source và target kind.

**Verification:** chạy `python -m pytest tests/foundation_input/test_contract.py -v`; fixture của hai dataset phải normalize về cùng canonical interface nhưng vẫn giữ `dataset_name` khác nhau.

### Task 1 — Manifest và read-only loader

**Files:** `manifest.py`, `loader.py`, `test_manifest.py`.

**Thực hiện:**

- [ ] Validate root directory, manifest/data report, `schema_version`, required Parquet columns và split names.
- [ ] Kiểm tra checksum/index references nếu manifest có khai báo; lỗi phải chỉ ra dataset, file và field cụ thể.
- [ ] Đọc snapshot/event qua public pretrain v2 interface hoặc reader tương đương, không sửa data directory.
- [ ] Parse JSON profile/context/action với error context gồm `sample_id`, `event_id`, `source_record_id`.
- [ ] Giữ nguyên metadata `history_truncated`, `history_group_truncated`, `history_order_known`.

**Verification:** chạy `python -m pytest tests/foundation_input/test_manifest.py -v`; thiếu field, sai schema version, checksum mismatch, malformed JSON, unknown split và sample_id/event_id không khớp phải fail rõ ràng.

### Task 2 — Temporal/as-of validation

**Files:** `validation.py`, `test_loader_validation.py`.

**Thực hiện:**

- [ ] Enforce `user_id` của snapshot và event.
- [ ] Kiểm tra cutoff, eligibility và half-open history semantics; không để event sau cutoff đi vào input.
- [ ] Xác định thứ tự bằng các field đã có: `eligibility_time`, inclusive flag, source record index, action index.
- [ ] Với `time_precision` không chắc chắn hoặc date-only, giữ uncertainty flag và không giả tạo độ chính xác.
- [ ] Kiểm tra profile `as_of_time` không vượt cutoff.
- [ ] Validate finite numeric, non-negative count khi field semantics yêu cầu, và null/missing theo policy.
- [ ] Từ chối implicit cross-source user join; dataset identity phải là một phần của sample context.

**Policy:** có thể sort theo order key nếu đó là policy của loader; nếu input vi phạm invariant nghiêm trọng thì reject. Không âm thầm sửa giá trị business.

**Verification:** chạy `python -m pytest tests/foundation_input/test_loader_validation.py -v`; test out-of-order event, future profile, event sau cutoff, duplicate event identity, uncertain ordering, NaN/Inf và sample bị group truncate.

### Task 3 — Train-only vocabulary, normalizer và metadata

**Files:** `vocab.py`, `test_vocab.py`.

**Thực hiện:**

- [ ] Fit chỉ trên split train của đúng dataset run; validation/test chỉ transform.
- [ ] Vocabulary có namespace tối thiểu `dataset:modality:key`; không dùng chung `product_id`, category hoặc user ID giữa nguồn khi chưa có taxonomy chung.
- [ ] OOV/null/empty map về `UNK` hoặc missing policy; transform không được làm vocabulary tăng.
- [ ] Serialize vocabulary version, fit split, source namespace, counts, fingerprints và config.
- [ ] Numeric policy:
   - count fields có thể normalize theo policy đã ghi rõ;
   - numeric thực dùng scaler/bucket metadata fit trên train;
   - interval/bucket source như `order_price_interval` giữ đúng bucket semantics;
   - không coi bucket là exact amount.
- [ ] Text policy phải deterministic, offline, giới hạn kích thước và có OOV/hash collision metadata nếu dùng hashed IDs.
- [ ] Expose time-gap bucket boundaries; time gap không được leak absolute future information.

**Verification:** chạy `python -m pytest tests/foundation_input/test_vocab.py -v`; test train-only fit, OOV val/test, vocabulary fingerprint ổn định, source/field collision isolation, numeric missingness, interval preservation và round-trip serialization.

### Task 4 — Key/value tokenization

**Files:** `tokenizer.py`, `test_tokenizer.py`.

**Thực hiện:**

- [ ] Profile -> tabular `KeyValueToken` với `source="tabular"`, key ổn định và timestamp `None`.
- [ ] Event -> nhiều field tokens với `source="sequence"`, cùng event group/source record, timestamp theo event precision.
- [ ] Emit các field có trong allowlist; loại metadata không phải business feature.
- [ ] Emit modality rõ ràng: categorical, numeric, text, time hoặc missing.
- [ ] Encode key/modality/source/value qua metadata train-only và map unknown đúng policy.
- [ ] Tính `time_gap_ids` theo thứ tự event; tabular tokens không bị ép vào thứ tự thời gian của event.
- [ ] Giữ token/group provenance để coordinated masking và next-action objective hoạt động.
- [ ] Chặn token hóa field bị cấm hoặc field không có bằng chứng trong source.

**Verification:** chạy `python -m pytest tests/foundation_input/test_tokenizer.py -v`; test tabular column permutation invariant, stable event order, time-gap bucket, field namespace, missing/OOV, no invented location, no interval-to-exact conversion và cross-service event order.

### Task 5 — Dataset và collator trả về FoundationBatch

**Files:** `dataset.py`, `types.py`, `test_dataset.py`.

**Thực hiện:**

- [ ] Implement `FoundationPretrainDataset` theo split, dataset root, vocabulary artifact và tokenizer config.
- [ ] Implement collator layout `[CLS] [tabular...] [SEP] [event tokens...] [PAD...]`.
- [ ] Trả về đúng shape/dtype:
   - IDs là `torch.long`;
   - `attention_mask` là `torch.bool`, `True` cho token thật;
   - `position_ids`, `time_gap_ids`, provenance và targets có shape batch-aligned;
   - `user_ids` không làm mất dataset namespace.
- [ ] Tách `max_event_count=30` và `max_tokens`; truncate deterministically ở boundary event group.
- [ ] Không mask `[CLS]`, `[SEP]`, `[PAD]`; target metadata không chứa giá trị của token không còn trong input.
- [ ] Hỗ trợ zero-event và missing-profile sample với special/missing representation hợp lệ.
- [ ] Không lưu toàn bộ dataset vào RAM; loader phải phù hợp với Parquet hiện tại.

**Verification:** chạy `python -m pytest tests/foundation_input/test_dataset.py -v`; test batch shape/dtype, padding invariance, max event/token truncation, zero event, mixed source batch policy, no target leakage và deterministic collation.

### Task 6 — Smoke run trên pretrain data thật

**Mục tiêu:** chứng minh adapter chạy trên data hiện tại, không chỉ trên fixture.

**Thực hiện:**

- [ ] Chạy 100 sample train từ `OmniBehavior_data` trước, vì đây là canonical pretraining source theo README.
- [ ] Chạy 100 sample train từ `meituan_data` sau, vì đây là compatibility source cho downstream Food.
- [ ] Thu metrics tối thiểu:
   - sample/event/token count;
   - tỷ lệ missing/OOV theo modality/key;
   - số event order violation và số sample uncertain-order;
   - số sample bị truncate theo event/group/token;
   - max/percentile token length;
   - user/source namespace collision check;
   - batch contract/dtype check.
- [ ] Lưu report ở thư mục tạm hoặc output đã thống nhất; không ghi đè pretrain data.
- [ ] Nếu environment thiếu `pyarrow`, dừng ở dependency gate và báo rõ; không đánh dấu smoke pass bằng fixture là đã pass data thật.

**Verification commands:**

```powershell
python -m pytest tests/foundation_input -q
python -m ruff check src/foundation_input tests/foundation_input
```

### Task 7 — Verification và handoff

**Thực hiện:**

- [ ] Chạy unit tests của `tests/foundation_input/`.
- [ ] Chạy toàn bộ `pytest` và `ruff` theo command repository.
- [ ] Kiểm tra import độc lập của `src.foundation_input`.
- [ ] Review diff để đảm bảo không sửa ngoài ownership boundary.
- [ ] Gửi cho Người 2 một contract example gồm batch shapes, token layout, special IDs và target metadata; gửi cho Người 3 chính sách masking/next-action assumptions.
- [ ] Ghi rõ limitation còn lại: current data chỉ train-only cho Meituan generated set, nguồn timezone/precision không đồng nhất, và PDF chưa chốt mask rate/max token length.

## Global Constraints

- `docs/parallel_vibe_implementation_plan.md` là contract triển khai và ownership; PDF là tài liệu ý tưởng/POC, không tự động override repository contract.
- Không sửa hoặc regenerate `data/pretrain_data/**` trong task này.
- Không join user giữa Meituan, OmniBehavior, Taobao hoặc các source khác.
- Không biến event thiếu semantics thành label/target mạnh hơn dữ liệu gốc cho phép.
- Không fit vocabulary, scaler, bucket boundary hoặc taxonomy trên validation/test.
- Không random split pretraining data; split/time boundary phải được source loader giữ nguyên.
- Không dùng absolute future time hoặc profile sau cutoff.
- Không thêm dependency network/runtime tokenizer bên ngoài khi chưa có quyết định của integration owner.
- Nếu cần thêm `pyarrow` vào `pyproject.toml`, tạo follow-up integration task; Người 1 không tự sửa config chung trong branch này.
- Mỗi commit nên giữ một thay đổi logic nhỏ, có test tương ứng; không bundle model/training/downstream changes vào branch adapter.

## Review Focus

Reviewer cần tập trung vào các điểm rủi ro sau:

1. **Leakage:** profile/event sau cutoff, next-action target còn nằm trong input, hoặc scaler/vocab fit nhầm trên non-train split.
2. **Temporal correctness:** sort key có đúng với `eligibility_time`/inclusive/index không; date-only/uncertain timestamps có bị giả precision không.
3. **Schema fidelity:** context/action JSON có được map theo allowlist và semantics thật không; interval có bị dùng như exact value không.
4. **Token contract:** special IDs, tensor dtype, attention mask, padding, tabular unordered semantics và event-group boundary có đúng không.
5. **Source isolation:** user/key vocabulary có namespace theo source không; batch mixed-source có policy rõ ràng không.
6. **Scalability:** không load 4GB+ dataset vào RAM; max 30 event không bị hiểu nhầm là max 30 token.
7. **Integration:** batch output khớp input mà Person 2 sẽ consume; target metadata đủ cho Person 3 nhưng không lấn sang model loss.
8. **Reproducibility:** vocabulary/config fingerprint, deterministic tokenization, smoke metrics và command chạy lại được.

## Definition of Done

- Hai pretrain source hiện tại đều có thể normalize vào cùng interface, với source limitations được bảo toàn.
- Unit tests cho manifest, temporal validation, train-only vocab, tokenization, collator và leakage đều pass.
- Smoke run trên Meituan và OmniBehavior train samples pass trong environment có `pyarrow` phù hợp.
- `FoundationBatch` khớp contract và có thể được Person 2 consume mà không cần đọc raw Parquet.
- Không có thay đổi ngoài ownership boundary; docs contract và artifact metadata đủ để tái lập run.
