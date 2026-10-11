# Foundation Encoder and Dual User Embeddings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Member 2 foundation encoder: modality-aware token representations, safe bidirectional/next-action Transformer attention, and stable `long_term` and `short_term` user embeddings consumable by pretraining and both downstream tasks.

**Architecture:** `FoundationModel` consumes the existing `FoundationBatch` produced by `src.foundation_input`, maps each valid token into `d_model`, adds only sequence-appropriate positional/time features, and runs a CPU-compatible PyTorch Transformer. The User-ID adapter creates `long_term`; a masked sequence pooler creates `short_term`; downstream code receives them only through `embed_users()`.

**Tech Stack:** Python 3.10+, PyTorch CPU, dataclasses, pytest, ruff. PyTorch is installed in the current development environment; Member 3 owns adding its runtime dependency to `pyproject.toml`.

**Spec:** `docs/parallel_vibe_implementation_plan.md` sections 4.1–4.3, 4.6A, 5 (Người 2), and 7; `docs/foundation_input_contract.md`; `src/foundation_input/types.py`; `src/foundation_input/dataset.py`.

## Global Constraints

- Work on branch `feat/foundation-encoder` and only create/modify `src/models/adapters.py`, `src/models/encoder.py`, `src/models/pooling.py`, and `tests/models/**`.
- Do not modify `src/foundation_input/**`, `src/models/objectives.py`, `src/training/**`, `src/downstream/**`, `src/baselines/**`, `src/evaluation/**`, `src/data/**`, `data/**`, or YAML configs.
- `FoundationBatch.attention_mask` is `torch.bool`, `[B, L]`, and `True` means a real token. PyTorch `src_key_padding_mask` uses the inverse convention: `True` means ignore.
- Batch layout is fixed: `[CLS] [tabular...] [SEP] [event sequence...] [PAD...]`; special IDs are `PAD=0`, `CLS=1`, `SEP=2`, `MASK=3`, `UNK=4`.
- Vocabularies, numeric bucket boundaries, text hash policy, and source/modality IDs are already fit by Member 1 on the train split. This branch must never fit or mutate them.
- Do not read Parquet, manifests, raw profile/event dictionaries, candidates, labels, or future data from this branch.
- No network model, pretrained text model, checkpoint, artifact, or generated dataset is committed to Git.
- `FoundationOutput.hidden_states`, `long_term`, and `short_term` must be `torch.float32`, batch-aligned, and finite at every valid token. A malformed input must raise a contextual `ValueError`, never return a partial embedding.
- In `eval()` mode, same model weights and same batch must produce the same output within `1e-6` on the same device.

## Upstream Input-Contract Requirements

The checked-in source is still at the pre-extension state: `FoundationBatch` currently has only ID tensors (`input_ids`, `key_ids`, `modality_ids`, `source_ids`, `position_ids`, `time_gap_ids`), a validity mask, user IDs, and target metadata. The selected v2 contract adds required raw normalized numeric floats and numeric missing flags. It does **not** add latitude/longitude tensors because the current data has no geo business fields. `tokenizer.py` emits `categorical`, `numeric`, `text`, and `missing`; it does not emit `geo`.

`sequence_source_id` is not a universal constant: Member 3 must take it from the selected artifact's `source_to_id["sequence"]` mapping, alongside returned cardinalities.

Tasks 1–6 in this plan can be implemented against fixture batches immediately. Member 3 must not train a production foundation checkpoint until Member 1 has merged the following mandatory input-contract repairs:

1. **Stable OOV user identity.** In `encode_sample`, set `user_id_value_id=None` when the `user_id` token is `UNK=4`; this makes the existing collator invoke `_stable_user_id(dataset_name, user_id)`. Retain a train-vocabulary ID only when it is not a special ID. Add a test with two validation users outside the train vocabulary and assert their `batch.user_ids` differ and are deterministic.
2. **Safe next-action eligibility.** With `DatasetConfig.next_action_prediction=True`, reject a sample where `history_order_known=False`, where the final retained target event group is incomplete, or where fewer than two complete sequence event groups remain. The error must contain `sample_id` and the violated condition. An unordered history may still be used for bidirectional masked-value pretraining; it must not create a next-action target.
3. **Group identity independent of temporal position.** Build `targets["event_group_ids"]` from a per-sample mapping of each retained `TokenRecord.token.event_group_id` to a nonzero group ID. Keep `position_ids=0` when history order is unknown, but never use position IDs as group IDs. This preserves coordinated masking without inventing temporal order.
4. **Recent-event retention.** When token limits truncate a known event history, retain the most recent complete event groups, not the earliest groups. Reserve enough token budget for at least one sequence group; in next-action mode reserve two complete groups so one remains input context after the final group becomes the target. If a single event group cannot fit, reject that sample with its `sample_id`; do not split the group.
5. **Published cardinalities.** Add `VocabularyArtifacts.model_cardinalities()` returning explicit `value_vocab_size`, `key_vocab_size`, `modality_vocab_size`, `source_vocab_size`, `time_gap_vocab_size`, and `text_hash_buckets`. Each returned vocabulary size is `max_possible_encoded_id + 1`, including special IDs, numeric bucket IDs, and text-hash IDs. Member 3 passes only the supported `*_vocab_size` fields to `FoundationModelConfig`; `text_hash_buckets` is metadata unless the config explicitly accepts it. It must not use the count of Python dictionary entries as an embedding size.
6. **One modality naming contract.** Document `numeric` as the canonical existing modality spelling, and document `missing` as a current ID-level modality. Do not have one teammate route `numerical` while another emits `numeric`.

After those repairs, this encoder consumes the selected full-numeric v2 contract:

1. categorical, text, and missing tokens route through ID-based adapters; numeric values use Member 1's train-normalized continuous payload plus missing mask through a numeric MLP, while bucket IDs remain available for masked-value objectives;
2. `input_ids` are reused per vocabulary namespace, so a categorical representation is keyed by the deterministic pair `(key_id, input_id)`, not `input_id` alone;
3. the Member 2 `UserIdAdapter` still maps every stable ID into `num_user_buckets` with `torch.remainder`;
4. no geo token is routed because the current input data does not provide it.

### Selected full-numeric handoff

- Member 1 adds required `numeric_values: FloatTensor [B,L]` and `numeric_missing_mask: BoolTensor [B,L]` to `FoundationBatch`. Numeric normalization statistics and transform policies are fit only on train and serialized with vocabulary metadata. Missing values are represented by a finite zero payload plus their mask, never NaN.
- Member 2 adds `normalized float + missing flag -> MLP` for numeric tokens. Categorical/text/missing pair-ID routing, masks, Transformer, pooling, and public output API do not change.
- Geo remains unsupported until Data confirms meaningful latitude/longitude. Member 1 must not manufacture coordinates.

Record the selected full-numeric v2 contract in the encoder PR description and ask Member 1 and Member 3 to acknowledge the required constructor change before merge.

## File Structure

```text
src/models/
  adapters.py        # Config, per-modality adapters, token feature registry, User-ID adapter
  pooling.py         # Sequence-only pooling with CLS fallback
  encoder.py         # FoundationOutput, attention masks, Transformer, public model API
tests/models/
  conftest.py        # Deterministic, small FoundationBatch and model config fixtures
  test_adapters.py   # Routing, pair-ID behavior, sequence-only temporal features, user hashing
  test_pooling.py    # Event-only pooling and zero-event fallback
  test_encoder.py    # Shapes, padding, masks, causality, public API and error handling
```

`src/models/__init__.py` is intentionally unchanged. Member 3 can import public classes from their concrete modules until an integration owner explicitly requests package exports.

---

### Task 1: Define encoder configuration and test fixtures

**Files:**
- Create: `tests/models/conftest.py`
- Create: `tests/models/test_encoder.py`
- Modify: `src/models/encoder.py`

**Interfaces:**
- Consumes: `src.foundation_input.types.FoundationBatch` v2 after Member 1's numeric extension is merged.
- Produces: `FoundationModelConfig` and `FoundationOutput`, imported later as `from src.models.encoder import FoundationModelConfig, FoundationOutput`.

- [ ] **Step 1: Write failing configuration and batch-contract tests**

```python
# tests/models/test_encoder.py
import pytest

from src.models.encoder import FoundationModelConfig


def test_config_rejects_non_divisible_attention_dimension() -> None:
    with pytest.raises(ValueError, match="d_model=.*num_heads"):
        FoundationModelConfig(d_model=30, num_heads=8)


def test_config_rejects_unknown_default_attention_mode() -> None:
    with pytest.raises(ValueError, match="default_attention_mode"):
        FoundationModelConfig(default_attention_mode="causal")
```

Create `batch_factory()` in `conftest.py`. It must return a real CPU `FoundationBatch` v2 of shape `[2, 7]` with layout `[CLS, tabular, SEP, event-1, event-2, PAD, PAD]` for row zero and a zero-event row one. With the current sorted fixture vocabulary, use source ID `1` for sequence and `2` for tabular. Make every tensor explicit: IDs are `torch.long`, the attention and numeric missing masks are `torch.bool`, and `numeric_values` is `torch.float32`. Also create `model_config(**overrides)` returning `FoundationModelConfig(d_model=16, num_heads=4, num_layers=2, dropout=0.0, value_vocab_size=64, key_vocab_size=32, modality_vocab_size=8, source_vocab_size=4, time_gap_vocab_size=8, num_user_buckets=17, categorical_pair_buckets=97, sequence_source_id=1, **overrides)` so every test has valid ID-table capacity.

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
python -m pytest tests/models/test_encoder.py::test_config_rejects_non_divisible_attention_dimension -v
```

Expected: FAIL because `FoundationModelConfig` does not exist.

- [ ] **Step 3: Add the exact public dataclasses**

```python
# src/models/encoder.py
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FoundationModelConfig:
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    max_length: int = 256
    value_vocab_size: int = 50_007
    key_vocab_size: int = 1
    modality_vocab_size: int = 1
    source_vocab_size: int = 1
    time_gap_vocab_size: int = 8
    num_user_buckets: int = 50_000
    categorical_pair_buckets: int = 100_003
    sequence_source_id: int = 1
    default_attention_mode: str = "bidirectional"

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if self.num_layers <= 0 or self.max_length <= 0:
            raise ValueError("num_layers and max_length must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0)")
        if self.default_attention_mode not in {"bidirectional", "next_action"}:
            raise ValueError("default_attention_mode must be bidirectional or next_action")
        for name in (
            "value_vocab_size", "key_vocab_size", "modality_vocab_size",
            "source_vocab_size", "time_gap_vocab_size", "num_user_buckets",
            "categorical_pair_buckets",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass
class FoundationOutput:
    hidden_states: torch.Tensor
    long_term: torch.Tensor
    short_term: torch.Tensor
    padding_mask: torch.BoolTensor
```

- [ ] **Step 4: Run the configuration tests**

Run:

```powershell
python -m pytest tests/models/test_encoder.py -q
```

Expected: configuration tests PASS; forward tests are not added until Task 5.

- [ ] **Step 5: Commit the isolated API scaffold**

```powershell
git add src/models/encoder.py tests/models/conftest.py tests/models/test_encoder.py
git commit -m "feat: define foundation encoder contracts"
```

### Task 2: Implement modality adapters and safe User-ID lookup

**Files:**
- Create: `src/models/adapters.py`
- Create: `tests/models/test_adapters.py`

**Interfaces:**
- Consumes: `FoundationBatch` and `FoundationModelConfig`.
- Produces: `AdapterRegistry.forward(batch) -> torch.Tensor` of shape `[B, L, D]`; `UserIdAdapter.forward(user_ids) -> torch.Tensor` of shape `[B, D]`.
- Later used by: `FoundationModel` in `src/models/encoder.py`.

- [ ] **Step 1: Write failing adapter tests**

```python
# tests/models/test_adapters.py
import torch

from src.models.adapters import AdapterRegistry, UserIdAdapter
from src.models.encoder import FoundationModelConfig


def test_pair_keying_distinguishes_reused_value_ids(batch_factory) -> None:
    batch = batch_factory()
    batch.input_ids[0, 1] = 9
    batch.input_ids[0, 3] = 9
    batch.key_ids[0, 1] = 3
    batch.key_ids[0, 3] = 4
    registry = AdapterRegistry(FoundationModelConfig(d_model=16, num_heads=4))
    representations = registry(batch)
    assert not torch.equal(representations[0, 1], representations[0, 3])


def test_large_hashed_user_id_is_mapped_inside_embedding_range() -> None:
    adapter = UserIdAdapter(num_user_buckets=17, d_model=8)
    output = adapter(torch.tensor([2**62 + 123, 5], dtype=torch.long))
    assert output.shape == (2, 8)
    assert torch.isfinite(output).all()
```

Also write a test that verifies position and time-gap changes alter a sequence token but do not alter a tabular token, with dropout disabled and identical base IDs.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/models/test_adapters.py -v
```

Expected: FAIL because `adapters.py` does not exist.

- [ ] **Step 3: Implement focused adapters**

Define these public classes in `adapters.py`:

```python
class UserIdAdapter(nn.Module):
    def __init__(self, *, num_user_buckets: int, d_model: int) -> None: ...
    def forward(self, user_ids: torch.LongTensor) -> torch.Tensor: ...

class GeoAdapter(nn.Module):
    def __init__(self, *, d_model: int) -> None: ...
    def forward(
        self,
        coordinates: torch.Tensor,  # [B, L, 2]
        missing_mask: torch.BoolTensor,  # [B, L]
    ) -> torch.Tensor: ...

class AdapterRegistry(nn.Module):
    def __init__(self, config: FoundationModelConfig) -> None: ...
    def forward(self, batch: FoundationBatch) -> torch.Tensor: ...
```

Implementation rules:

```python
pair_ids = torch.remainder(
    batch.key_ids.to(torch.long) * 1_000_003 + batch.input_ids.to(torch.long),
    self.config.categorical_pair_buckets,
)
base = self.value_embedding(batch.input_ids.clamp(0, self.config.value_vocab_size - 1))
pair = self.pair_embedding(pair_ids)
features = base + pair + self.key_embedding(...) + self.modality_embedding(...) + self.source_embedding(...)

sequence_mask = batch.attention_mask & batch.source_ids.eq(self.config.sequence_source_id)
features = features + self.position_embedding(position_ids) * sequence_mask.unsqueeze(-1)
features = features + self.time_gap_embedding(time_gap_ids) * sequence_mask.unsqueeze(-1)
features = features.masked_fill(~batch.attention_mask.unsqueeze(-1), 0.0)
```

Use `padding_idx=0` for all token-table embeddings where index zero is padding. Clamp only ID values used by tables after validating that all IDs are non-negative. Reject a negative ID with a `ValueError` that names the offending field. `GeoAdapter` must concatenate normalized coordinates with `sin` and `cos` coordinate features, project with a two-layer MLP, and return zero rows where `missing_mask=True`; it is not called by `AdapterRegistry` until a future batch contract supplies geo tensors.

`UserIdAdapter.forward()` must use:

```python
bucket_ids = torch.remainder(user_ids.to(torch.long), self.embedding.num_embeddings)
return self.embedding(bucket_ids)
```

- [ ] **Step 4: Run adapter tests**

Run:

```powershell
python -m pytest tests/models/test_adapters.py -q
```

Expected: PASS. Confirm every valid token representation is finite and padding rows are all zero.

- [ ] **Step 5: Commit modality work**

```powershell
git add src/models/adapters.py tests/models/test_adapters.py
git commit -m "feat: add foundation modality adapters"
```

### Task 3: Implement sequence-only short-term pooling

**Files:**
- Create: `src/models/pooling.py`
- Create: `tests/models/test_pooling.py`

**Interfaces:**
- Consumes: `hidden_states: torch.Tensor [B, L, D]`, `attention_mask: torch.BoolTensor [B, L]`, `source_ids: torch.LongTensor [B, L]`, and `sequence_source_id: int`.
- Produces: `SequencePooler.forward(...) -> torch.Tensor [B, D]`.
- Later used by: `FoundationModel.forward`.

- [ ] **Step 1: Write failing pooling tests**

```python
# tests/models/test_pooling.py
import torch

from src.models.pooling import SequencePooler


def test_pooler_excludes_tabular_and_padding_tokens() -> None:
    hidden = torch.tensor([[[100.0, 100.0], [2.0, 2.0], [4.0, 4.0], [999.0, 999.0]]])
    valid = torch.tensor([[True, True, True, False]])
    source = torch.tensor([[0, 1, 1, 1]])
    result = SequencePooler(sequence_source_id=1)(hidden, valid, source)
    assert torch.equal(result, torch.tensor([[3.0, 3.0]]))


def test_pooler_uses_cls_when_user_has_no_event() -> None:
    hidden = torch.tensor([[[7.0, 8.0], [2.0, 2.0]]])
    valid = torch.tensor([[True, True]])
    source = torch.tensor([[0, 2]])
    result = SequencePooler(sequence_source_id=1)(hidden, valid, source)
    assert torch.equal(result, torch.tensor([[7.0, 8.0]]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/models/test_pooling.py -v
```

Expected: FAIL because `pooling.py` does not exist.

- [ ] **Step 3: Implement the pooler with explicit validation**

```python
# src/models/pooling.py
class SequencePooler(nn.Module):
    def __init__(self, *, sequence_source_id: int) -> None:
        super().__init__()
        self.sequence_source_id = sequence_source_id

    def forward(self, hidden_states, attention_mask, source_ids):
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [B, L, D]")
        event_mask = attention_mask & source_ids.eq(self.sequence_source_id)
        weights = event_mask.unsqueeze(-1).to(hidden_states.dtype)
        pooled_events = (hidden_states * weights).sum(dim=1)
        event_count = weights.sum(dim=1)
        pooled = pooled_events / event_count.clamp_min(1.0)
        return torch.where(event_count.gt(0), pooled, hidden_states[:, 0, :])
```

Validate that the three tensors share `[B, L]` dimensions and that CLS is valid when an eventless fallback is needed. Raise an error naming the expected and actual shapes otherwise.

- [ ] **Step 4: Run pooling tests**

Run:

```powershell
python -m pytest tests/models/test_pooling.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit pooling**

```powershell
git add src/models/pooling.py tests/models/test_pooling.py
git commit -m "feat: add sequence user embedding pooler"
```

### Task 4: Build and test leak-safe attention masks

**Files:**
- Modify: `src/models/encoder.py`
- Modify: `tests/models/test_encoder.py`

**Interfaces:**
- Consumes: `attention_mask`, `source_ids`, and `position_ids` from a `[B, L]` batch.
- Produces: `build_attention_mask(batch, attention_mode) -> torch.BoolTensor [B, L, L]`, where `True` blocks a query-to-key edge.
- Later used by: `FoundationModel.forward` to create per-example Transformer masks.

- [ ] **Step 1: Write failing mask tests**

```python
from src.models.encoder import build_attention_mask


def test_next_action_blocks_future_events_but_allows_profile_context(batch_factory) -> None:
    batch = batch_factory()
    blocked = build_attention_mask(batch, "next_action")
    # query event at index 3 cannot see event at index 4, whose group position is later
    assert blocked[0, 3, 4].item() is True
    # the same event can see tabular context at index 1
    assert blocked[0, 3, 1].item() is False
    # tabular query cannot absorb event information and leak it in a later layer
    assert blocked[0, 1, 3].item() is True


def test_bidirectional_does_not_add_causal_blocks(batch_factory) -> None:
    batch = batch_factory()
    blocked = build_attention_mask(batch, "bidirectional")
    assert blocked[0, 3, 4].item() is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/models/test_encoder.py::test_next_action_blocks_future_events_but_allows_profile_context -v
```

Expected: FAIL because `build_attention_mask` does not exist.

- [ ] **Step 3: Implement the exact mask policy**

`build_attention_mask` must first validate the mode and shapes. Start with a `False` `[B, L, L]` tensor. For `bidirectional`, leave it entirely false; padding is handled separately by `src_key_padding_mask`.

For `next_action`, use:

```python
valid = batch.attention_mask
is_event = valid & batch.source_ids.eq(sequence_source_id)
future_event = batch.position_ids.unsqueeze(1) > batch.position_ids.unsqueeze(2)
query_is_event = is_event.unsqueeze(2)
key_is_event = is_event.unsqueeze(1)

# Event query cannot view a later event group.
blocked |= query_is_event & key_is_event & future_event
# Non-event query must never read event content; this avoids multi-layer leakage.
blocked |= (~is_event).unsqueeze(2) & key_is_event
```

The implementation may accept `sequence_source_id` as an explicit keyword argument. It must not use raw token order as a proxy for future; it must compare group-level `position_ids`, so fields belonging to one event can attend to each other.

- [ ] **Step 4: Run all mask tests**

Run:

```powershell
python -m pytest tests/models/test_encoder.py -q
```

Expected: mask tests PASS.

- [ ] **Step 5: Commit the masking contract**

```powershell
git add src/models/encoder.py tests/models/test_encoder.py
git commit -m "feat: add leak-safe foundation attention masks"
```

### Task 5: Implement FoundationModel and public embedding API

**Files:**
- Modify: `src/models/encoder.py`
- Modify: `tests/models/test_encoder.py`

**Interfaces:**
- Consumes: `AdapterRegistry`, `UserIdAdapter`, `SequencePooler`, `FoundationBatch`, and `FoundationModelConfig`.
- Produces:

```python
class FoundationModel(nn.Module):
    def forward(self, batch: FoundationBatch, *, attention_mode: str) -> FoundationOutput: ...
    def encode(self, batch: FoundationBatch) -> FoundationOutput: ...
    def embed_users(self, batch: FoundationBatch) -> dict[str, torch.Tensor]: ...
```

- [ ] **Step 1: Write failing forward/API tests**

```python
import torch

from src.models.encoder import FoundationModel, FoundationModelConfig


def test_forward_returns_contract_shapes_and_finite_values(batch_factory) -> None:
    model = FoundationModel(FoundationModelConfig(d_model=16, num_heads=4, num_layers=2, dropout=0.0))
    output = model(batch_factory(), attention_mode="bidirectional")
    assert output.hidden_states.shape == (2, 7, 16)
    assert output.long_term.shape == (2, 16)
    assert output.short_term.shape == (2, 16)
    assert output.padding_mask.dtype is torch.bool
    assert torch.isfinite(output.hidden_states[output.padding_mask]).all()


def test_embed_users_returns_only_the_two_public_embeddings(batch_factory) -> None:
    model = FoundationModel(FoundationModelConfig(d_model=16, num_heads=4, num_layers=1, dropout=0.0))
    model.eval()
    result = model.embed_users(batch_factory())
    assert set(result) == {"long_term", "short_term"}
    assert result["long_term"].shape == (2, 16)


def test_eval_encoding_is_deterministic(batch_factory) -> None:
    model = FoundationModel(FoundationModelConfig(d_model=16, num_heads=4, num_layers=1, dropout=0.2))
    model.eval()
    batch = batch_factory()
    first = model.encode(batch).short_term
    second = model.encode(batch).short_term
    torch.testing.assert_close(first, second, rtol=0.0, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/models/test_encoder.py::test_forward_returns_contract_shapes_and_finite_values -v
```

Expected: FAIL because `FoundationModel` does not exist.

- [ ] **Step 3: Implement the minimal encoder**

Construct `nn.TransformerEncoderLayer` with `batch_first=True`, `d_model=config.d_model`, `nhead=config.num_heads`, `dropout=config.dropout`, and `activation="gelu"`; wrap it in `nn.TransformerEncoder` with `num_layers=config.num_layers`.

Implement `forward` as follows:

```python
token_features = self.adapters(batch)                         # [B, L, D]
edge_mask = build_attention_mask(batch, attention_mode, self.config.sequence_source_id)
padding_mask = ~batch.attention_mask                           # True means ignored by PyTorch

# TransformerEncoder accepts one [L, L] mask, not a distinct [B, L, L]
# mask in all supported PyTorch versions. Encode each row with its own
# [L, L] edge mask, then concatenate in original batch order.
encoded_rows = []
for index in range(token_features.shape[0]):
    encoded_rows.append(
        self.transformer(
            token_features[index : index + 1],
            mask=edge_mask[index],
            src_key_padding_mask=padding_mask[index : index + 1],
        )
    )
hidden_states = torch.cat(encoded_rows, dim=0)
hidden_states = hidden_states.masked_fill(~batch.attention_mask.unsqueeze(-1), 0.0)
long_term = self.user_adapter(batch.user_ids)
short_term = self.pooler(hidden_states, batch.attention_mask, batch.source_ids)
return FoundationOutput(hidden_states, long_term, short_term, batch.attention_mask)
```

Implement `encode(batch)` as `forward(batch, attention_mode=self.config.default_attention_mode)`. Implement `embed_users(batch)` by calling `encode` and returning exactly the two public tensors.

Before returning, verify shape agreement, dtype `torch.float32`, device agreement, and finiteness at valid tokens. Add this private helper to make the check concrete:

```python
def _validate_output(output: FoundationOutput, batch: FoundationBatch) -> None:
    batch_size, sequence_length = batch.input_ids.shape
    expected_hidden = (batch_size, sequence_length, output.hidden_states.shape[-1])
    for name, value, expected in (
        ("hidden_states", output.hidden_states, expected_hidden),
        ("long_term", output.long_term, (batch_size, output.hidden_states.shape[-1])),
        ("short_term", output.short_term, (batch_size, output.hidden_states.shape[-1])),
    ):
        if value.shape != expected or value.dtype != torch.float32:
            raise ValueError(
                f"{name}: expected shape={expected}, dtype=torch.float32; "
                f"got shape={tuple(value.shape)}, dtype={value.dtype}; batch_size={batch_size}"
            )
        if value.device != batch.input_ids.device:
            raise ValueError(f"{name}: output device differs from batch device; batch_size={batch_size}")
    if output.padding_mask.dtype != torch.bool or not torch.equal(output.padding_mask, batch.attention_mask):
        raise ValueError(f"padding_mask must equal attention_mask; batch_size={batch_size}")
    if not torch.isfinite(output.hidden_states[batch.attention_mask]).all():
        raise ValueError(f"hidden_states contains non-finite valid values; batch_size={batch_size}")
    if not torch.isfinite(output.long_term).all() or not torch.isfinite(output.short_term).all():
        raise ValueError(f"user embedding contains non-finite values; batch_size={batch_size}")
```

Do not inspect or validate padded floating values as an output invariant.

- [ ] **Step 4: Run forward and API tests**

Run:

```powershell
python -m pytest tests/models/test_encoder.py -q
```

Expected: all shape, dtype, finiteness, API, and deterministic-eval tests PASS.

- [ ] **Step 5: Commit the model**

```powershell
git add src/models/encoder.py tests/models/test_encoder.py
git commit -m "feat: add foundation transformer encoder"
```

### Task 6: Prove padding, unordered tabular invariance, and future-event isolation

**Files:**
- Modify: `tests/models/test_encoder.py`

**Interfaces:**
- Consumes: completed `FoundationModel` public API only.
- Produces: regression tests protecting the three model invariants relied on by Member 3.

- [ ] **Step 1: Write the three regression tests**

```python
def test_changing_padding_ids_does_not_change_valid_outputs(batch_factory) -> None:
    model = FoundationModel(FoundationModelConfig(d_model=16, num_heads=4, num_layers=2, dropout=0.0)).eval()
    original = batch_factory()
    changed = batch_factory()
    changed.input_ids[0, 5:] = torch.tensor([31, 47])
    changed.key_ids[0, 5:] = torch.tensor([9, 10])
    first = model.encode(original).hidden_states[0, :5]
    second = model.encode(changed).hidden_states[0, :5]
    torch.testing.assert_close(first, second, rtol=0.0, atol=1e-6)


def test_permuting_tabular_columns_preserves_short_term_in_eval(batch_factory) -> None:
    model = FoundationModel(FoundationModelConfig(d_model=16, num_heads=4, num_layers=2, dropout=0.0)).eval()
    first = batch_factory(two_tabular_tokens=True)
    second = batch_factory(two_tabular_tokens=True)
    second.input_ids[0, 1:3] = second.input_ids[0, [2, 1]]
    second.key_ids[0, 1:3] = second.key_ids[0, [2, 1]]
    torch.testing.assert_close(model.encode(first).short_term, model.encode(second).short_term, rtol=0.0, atol=1e-6)


def test_future_event_cannot_change_past_event_hidden_state(batch_factory) -> None:
    model = FoundationModel(FoundationModelConfig(d_model=16, num_heads=4, num_layers=2, dropout=0.0)).eval()
    first = batch_factory()
    changed = batch_factory()
    changed.input_ids[0, 4] = 37
    before = model(first, attention_mode="next_action").hidden_states[0, 3]
    after = model(changed, attention_mode="next_action").hidden_states[0, 3]
    torch.testing.assert_close(before, after, rtol=0.0, atol=1e-6)
```

`batch_factory(two_tabular_tokens=True)` must place the second tabular token before SEP and preserve the same `position_id=0`, `time_gap_id=0`, `source_id=2` for both profile tokens.

- [ ] **Step 2: Run tests to verify their initial behavior**

Run:

```powershell
python -m pytest tests/models/test_encoder.py -q
```

Expected: PASS. If the future-event test fails, do not weaken the assertion; inspect `build_attention_mask` for a path from an event to tabular/CLS/SEP and then back to a past event across Transformer layers.

- [ ] **Step 3: Add malformed-batch coverage**

Add tests asserting that a negative `input_ids` value, mismatched `[B, L]` tensor shape, unsupported `attention_mode`, and an eventless row with invalid CLS each raise `ValueError` naming the failing field.

- [ ] **Step 4: Run the complete Member 2 suite**

Run:

```powershell
python -m pytest tests/models -q
python -m ruff check src/models tests/models
```

Expected: all tests and lint checks PASS.

- [ ] **Step 5: Commit regression protection**

```powershell
git add tests/models/test_encoder.py
git commit -m "test: protect foundation encoder invariants"
```

### Task 7: Integrate with the public input-batch API without changing its ownership

**Files:**
- Modify: `tests/models/test_encoder.py`

**Interfaces:**
- Consumes: Member 1's `fit_vocabulary_from_samples`, `tokenize_sample`, and `foundation_collate`; Member 2's `FoundationModel`.
- Produces: a CPU smoke test demonstrating that the model accepts a real `FoundationBatch`.
- Later used by: Member 3 as confirmation that training can consume the data adapter output.

- [ ] **Step 1: Write the integration test using the public collator**

Do not import private helpers from `tests/foundation_input`, because the repository's test directory is not a Python package. Do not point a test at multi-gigabyte checked-in data. Instead, construct two `EncodedSample` values with `TokenRecord` entries, call the public `foundation_collate(samples)`, and pass the resulting real PyTorch `FoundationBatch` to `FoundationModel.encode`.

Use this exact construction pattern:

```python
from src.foundation_input.dataset import foundation_collate
from src.foundation_input.types import EncodedSample, KeyValueToken, TokenRecord

records = (
    TokenRecord(
        token=KeyValueToken("profile.city", "Hanoi", "categorical", "tabular", None),
        value_id=7, key_id=3, modality_id=1, source_id=2, position_id=0, time_gap_id=0,
    ),
    TokenRecord(
        token=KeyValueToken("event.event_type", "view", "categorical", "sequence", None, "e-1"),
        value_id=9, key_id=4, modality_id=1, source_id=1, position_id=1, time_gap_id=0,
    ),
)
batch = foundation_collate([
    EncodedSample("sample-1", "fixture-v1", "user-1", records, user_id_value_id=6),
    EncodedSample("sample-2", "fixture-v1", "user-2", records, user_id_value_id=8),
])
```

Instantiate the model with `model_config()` from `conftest.py`; its table sizes already cover these IDs.

The assertions must be:

```python
assert output.hidden_states.shape[0] == 2
assert output.hidden_states.shape[1] == batch.input_ids.shape[1]
assert output.long_term.shape == output.short_term.shape == (2, config.d_model)
assert torch.isfinite(output.long_term).all()
assert torch.isfinite(output.short_term).all()
```

- [ ] **Step 2: Run the smoke test**

Run:

```powershell
python -m pytest tests/models/test_encoder.py -k foundation_input_smoke -v
```

Expected: PASS on CPU without downloading a model or accessing the network.

- [ ] **Step 3: Run repository verification and record the known unrelated failure accurately**

Run:

```powershell
python -m pytest tests/models -q
python -m pytest -q
python -m ruff check src/models tests/models
```

Expected: `tests/models` and ruff PASS. At the time this plan was written, repository-wide pytest stops during collection at the pre-existing `tests/test_imports.py` relative-import error. Do not change that file in this branch; report it separately in the PR together with the passing Member 2 command.

- [ ] **Step 4: Commit the integration smoke test**

```powershell
git add tests/models/test_encoder.py
git commit -m "test: smoke foundation encoder with input adapter batch"
```

### Task 8: Handoff and PR verification

**Files:**
- Modify: no source file; PR description and review checklist only.

**Interfaces:**
- Member 3 consumes `FoundationModelConfig`, `FoundationModel.forward`, `FoundationModel.encode`, and `FoundationModel.embed_users`.
- Members 3 and 4 consume only `embed_users()` from downstream code.

- [ ] **Step 1: Check ownership and public surface**

Run:

```powershell
git diff --name-only main...HEAD
```

Expected: only `src/models/adapters.py`, `src/models/encoder.py`, `src/models/pooling.py`, and `tests/models/**` appear.

- [ ] **Step 2: State the exact handoff in the PR**

Include this API example:

```python
model = FoundationModel(model_config)
pretrain_output = model(batch, attention_mode="bidirectional")
next_action_output = model(batch, attention_mode="next_action")
embeddings = model.embed_users(batch)
assert set(embeddings) == {"long_term", "short_term"}
```

State that `long_term` is a hashed-bucket User-ID embedding and `short_term` is event-only pooled Transformer output. State that downstream must not read `hidden_states`, access adapters, or use hooks.

- [ ] **Step 3: Include the upstream contract status verbatim**

```text
Production training is gated on Member 1's stable OOV user IDs, next-action eligibility,
independent event-group IDs, recent-event retention, published vocabulary cardinalities,
one documented modality spelling, and the selected FoundationBatch v2 numeric tensors.
This encoder uses the numeric MLP route from normalized floats plus the missing mask.
Geo remains unsupported because the current pretrain manifests do not supply meaningful
latitude/longitude fields.
```

- [ ] **Step 4: Open the PR only after the checks pass**

Run:

```powershell
python -m pytest tests/models -q
python -m ruff check src/models tests/models
git status --short
```

Expected: tests/lint PASS and no generated data, checkpoints, test cache, or experiment artifacts staged for commit.

## Spec Coverage Review

- `FoundationModelConfig`, modality adapters, User-ID/key/source embeddings, sequence-only positional/time features: Tasks 1–2.
- Bidirectional and next-action attention, including PyTorch mask convention and future-event isolation: Task 4 and Task 6.
- `FoundationModel`, `FoundationOutput`, `encode`, `embed_users`, `long_term`, `short_term`, and zero-event fallback: Tasks 3 and 5.
- Required CPU shape/finite/routing/padding/tabular-permutation/future-event/zero-event tests: Tasks 2–7.
- Strict ownership and no changes to data, training, downstream, baseline, evaluator, or configs: Global Constraints and Task 8.
- The selected raw-numeric architecture is covered by the v2 handoff; geo remains explicitly conditional on a future Data contract and is not part of this implementation.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-10-11-foundation-encoder.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?
