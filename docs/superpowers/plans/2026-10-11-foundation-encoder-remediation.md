# Foundation Encoder Contract Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Member 2 encoder reject inconsistent vocabulary metadata and prove that categorical, numeric, and text modality IDs select distinct projections.

**Architecture:** The supported input remains ID/bucket `FoundationBatch` v1. `FoundationModelConfig` receives actual vocabulary cardinalities and `source_to_id["sequence"]`; `AdapterRegistry` retains its per-modality projection table and exposes that routing through deterministic tests.

**Tech Stack:** Python, PyTorch CPU, pytest, ruff.

**Spec:** `docs/parallel_vibe_implementation_plan.md`, `docs/foundation_input_contract.md`, `src/foundation_input/vocab.py`, `docs/superpowers/plans/2026-10-11-foundation-encoder.md`.

## Global Constraints

- Modify only `src/models/encoder.py`, `src/models/adapters.py`, `tests/models/conftest.py`, `tests/models/test_encoder.py`, `tests/models/test_adapters.py`, and the Member 2 plan document.
- Do not modify Member 1's input adapter or vocabulary code.
- Do not add raw numeric MLP or geo routing. Those require a later Data + Member 1 FoundationBatch extension.
- Preserve `FoundationModel.forward`, `encode`, and `embed_users` signatures.
- Member 3 must pass cardinalities and the sequence source ID derived from the chosen vocabulary artifact; no default may guess them.

---

### Task 1: Validate all model input metadata before creating embeddings

**Files:**
- Modify: `src/models/encoder.py`
- Modify: `tests/models/test_encoder.py`
- Modify: `tests/models/conftest.py`

**Interfaces:**
- Consumes: `value_vocab_size`, `key_vocab_size`, `modality_vocab_size`, `source_vocab_size`, `time_gap_vocab_size`, and `sequence_source_id` supplied by Member 3 from Member 1 vocabulary metadata.
- Produces: a `FoundationModelConfig` that cannot represent an invalid source/embedding layout.

- [ ] **Step 1: Add failing validation tests**

```python
def test_config_rejects_zero_attention_heads(model_config) -> None:
    with pytest.raises(ValueError, match="num_heads"):
        model_config(num_heads=0)


def test_config_rejects_sequence_source_outside_source_vocab(model_config) -> None:
    with pytest.raises(ValueError, match="sequence_source_id"):
        model_config(source_vocab_size=2, sequence_source_id=2)


def test_config_requires_real_input_cardinalities() -> None:
    with pytest.raises(ValueError, match="value_vocab_size"):
        FoundationModelConfig()
```

- [ ] **Step 2: Run the tests and confirm current failures**

```powershell
python -m pytest tests/models/test_encoder.py -k config -v
```

Expected: zero heads currently raises `ZeroDivisionError`; invalid sequence source and no-cardinality config currently do not raise.

- [ ] **Step 3: Change the data-derived defaults to explicit metadata requirements**

Use these dataclass fields:

```python
value_vocab_size: int | None = None
key_vocab_size: int | None = None
modality_vocab_size: int | None = None
source_vocab_size: int | None = None
time_gap_vocab_size: int | None = None
sequence_source_id: int | None = None
```

At the beginning of `__post_init__`, prevent modulo by zero:

```python
if self.num_heads <= 0:
    raise ValueError(f"num_heads must be positive, got {self.num_heads}")
if self.d_model <= 0 or self.d_model % self.num_heads != 0:
    raise ValueError(
        f"d_model={self.d_model} must be positive and divisible by "
        f"num_heads={self.num_heads}"
    )
```

After attention-mode validation, require metadata and check its range:

```python
metadata = (
    "value_vocab_size", "key_vocab_size", "modality_vocab_size",
    "source_vocab_size", "time_gap_vocab_size", "sequence_source_id",
)
for name in metadata:
    if getattr(self, name) is None:
        raise ValueError(f"{name} must be supplied from vocabulary metadata")
for name in metadata[:-1]:
    if getattr(self, name) <= 0:
        raise ValueError(f"{name} must be positive")
if not 0 <= self.sequence_source_id < self.source_vocab_size:
    raise ValueError(
        "sequence_source_id must be inside source vocabulary: "
        f"got sequence_source_id={self.sequence_source_id}, "
        f"source_vocab_size={self.source_vocab_size}"
    )
```

- [ ] **Step 4: Align fixtures to actual source IDs**

`fit_vocabulary` sorts source names, therefore the fixture must use `sequence=1` and `tabular=2`. In `conftest.py`, replace all event `source_id=2` with `1`, all profile `source_id=1` with `2`, and set `source_vocab_size=3, sequence_source_id=1` in `model_config()`.

- [ ] **Step 5: Verify and commit**

```powershell
python -m pytest tests/models/test_encoder.py -k config -q
python -m ruff check src/models/encoder.py tests/models/conftest.py tests/models/test_encoder.py
git add src/models/encoder.py tests/models/conftest.py tests/models/test_encoder.py
git commit -m "fix: validate foundation encoder input metadata"
```

Expected: config tests and lint pass.

### Task 2: Prove deterministic routing through each modality projection

**Files:**
- Modify: `src/models/adapters.py`
- Modify: `tests/models/test_adapters.py`

**Interfaces:**
- Consumes: `AdapterRegistry.modality_projections`, indexed by `batch.modality_ids`.
- Produces: a deterministic output route for each modality ID, tested with fixture IDs categorical `1`, numeric `2`, text `3`.

- [ ] **Step 1: Add a failing routing test and local batch factory**

Create `_three_modality_batch()` in `test_adapters.py`. It must return one row `[CLS, categorical, numeric, text, SEP]` whose three field tokens use the same `input_id=9`, `key_id=3`, `source_id=2`, `position_id=0`, `time_gap_id=0`, and modality IDs `1`, `2`, `3`.

```python
def test_registry_routes_categorical_numeric_and_text_to_distinct_projections(model_config) -> None:
    config = model_config(modality_vocab_size=4)
    registry = AdapterRegistry(config).eval()
    with torch.no_grad():
        for embedding in (
            registry.value_embedding, registry.pair_embedding, registry.key_embedding,
            registry.modality_embedding, registry.source_embedding,
            registry.position_embedding, registry.time_gap_embedding,
        ):
            embedding.weight.zero_()
        registry.value_embedding.weight[9].fill_(1.0)
        for projection in registry.modality_projections:
            projection.weight.zero_()
        registry.modality_projections[1].weight.copy_(torch.eye(config.d_model))
        registry.modality_projections[2].weight.copy_(2.0 * torch.eye(config.d_model))
        registry.modality_projections[3].weight.copy_(3.0 * torch.eye(config.d_model))

    output = registry(_three_modality_batch())

    torch.testing.assert_close(output[0, 1], torch.ones(config.d_model))
    torch.testing.assert_close(output[0, 2], torch.full((config.d_model,), 2.0))
    torch.testing.assert_close(output[0, 3], torch.full((config.d_model,), 3.0))
```

- [ ] **Step 2: Run the test and confirm it fails because the helper is absent**

```powershell
python -m pytest tests/models/test_adapters.py::test_registry_routes_categorical_numeric_and_text_to_distinct_projections -v
```

Expected: FAIL with `_three_modality_batch` not defined.

- [ ] **Step 3: Make the routing path explicit**

Extract the existing projection loop in `AdapterRegistry`:

```python
def _project_by_modality(
    self, features: torch.Tensor, modality_ids: torch.LongTensor
) -> torch.Tensor:
    projected = torch.zeros_like(features)
    for modality_id, projection in enumerate(self.modality_projections):
        selected = modality_ids.eq(modality_id).unsqueeze(-1)
        projected = torch.where(selected, projection(features), projected)
    return projected
```

Replace the inline loop in `forward` with `features = self._project_by_modality(features, modality_ids)`. Add `_three_modality_batch()` using `FoundationBatch` and run the deterministic assertions.

- [ ] **Step 4: Verify and commit**

```powershell
python -m pytest tests/models/test_adapters.py -q
python -m ruff check src/models/adapters.py tests/models/test_adapters.py
git add src/models/adapters.py tests/models/test_adapters.py
git commit -m "test: prove modality projection routing"
```

Expected: all adapter tests and lint pass.

### Task 3: Document the exact v1 production boundary

**Files:**
- Modify: `src/models/adapters.py`
- Modify: `docs/superpowers/plans/2026-10-11-foundation-encoder.md`

**Interfaces:**
- Consumes: the current ID-only `FoundationBatch` v1.
- Produces: accurate documentation for Member 3 and reviewers.

- [ ] **Step 1: Update the AdapterRegistry docstring**

Add this exact statement:

```text
Consumes FoundationBatch v1 ID/bucket features for categorical, numeric, text, and missing values.
It does not receive raw numeric floats or geo tensors; continuous numeric MLP and geo routing require
the separately versioned FoundationBatch extension owned by the input/data teams.
```

- [ ] **Step 2: Update the Member 2 handoff plan**

Under “Upstream Input-Contract Gates”, add:

```text
`sequence_source_id` is not a universal constant: Member 3 must take it from the selected artifact's `source_to_id["sequence"]` mapping, alongside returned cardinalities.
```

- [ ] **Step 3: Run branch verification and commit**

```powershell
python -m pytest tests/models -q
python -m ruff check src/models tests/models
python -m pytest -q
git add src/models/adapters.py docs/superpowers/plans/2026-10-11-foundation-encoder.md
git commit -m "docs: clarify foundation adapter v1 scope"
```

Expected: Member 2 tests and ruff pass. Full repository pytest may still stop at the existing `tests/test_imports.py` relative-import collection error; do not change that test in this branch.

## Spec Coverage Review

- Zero heads, missing cardinalities, invalid sequence source IDs, and source-ID fixture mismatch: Task 1.
- Categorical/numeric/text projection routing: Task 2.
- Clear distinction between bucket-ID MVP and raw numeric/geo architecture: Task 3.
- No input adapter, data, training, downstream, baseline, or evaluation changes: Global Constraints.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-10-11-foundation-encoder-remediation.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using `executing-plans`, batch execution with checkpoints.

Which approach?
