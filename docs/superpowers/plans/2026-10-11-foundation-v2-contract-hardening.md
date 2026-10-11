# Foundation v2 Contract Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Prevent partial event groups, enforce valid next-action loss rows, and route continuous numeric values into the foundation encoder.

**Architecture:** Member 1 owns atomic truncation and the versioned input artifact. Member 2 routes raw numeric values through an MLP. Member 3 has not started implementation; their masked-loss task is a frozen handoff to execute when the pretraining branch starts. Production pretraining is blocked until all three contracts pass.

**Tech Stack:** Python 3.13, PyTorch CPU, pytest, Ruff, dataclasses, JSON metadata.

**Spec:** docs/foundation_input_contract.md, docs/parallel_vibe_implementation_plan.md, docs/superpowers/plans/2026-10-11-foundation-input-contract-extension.md, docs/superpowers/plans/2026-10-11-foundation-encoder.md.

## Global Constraints

- Member 1 changes only src/foundation_input/**, its tests, and input-contract docs.
- Member 2 changes only src/models/**, tests/models/**, and encoder docs.
- Member 3 has no current code task; when their pretraining branch starts, they change only objectives/training code and tests.
- The selected next-action policy is mask-based: invalid rows retain other objectives but have target 0 and valid mask False; their next-action loss is zero.
- Truncation never returns part of an event group. If the newest group cannot fit, raise an error containing sample_id.
- numeric_values is finite float32 [B,L]; numeric_missing_mask is bool [B,L]; numeric modality ID and sequence source ID come from the selected artifact.
- Do not commit vocabulary/checkpoint artifacts, generated data, pytest temporary folders, or .ai-log.

---

### Task 1: Atomic event-group truncation (Member 1)

**Files:**

- Modify: src/foundation_input/dataset.py:24-101
- Modify: tests/foundation_input/test_dataset.py
- Modify: docs/foundation_input_contract.md:56-58

**Interfaces:**

- Consumes: list[TokenRecord], DatasetConfig, and sample.sample_id.
- Produces: complete sequence groups only, or ValueError naming the sample.

- [ ] **Step 1: Write failing tests**

Add these exact helpers to `test_dataset.py` (the file already has
`_canonical_sample` and imports `replace`):

    def _artifacts_for(sample: CanonicalSample):
        return fit_vocabulary(
            semantic_tokens(sample, TokenizerConfig()),
            VocabularyConfig(dataset_name=sample.dataset_name),
        )

    def _too_large_group_sample() -> CanonicalSample:
        base = _canonical_sample()
        event = replace(base.events[0], action={"type": "click", "click_cnt": 1})
        return replace(base, events=(event,))

    def test_encode_sample_rejects_group_that_cannot_fit_without_splitting() -> None:
        sample = _too_large_group_sample()

        with pytest.raises(ValueError, match=rf"sample_id={sample.sample_id}.*complete event group"):
            encode_sample(sample, _artifacts_for(sample), TokenizerConfig(), DatasetConfig(max_tokens=4))

    def test_truncation_does_not_return_a_group_prefix() -> None:
        records = [
            _record(11, source="tabular", group=None, position=0),
            _record(21, source="sequence", group="recent", position=1),
            _record(22, source="sequence", group="recent", position=1),
            _record(23, source="sequence", group="recent", position=1),
        ]

        result = _truncate_records(records, DatasetConfig(max_tokens=4))

        assert result.rejected_recent_group
        assert all(record.token.source != "sequence" for record in result.records)

- [ ] **Step 2: Confirm the test fails**

Run:

    python -m pytest tests/foundation_input/test_dataset.py -k "cannot_fit or group_prefix" -v --basetemp .pytest-contract-hardening

Expected: current helper returns the first two recent tokens; encode_sample does not raise.

- [ ] **Step 3: Implement status-aware truncation**

Add this dataclass after DatasetConfig:

    @dataclass(frozen=True)
    class TruncationResult:
        records: tuple[TokenRecord, ...]
        rejected_recent_group: bool

Change _truncate_records to return TruncationResult. Replace the branch that appends group[:remaining_event_budget] with:

        elif not selected_groups and remaining_event_budget > 0:
            return TruncationResult(
                records=tuple(tabular[:record_budget]),
                rejected_recent_group=True,
            )

At normal completion return:

    return TruncationResult(records=tuple(selected), rejected_recent_group=False)

In encode_sample use:

    truncation = _truncate_records(records, dataset_config)
    if truncation.rejected_recent_group:
        raise ValueError(
            f"sample_id={sample.sample_id}: most recent complete event group "
            "does not fit within max_tokens; refusing to split the group"
        )
    selected = truncation.records

Update every existing direct helper test to assert result.records.

- [ ] **Step 4: Verify and commit**

    python -m pytest tests/foundation_input/test_dataset.py -q --basetemp .pytest-contract-hardening
    python -m ruff check src/foundation_input/dataset.py tests/foundation_input/test_dataset.py
    git add src/foundation_input/dataset.py tests/foundation_input/test_dataset.py docs/foundation_input_contract.md
    git commit -m "fix: preserve complete event groups during truncation"

Document: a non-fitting newest group is rejected, never split.

### Task 2: Freeze the mask-based next-action handoff (Member 1 now; Member 3 later)

**Files:**

- Modify: tests/foundation_input/test_dataset.py
- Modify: src/models/objectives.py
- Create: tests/models/test_objectives.py
- Modify: docs/foundation_input_contract.md
- Modify: docs/parallel_vibe_implementation_plan.md:240-247

**Interfaces:**

- Consumes: logits FloatTensor[B,C], next_action_type_ids LongTensor[B], next_action_valid_mask BoolTensor[B].
- Produces: masked_next_action_cross_entropy(logits, target_ids, valid_mask) -> tuple[Tensor, int].

**Scheduling rule:** Member 1 performs only Step 1 and Step 4 now, to prove and document the emitted mask. Do not create or modify `src/models/objectives.py` until Member 3 starts their pretraining branch. At that time, Member 3 executes Steps 2 and 3 exactly as written before implementing the trainer.

- [ ] **Step 1: Add input-side validity test**

Add these helpers to `test_dataset.py` before the tests below:

    def _two_complete_event_sample() -> CanonicalSample:
        base = _canonical_sample()
        second = replace(
            base.events[0],
            event_id="user-001:1:0",
            source_record_id="user-001:1",
            source_record_index=1,
        )
        return replace(base, events=(base.events[0], second))

    def _one_complete_event_sample() -> CanonicalSample:
        return _canonical_sample()

    @pytest.mark.parametrize(
        "replacement", [{"history_order_known": False}, {"history_group_truncated": True}]
    )
    def test_invalid_next_action_is_masked_not_targeted(replacement) -> None:
        sample = replace(_two_complete_event_sample(), **replacement)
        encoded = encode_sample(
            sample, _artifacts_for(sample), TokenizerConfig(), DatasetConfig(next_action_prediction=True)
        )
        batch = foundation_collate([encoded], tensor_backend=FakeTensorBackend())

        assert encoded.next_action_records == ()
        assert batch.targets["next_action_type_ids"].tolist() == [0]
        assert batch.targets["next_action_valid_mask"].tolist() == [False]

- [ ] **Step 2: Add failing loss tests**

    def test_next_action_loss_excludes_invalid_rows() -> None:
        logits = torch.tensor([[0.0, 4.0], [0.0, -4.0]], requires_grad=True)
        loss, count = masked_next_action_cross_entropy(
            logits, torch.tensor([1, 1]), torch.tensor([True, False])
        )

        expected = torch.nn.functional.cross_entropy(logits[:1], torch.tensor([1]))
        torch.testing.assert_close(loss, expected)
        assert count == 1

    def test_next_action_loss_is_differentiable_zero_without_valid_rows() -> None:
        logits = torch.randn(2, 4, requires_grad=True)
        loss, count = masked_next_action_cross_entropy(
            logits, torch.zeros(2, dtype=torch.long), torch.tensor([False, False])
        )
        loss.backward()

        assert loss.item() == 0.0
        assert count == 0
        assert torch.equal(logits.grad, torch.zeros_like(logits))

- [ ] **Step 3: Implement masked loss**

    import torch
    from torch.nn import functional as F

    def masked_next_action_cross_entropy(
        logits: torch.Tensor, target_ids: torch.LongTensor, valid_mask: torch.BoolTensor
    ) -> tuple[torch.Tensor, int]:
        if logits.ndim != 2:
            raise ValueError(f"logits must have shape [B, C], got {tuple(logits.shape)}")
        if target_ids.shape != (logits.shape[0],) or target_ids.dtype != torch.long:
            raise ValueError("target_ids must be torch.long with shape [B]")
        if valid_mask.shape != (logits.shape[0],) or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be torch.bool with shape [B]")
        if torch.any(target_ids[valid_mask] < 0).item() or torch.any(
            target_ids[valid_mask] >= logits.shape[1]
        ).item():
            raise ValueError("valid next-action target ID is outside logits classes")
        count = int(valid_mask.sum().item())
        if count == 0:
            return logits.sum() * 0.0, 0
        return F.cross_entropy(logits[valid_mask], target_ids[valid_mask]), count

- [ ] **Step 4: Member 1 document and commit now**

State that target 0 is a sentinel and Member 3 must pass the valid mask. Mark the old raise-on-invalid requirement as superseded; strict rejection belongs only to Task 1.

    python -m pytest tests/foundation_input/test_dataset.py -q --basetemp .pytest-contract-hardening
    python -m ruff check src/foundation_input tests/foundation_input
    git add tests/foundation_input/test_dataset.py docs/foundation_input_contract.md docs/parallel_vibe_implementation_plan.md
    git commit -m "docs: define masked next-action contract"

When Member 3 starts, they add `src/models/objectives.py` and
`tests/models/test_objectives.py` from Steps 2-3 in their own commit:

    python -m pytest tests/models/test_objectives.py -q --basetemp .pytest-contract-hardening
    python -m ruff check src/models/objectives.py tests/models/test_objectives.py
    git add src/models/objectives.py tests/models/test_objectives.py
    git commit -m "feat: mask invalid next-action loss rows"

### Task 3: Persist and validate the tokenizer contract (Member 1)

**Files:**

- Modify: src/foundation_input/vocab.py
- Modify: src/foundation_input/tokenizer.py:266-328
- Modify: src/foundation_input/dataset.py:115-127
- Modify: tests/foundation_input/test_vocab.py
- Modify: tests/foundation_input/test_tokenizer.py

**Interfaces:**

- Produces: VocabularyArtifacts.tokenizer_contract: dict[str, object], persisted to JSON.
- Consumes: transform TokenizerConfig; semantic mismatch raises before token IDs exist.

- [ ] **Step 1: Add failing tests**

Add the following helpers to `test_tokenizer.py`; that file already defines
`_sample` and `_event`:

    def _vocabulary_config() -> VocabularyConfig:
        return VocabularyConfig(dataset_name="omnibehavior_v2")

    def _score_sample(value: float | None) -> CanonicalSample:
        return _sample({"score": value}, (_event(),))

    def test_artifact_rejects_mismatched_time_gap_boundaries() -> None:
        config = TokenizerConfig(time_gap_boundaries_seconds=(60.0, 300.0))
        artifacts = fit_vocabulary_from_samples([_score_sample(1.0)], _vocabulary_config(), config)

        with pytest.raises(ValueError, match="TokenizerConfig"):
            tokenize_sample(_score_sample(1.0), artifacts, TokenizerConfig(time_gap_boundaries_seconds=(60.0,)))

    def test_artifact_rejects_mismatched_numeric_registry() -> None:
        config = TokenizerConfig(numeric_fields=frozenset({("tabular", "profile.score")}))
        artifacts = fit_vocabulary_from_samples([_score_sample(1.0)], _vocabulary_config(), config)

        with pytest.raises(ValueError, match="TokenizerConfig"):
            tokenize_sample(_score_sample(None), artifacts, TokenizerConfig(numeric_fields=frozenset()))

    def test_vocabulary_config_policy_is_not_silently_overwritten() -> None:
        artifacts = fit_vocabulary_from_samples(
            [_score_sample(1.0)],
            replace(_vocabulary_config(), numeric_transform_policy={("tabular", "profile.score"): "zscore"}),
            TokenizerConfig(numeric_fields=frozenset({("tabular", "profile.score")})),
        )
        assert artifacts.numeric_transform_policy[
            "omnibehavior_v2:tabular:numeric:profile.score"
        ] == "zscore"

- [ ] **Step 2: Implement metadata and validation**

Add tokenizer_contract: dict[str, object] = field(default_factory=dict) to VocabularyArtifacts. Build and persist this exact mapping in fit_vocabulary_from_samples:

    def _tokenizer_contract(config: TokenizerConfig) -> dict[str, object]:
        return {
            "context_allowlist": {
                key: list(config.context_allowlist[key]) for key in sorted(config.context_allowlist)
            },
            "include_action_content": config.include_action_content,
            "max_text_chars": config.max_text_chars,
            "numeric_fields": [list(field) for field in sorted(config.numeric_fields)],
            "time_gap_boundaries_seconds": list(config.time_gap_boundaries_seconds),
        }

Add validate_tokenizer_config that compares artifact metadata with _tokenizer_contract(config), then call it first in tokenize_sample. When FoundationPretrainDataset receives no config, reconstruct TokenizerConfig from this mapping. If mapping is empty, raise ValueError rather than guessing defaults.

Preserve policy precedence:

    config_policy = config.numeric_transform_policy
    token_policy = token_config.numeric_transform_policy
    if config_policy and token_policy and config_policy != token_policy:
        raise ValueError("VocabularyConfig and TokenizerConfig numeric transform policies differ")
    config = replace(config, numeric_transform_policy=token_policy or config_policy)

- [ ] **Step 3: Verify and commit**

    python -m pytest tests/foundation_input -q --basetemp .pytest-contract-hardening
    python -m ruff check src/foundation_input tests/foundation_input
    git add src/foundation_input/vocab.py src/foundation_input/tokenizer.py src/foundation_input/dataset.py tests/foundation_input/test_vocab.py tests/foundation_input/test_tokenizer.py docs/foundation_input_contract.md
    git commit -m "fix: bind tokenizer transforms to vocabulary artifacts"

### Task 4: Raw numeric MLP routing (Member 2)

**Files:**

- Modify: src/models/encoder.py
- Modify: src/models/adapters.py
- Modify: tests/models/conftest.py
- Modify: tests/models/test_adapters.py
- Modify: tests/models/test_encoder.py

**Interfaces:**

- Consumes: numeric_values, numeric_missing_mask, numeric_modality_id from artifacts.modality_to_id["numeric"].
- Produces: numeric vectors derived from raw value plus missingness, not numeric bucket ID.

- [ ] **Step 1: Write failing behavior tests**

Add numeric_modality_id=3 to the fixture config. Add _numeric_batch(value, missing), where token 1 has modality 3 and token 2 is categorical.

    def test_numeric_value_changes_only_the_numeric_token(model_config) -> None:
        registry = AdapterRegistry(model_config()).eval()
        first = registry(_numeric_batch(1.0, False))
        changed = registry(_numeric_batch(99.0, False))

        assert not torch.equal(first[0, 1], changed[0, 1])
        torch.testing.assert_close(first[0, 2], changed[0, 2])

    def test_numeric_zero_and_missing_are_distinct(model_config) -> None:
        registry = AdapterRegistry(model_config()).eval()

        assert not torch.equal(
            registry(_numeric_batch(0.0, False))[0, 1],
            registry(_numeric_batch(0.0, True))[0, 1],
        )

    def test_adapter_rejects_malformed_numeric_tensors(model_config) -> None:
        batch = _numeric_batch(1.0, False)
        batch.numeric_values = batch.numeric_values[:, :-1]

        with pytest.raises(ValueError, match="numeric_values"):
            AdapterRegistry(model_config())(batch)

- [ ] **Step 2: Confirm current failure**

    python -m pytest tests/models/test_adapters.py -k numeric -v --basetemp .pytest-contract-hardening

Expected: changing raw numeric value or missingness does not alter output; malformed numeric tensor is accepted.

- [ ] **Step 3: Implement metadata, validation, and route**

Add numeric_modality_id: int | None = None to FoundationModelConfig, include it in required metadata, then validate:

    if not 0 < self.numeric_modality_id < self.modality_vocab_size:
        raise ValueError("numeric_modality_id must be a non-padding modality ID inside modality vocabulary")

In AdapterRegistry constructor add:

    self.numeric_mlp = nn.Sequential(
        nn.Linear(2, config.d_model),
        nn.GELU(),
        nn.Linear(config.d_model, config.d_model),
    )

Add a helper that validates numeric_values as float32 [B,L], numeric_missing_mask as bool [B,L], and all numeric values finite. In forward, replace numeric bucket features only:

    is_numeric = modality_ids.eq(self.config.numeric_modality_id)
    id_features = self.value_embedding(input_ids) + self.pair_embedding(pair_ids)
    shared_features = (
        self.key_embedding(key_ids)
        + self.modality_embedding(modality_ids)
        + self.source_embedding(source_ids)
    )
    numeric_features = self.numeric_mlp(
        torch.stack((numeric_values, numeric_missing_mask.to(torch.float32)), dim=-1)
    )
    features = shared_features + torch.where(
        is_numeric.unsqueeze(-1), numeric_features, id_features
    )

Keep modality projection, position/time embeddings, and padding zeroing unchanged.

- [ ] **Step 4: Artifact smoke, verify, commit**

Add an encoder test that supplies numeric_modality_id=artifacts.modality_to_id["numeric"] and all five vocab sizes from artifacts.model_cardinalities(). Assert observed-zero and missing numeric batches produce finite shape-stable output.

    python -m pytest tests/models -q --basetemp .pytest-contract-hardening
    python -m ruff check src/models tests/models
    git add src/models/encoder.py src/models/adapters.py tests/models/conftest.py tests/models/test_adapters.py tests/models/test_encoder.py docs/superpowers/plans/2026-10-11-foundation-encoder.md
    git commit -m "feat: route numeric batch values through encoder"

### Task 5: Pretraining integration gate (Members 1–3)

**Files:**

- Modify: docs/parallel_vibe_implementation_plan.md

**Interfaces:**

- Member 3 receives artifact cardinalities, source_to_id["sequence"], modality_to_id["numeric"], and next_action_valid_mask.
- Member 3 records valid next-action coverage per epoch.

- [ ] **Step 1: Add this exact startup gate**

    Before production pretraining, record the vocabulary artifact path and checksum; pass
    source_to_id["sequence"] and modality_to_id["numeric"] into FoundationModelConfig; and log
    next_action_valid_count / batch_size every epoch. Abort startup on any tokenizer-contract or
    FoundationBatch validation error.

- [ ] **Step 2: Verify combined code**

    python -m pytest tests/foundation_input tests/models -q --basetemp .pytest-contract-hardening
    python -m ruff check src/foundation_input src/models tests/foundation_input tests/models

Then read 20 train samples from each data/pretrain_data/OmniBehavior_data and data/pretrain_data/meituan_data, fit an in-memory artifact, collate them, and assert both numeric tensors match input_ids.shape. Do not save the artifact.

- [ ] **Step 3: Commit the gate only after checks pass**

    git add docs/parallel_vibe_implementation_plan.md
    git commit -m "docs: gate pretraining on foundation v2 contract"

## Spec Coverage Review

- Partial-group prevention: Task 1.
- Explicit mask semantics and loss enforcement: Task 2.
- Tokenizer/artifact consistency and policy precedence: Task 3.
- Continuous numeric MLP plus missing signal: Task 4.
- Artifact-driven startup and real-data verification: Task 5.

## Execution Handoff

Plan complete and saved to docs/superpowers/plans/2026-10-11-foundation-v2-contract-hardening.md. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
