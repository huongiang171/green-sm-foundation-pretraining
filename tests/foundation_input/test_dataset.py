from __future__ import annotations

import numpy as np

from src.foundation_input.dataset import (
    DatasetConfig,
    FoundationPretrainDataset,
    _truncate_records,
    foundation_collate,
)
from src.foundation_input.loader import iter_normalized_samples
from src.foundation_input.tokenizer import TokenizerConfig, semantic_tokens
from src.foundation_input.types import EncodedSample, KeyValueToken, TokenRecord
from src.foundation_input.vocab import VocabularyConfig, fit_vocabulary

from .test_manifest import _write_dataset


class FakeTensorBackend:
    long = np.int64
    bool = np.bool_

    @staticmethod
    def tensor(values, *, dtype):
        return np.asarray(values, dtype=dtype)


def _record(value_id: int, *, source: str, group: str | None, position: int) -> TokenRecord:
    return TokenRecord(
        token=KeyValueToken(
            key="profile.x" if source == "tabular" else "event_type",
            value=value_id,
            modality="numeric" if source == "tabular" else "categorical",
            source=source,
            timestamp=None,
            event_group_id=group,
        ),
        value_id=value_id,
        key_id=value_id + 10,
        modality_id=1,
        source_id=1 if source == "tabular" else 2,
        position_id=position,
        time_gap_id=position,
    )


def test_collator_builds_layout_and_padding_without_changing_real_tokens() -> None:
    first = EncodedSample(
        sample_id="s1",
        dataset_name="omnibehavior_v2",
        user_id="u1",
        records=(
            _record(11, source="tabular", group=None, position=0),
            _record(21, source="sequence", group="e1", position=1),
        ),
    )
    second = EncodedSample(
        sample_id="s2",
        dataset_name="omnibehavior_v2",
        user_id="u2",
        records=(
            _record(12, source="tabular", group=None, position=0),
            _record(22, source="sequence", group="e2", position=1),
            _record(23, source="sequence", group="e3", position=2),
        ),
    )

    batch = foundation_collate([first, second], tensor_backend=FakeTensorBackend())

    assert batch.input_ids.shape == (2, 5)
    assert batch.input_ids[0, :4].tolist() == [1, 11, 2, 21]
    assert batch.input_ids[0, 4:].tolist() == [0]
    assert batch.input_ids[1, :5].tolist() == [1, 12, 2, 22, 23]
    assert batch.attention_mask.dtype == np.bool_
    assert batch.attention_mask[0].tolist() == [True, True, True, True, False]
    assert batch.targets["event_group_ids"][0].tolist() == [0, 0, 0, 1, 0]


def test_collator_keeps_tabular_before_sequence_tokens() -> None:
    sample = EncodedSample(
        sample_id="s1",
        dataset_name="omnibehavior_v2",
        user_id="u1",
        records=(
            _record(11, source="tabular", group=None, position=0),
            _record(12, source="tabular", group=None, position=0),
            _record(21, source="sequence", group="e1", position=1),
        ),
    )
    batch = foundation_collate([sample], tensor_backend=FakeTensorBackend())

    assert batch.input_ids[0].tolist() == [1, 11, 12, 2, 21]


def test_foundation_dataset_streams_encoded_samples(tmp_path) -> None:
    _write_dataset(tmp_path)
    canonical = next(iter_normalized_samples(tmp_path, split="train", validate=True))
    artifacts = fit_vocabulary(
        semantic_tokens(canonical, TokenizerConfig()),
        VocabularyConfig(dataset_name=canonical.dataset_name),
    )
    dataset = FoundationPretrainDataset(
        str(tmp_path),
        "train",
        artifacts,
        dataset_config=DatasetConfig(max_event_count=30, max_tokens=16),
    )

    encoded = dataset[0]
    assert encoded.sample_id == canonical.sample_id
    assert encoded.records


def test_next_action_target_is_excluded_from_input() -> None:
    first = _record(21, source="sequence", group="e1", position=1)
    second = _record(22, source="sequence", group="e2", position=2)
    sample = EncodedSample(
        sample_id="s1",
        dataset_name="omnibehavior_v2",
        user_id="u1",
        records=(first, second),
        next_action_records=(second,),
    )

    batch = foundation_collate([sample], tensor_backend=FakeTensorBackend())

    assert batch.input_ids[0].tolist() == [1, 2, 21]
    assert batch.targets["event_group_ids"][0].tolist() == [0, 0, 1]
    assert batch.targets["next_action_group_ids"].tolist() == [2]
    assert batch.targets["next_action_type_ids"].tolist() == [22]


def test_foundation_dataset_does_not_keep_unbounded_sample_cache(tmp_path) -> None:
    _write_dataset(tmp_path)
    canonical = next(iter_normalized_samples(tmp_path, split="train", validate=True))
    artifacts = fit_vocabulary(
        semantic_tokens(canonical, TokenizerConfig()),
        VocabularyConfig(dataset_name=canonical.dataset_name),
    )
    dataset = FoundationPretrainDataset(str(tmp_path), "train", artifacts)

    list(dataset)

    assert not hasattr(dataset, "_cache")


def test_max_tokens_reserves_cls_and_sep() -> None:
    sample = EncodedSample(
        sample_id="s1",
        dataset_name="omnibehavior_v2",
        user_id="u1",
        records=(
            _record(11, source="tabular", group=None, position=0),
            _record(21, source="sequence", group="e1", position=1),
        ),
    )
    retained = _truncate_records(list(sample.records), DatasetConfig(max_tokens=4))
    assert [record.value_id for record in retained] == [11, 21]
