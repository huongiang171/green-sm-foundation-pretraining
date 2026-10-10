"""Streaming dataset and padded batch collation for foundation pretraining."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .loader import iter_normalized_samples
from .manifest import load_manifest
from .tokenizer import TokenizerConfig, tokenize_sample
from .types import EncodedSample, FoundationBatch, TokenRecord
from .vocab import VocabularyArtifacts


@dataclass(frozen=True)
class DatasetConfig:
    max_event_count: int = 30
    max_tokens: int = 256


def _truncate_records(records: list[TokenRecord], config: DatasetConfig) -> tuple[TokenRecord, ...]:
    tabular = [record for record in records if record.token.source == "tabular"]
    sequence = [record for record in records if record.token.source == "sequence"]
    groups: list[list[TokenRecord]] = []
    for record in sequence:
        if not groups or groups[-1][0].token.event_group_id != record.token.event_group_id:
            groups.append([])
        groups[-1].append(record)
    kept_groups = groups[: config.max_event_count]
    selected = tabular[: config.max_tokens]
    for group in kept_groups:
        if len(selected) + len(group) > config.max_tokens:
            break
        selected.extend(group)
    return tuple(selected)


def encode_sample(
    sample,
    artifacts: VocabularyArtifacts,
    tokenizer_config: TokenizerConfig,
    dataset_config: DatasetConfig,
) -> EncodedSample:
    records = tokenize_sample(sample, artifacts, tokenizer_config)
    user_id_record = next(
        (record for record in records if record.token.key == "user_id"), None
    )
    return EncodedSample(
        sample_id=sample.sample_id,
        dataset_name=sample.dataset_name,
        user_id=sample.user_id,
        records=_truncate_records(records, dataset_config),
        user_id_value_id=user_id_record.value_id if user_id_record is not None else None,
    )


class FoundationPretrainDataset:
    """Lazy Parquet-backed dataset; samples are encoded only when iterated."""

    def __init__(
        self,
        dataset_root: str,
        split: str,
        artifacts: VocabularyArtifacts,
        tokenizer_config: TokenizerConfig | None = None,
        dataset_config: DatasetConfig | None = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.split = split
        self.artifacts = artifacts
        if tokenizer_config is None:
            manifest = load_manifest(dataset_root, validate_shard_files=False)
            allowlist = {
                key: tuple(value)
                for key, value in manifest.config.get("context_allowlist", {}).items()
            }
            tokenizer_config = TokenizerConfig(context_allowlist=allowlist)
        self.tokenizer_config = tokenizer_config
        self.dataset_config = dataset_config or DatasetConfig()
        self._cache: list[EncodedSample] = []

    def __iter__(self) -> Iterator[EncodedSample]:
        for sample in iter_normalized_samples(self.dataset_root, self.split, validate=True):
            encoded = encode_sample(
                sample, self.artifacts, self.tokenizer_config, self.dataset_config
            )
            self._cache.append(encoded)
            yield encoded

    def __getitem__(self, index: int) -> EncodedSample:
        if index < 0:
            raise IndexError("Negative indices are not supported by the streaming dataset")
        while len(self._cache) <= index:
            try:
                next(self._cache_iterator)
            except AttributeError:
                self._cache_iterator = iter(self)
                next(self._cache_iterator)
            except StopIteration as error:
                raise IndexError(index) from error
        return self._cache[index]


def _default_tensor_backend() -> Any:
    try:
        import torch
    except Exception as error:  # torch may be absent or have unavailable native DLLs
        raise RuntimeError(
            "FoundationBatch collation requires a working PyTorch installation"
        ) from error
    return torch


def _stable_user_id(dataset_name: str, user_id: str) -> int:
    digest = hashlib.sha256(f"{dataset_name}\0{user_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def foundation_collate(
    samples: list[EncodedSample], *, tensor_backend: Any | None = None
) -> FoundationBatch:
    if not samples:
        raise ValueError("Cannot collate an empty sample list")
    dataset_names = {sample.dataset_name for sample in samples}
    if len(dataset_names) != 1:
        raise ValueError(
            "Mixed-source batches are disabled; collate one dataset namespace at a time"
        )
    backend = tensor_backend or _default_tensor_backend()

    rows: list[dict[str, list[int | bool]]] = []
    for sample in samples:
        tabular = [record for record in sample.records if record.token.source == "tabular"]
        sequence = [record for record in sample.records if record.token.source == "sequence"]
        input_ids = [1]
        key_ids = [0]
        modality_ids = [0]
        source_ids = [0]
        position_ids = [0]
        time_gap_ids = [0]
        group_ids = [0]
        input_ids.extend(record.value_id for record in tabular)
        key_ids.extend(record.key_id for record in tabular)
        modality_ids.extend(record.modality_id for record in tabular)
        source_ids.extend(record.source_id for record in tabular)
        position_ids.extend(record.position_id for record in tabular)
        time_gap_ids.extend(record.time_gap_id for record in tabular)
        group_ids.extend(0 for _ in tabular)
        input_ids.append(2)
        key_ids.append(0)
        modality_ids.append(0)
        source_ids.append(0)
        position_ids.append(0)
        time_gap_ids.append(0)
        group_ids.append(0)
        for record in sequence:
            input_ids.append(record.value_id)
            key_ids.append(record.key_id)
            modality_ids.append(record.modality_id)
            source_ids.append(record.source_id)
            position_ids.append(record.position_id)
            time_gap_ids.append(record.time_gap_id)
            group_ids.append(record.position_id)
        rows.append(
            {
                "input_ids": input_ids,
                "key_ids": key_ids,
                "modality_ids": modality_ids,
                "source_ids": source_ids,
                "position_ids": position_ids,
                "time_gap_ids": time_gap_ids,
                "group_ids": group_ids,
            }
        )

    max_length = max(len(row["input_ids"]) for row in rows)
    padded: dict[str, list[list[int | bool]]] = {}
    for field in (
        "input_ids",
        "key_ids",
        "modality_ids",
        "source_ids",
        "position_ids",
        "time_gap_ids",
        "group_ids",
    ):
        padded[field] = [row[field] + [0] * (max_length - len(row[field])) for row in rows]
    padded["attention_mask"] = [
        [True] * len(row["input_ids"]) + [False] * (max_length - len(row["input_ids"]))
        for row in rows
    ]
    tensor = backend.tensor
    return FoundationBatch(
        input_ids=tensor(padded["input_ids"], dtype=backend.long),
        key_ids=tensor(padded["key_ids"], dtype=backend.long),
        modality_ids=tensor(padded["modality_ids"], dtype=backend.long),
        source_ids=tensor(padded["source_ids"], dtype=backend.long),
        position_ids=tensor(padded["position_ids"], dtype=backend.long),
        time_gap_ids=tensor(padded["time_gap_ids"], dtype=backend.long),
        attention_mask=tensor(padded["attention_mask"], dtype=backend.bool),
        user_ids=tensor(
            [
                [
                    samples[index].user_id_value_id
                    if samples[index].user_id_value_id is not None
                    else _stable_user_id(samples[index].dataset_name, samples[index].user_id)
                ]
                for index in range(len(samples))
            ],
            dtype=backend.long,
        ).reshape(len(samples)),
        targets={
            "event_group_ids": tensor(padded["group_ids"], dtype=backend.long),
        },
    )
