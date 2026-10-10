from __future__ import annotations

from pathlib import Path

from src.foundation_input.types import SPECIAL_TOKEN_IDS, KeyValueToken
from src.foundation_input.vocab import VocabularyConfig, fit_vocabulary


def _token(key: str, value, modality: str = "categorical") -> KeyValueToken:
    return KeyValueToken(
        key=key,
        value=value,
        modality=modality,
        source="tabular",
        timestamp=None,
    )


def test_fit_vocabulary_is_train_only_and_maps_oov_to_unk() -> None:
    artifacts = fit_vocabulary(
        [
            _token("city", "hanoi"),
            _token("city", "danang"),
        ],
        VocabularyConfig(dataset_name="omnibehavior_v2"),
    )
    before = artifacts.value_to_id.copy()

    assert artifacts.encode_value(_token("city", "hanoi")) != SPECIAL_TOKEN_IDS["UNK"]
    assert artifacts.encode_value(_token("city", "unknown-city")) == SPECIAL_TOKEN_IDS["UNK"]
    assert artifacts.value_to_id == before


def test_numeric_values_use_train_fitted_boundaries() -> None:
    artifacts = fit_vocabulary(
        [_token("event_count", value, "numeric") for value in (0, 2, 4, 8)],
        VocabularyConfig(dataset_name="meituan_trd_v2", numeric_bucket_count=4),
    )

    assert artifacts.numeric_boundaries["meituan_trd_v2:tabular:numeric:event_count"]
    assert artifacts.encode_value(_token("event_count", 3, "numeric")) != SPECIAL_TOKEN_IDS["UNK"]


def test_text_values_use_deterministic_hash_buckets() -> None:
    artifacts = fit_vocabulary(
        [], VocabularyConfig(dataset_name="omnibehavior_v2", text_hash_buckets=32)
    )
    first = artifacts.encode_value(_token("caption", "same text", "text"))
    second = artifacts.encode_value(_token("caption", "same text", "text"))
    assert first == second
    assert first >= SPECIAL_TOKEN_IDS["MASK"] + 2


def test_vocabulary_metadata_round_trips(tmp_path: Path) -> None:
    artifacts = fit_vocabulary(
        [_token("city", "hanoi")], VocabularyConfig(dataset_name="omnibehavior_v2")
    )
    path = tmp_path / "vocabulary.json"
    artifacts.save(path)
    loaded = artifacts.load(path)

    assert loaded.dataset_name == artifacts.dataset_name
    assert loaded.encode_value(_token("city", "hanoi")) == artifacts.encode_value(
        _token("city", "hanoi")
    )
