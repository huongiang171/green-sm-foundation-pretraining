from __future__ import annotations

from datetime import datetime, timezone

from src.foundation_input.tokenizer import TokenizerConfig, semantic_tokens, tokenize_sample
from src.foundation_input.types import CanonicalSample, EventRecord
from src.foundation_input.vocab import VocabularyConfig, fit_vocabulary

from .fixtures.samples import sample_mapping


def _sample(profile: dict, events: tuple[EventRecord, ...]) -> CanonicalSample:
    raw = sample_mapping("omnibehavior_v2")
    return CanonicalSample(**{**raw, "profile": profile, "events": events})


def _event(**overrides) -> EventRecord:
    raw = sample_mapping("omnibehavior_v2")["events"][0]
    return EventRecord(**{**raw, **overrides})


def test_profile_column_order_does_not_change_tabular_token_set() -> None:
    event = _event()
    first = _sample({"event_count_30d": 2, "active_days_30d": 1}, (event,))
    second = _sample({"active_days_30d": 1, "event_count_30d": 2}, (event,))

    first_tokens = {
        (token.key, token.value, token.modality)
        for token in semantic_tokens(first, TokenizerConfig())
        if token.source == "tabular"
    }
    second_tokens = {
        (token.key, token.value, token.modality)
        for token in semantic_tokens(second, TokenizerConfig())
        if token.source == "tabular"
    }
    assert first_tokens == second_tokens
    assert ("user_id", "user-001", "categorical") in first_tokens


def test_event_tokens_follow_allowlist_and_keep_source_record_group() -> None:
    event = _event(
        context={"product_id": "p-1", "secret_outcome": "must-not-enter"},
        action={"type": "click", "click_cnt": 1},
    )
    tokens = semantic_tokens(_sample({}, (event,)), TokenizerConfig())
    keys = {token.key for token in tokens}

    assert "context.product_id" in keys
    assert "context.secret_outcome" not in keys
    assert all(
        token.event_group_id == "user-001:0" for token in tokens if token.source == "sequence"
    )


def test_tokenize_assigns_ids_and_time_gap_buckets() -> None:
    first = _event()
    second = _event(
        event_id="user-001:1:0",
        source_record_id="user-001:1",
        source_record_index=1,
        event_time=datetime(2025, 9, 19, 1, tzinfo=timezone.utc),
        eligibility_time=datetime(2025, 9, 19, 1, tzinfo=timezone.utc),
    )
    sample = _sample({"event_count_30d": 2}, (first, second))
    config = TokenizerConfig()
    semantic = semantic_tokens(sample, config)
    artifacts = fit_vocabulary(semantic, VocabularyConfig(dataset_name=sample.dataset_name))
    encoded = tokenize_sample(sample, artifacts, config)

    assert encoded
    assert all(record.value_id != 4 for record in encoded)
    event_gaps = [record.time_gap_id for record in encoded if record.token.source == "sequence"]
    assert event_gaps[0] == 0
    assert max(event_gaps) > 0


def test_interval_values_remain_categorical_strings() -> None:
    event = _event(action={"type": "purchase", "order_price_interval": "[29,36)"})
    tokens = semantic_tokens(_sample({}, (event,)), TokenizerConfig())
    interval = [token for token in tokens if token.key == "action.order_price_interval"]
    assert interval[0].value == "[29,36)"
    assert interval[0].modality == "categorical"
