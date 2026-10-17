from src.evaluation.policy import METRIC_POLICY, score_semantics_for_task


def test_metric_policy_identity_is_complete() -> None:
    assert METRIC_POLICY.identity() == {
        "evaluator_version": "green_sm_evaluator_v1",
        "metric_policy_version": "green_sm_metric_policy_v1",
        "tie_break_policy": "score_desc__candidate_id_asc",
        "timestamp_policy": "parse_to_utc_instant",
    }


def test_score_semantics_are_fixed_by_downstream_task() -> None:
    assert score_semantics_for_task("food_recommendation") == "ranking"
    assert score_semantics_for_task("ad_optimization") == "probability"
