from src.memory.manager import ModelingBriefStore, ModelResultStore, compute_missing_model_result_fields
from src.tools.model_result import make_finalize_model_result_tool


def _full_preprocessing_rules():
    return [
        {"category": "missing_values", "rule": "none needed", "detail": "d"},
        {"category": "placeholder_values", "rule": "none needed", "detail": "d"},
        {"category": "scaling", "rule": "none needed", "detail": "d"},
        {"category": "encoding", "rule": "none needed", "detail": "d"},
    ]


def _save_brief(db_session, dataset_id, has_anchor=True, feature_set=None):
    brief_store = ModelingBriefStore(db_session)
    validation_anchor = (
        {"has_anchor": True, "anchor_entity": "e1", "anchor_condition": "c", "expected_label": "anomalous"}
        if has_anchor else
        {"has_anchor": False, "reason": "checked thoroughly, nothing found"}
    )
    return brief_store.save_brief(
        dataset_id=dataset_id,
        label_status="absent",
        label_column=None,
        label_notes="no label",
        feature_set=feature_set or [{"column": "a", "role": "feature", "reason": "r"}],
        preprocessing_rules=_full_preprocessing_rules(),
        validation_strategy="v",
        confidence=0.8,
        validation_anchor=validation_anchor
    )


def _full_validation_results(with_anchor=True):
    results = {
        "temporal_holdout": {"performed": True, "result": "holds up", "detail": "checked on a later time slice"},
        "cross_entity_generalization": {"performed": True, "result": "generalizes", "detail": "checked on a held-out entity"},
    }
    if with_anchor:
        results["anchor_validation"] = {"result": "flagged as expected"}
    return results


def _full_feature_engineering(columns=("a",)):
    return [
        {"column": c, "status": "used_directly", "output_features": [c], "detail": "used as-is"}
        for c in columns
    ]


def test_compute_missing_model_result_fields_all_missing_when_no_result():
    missing = compute_missing_model_result_fields(None, None)
    assert len(missing) == 7


def test_compute_missing_model_result_fields_flags_unaddressed_feature_columns(db_session, dataset_id, tmp_path):
    brief = _save_brief(
        db_session, dataset_id,
        feature_set=[
            {"column": "a", "role": "feature", "reason": "r"},
            {"column": "b", "role": "feature", "reason": "r"},
        ]
    )
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"fake")
    result_store = ModelResultStore(db_session)
    result = result_store.save_model_result(
        dataset_id=dataset_id,
        algorithm="IsolationForest",
        algorithm_rationale="scales well",
        applied_feature_engineering=_full_feature_engineering(columns=("a",)),
        model_path=str(model_path),
        validation_results=_full_validation_results(),
        confidence=0.7,
        limitations_notes="none"
    )

    missing = compute_missing_model_result_fields(result, brief)

    assert len(missing) == 1
    assert "column(s): b" in missing[0]


def test_compute_missing_model_result_fields_satisfied_when_all_columns_addressed(db_session, dataset_id, tmp_path):
    brief = _save_brief(
        db_session, dataset_id,
        feature_set=[
            {"column": "a", "role": "feature", "reason": "r"},
            {"column": "b", "role": "feature", "reason": "r"},
        ]
    )
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"fake")
    result_store = ModelResultStore(db_session)
    result = result_store.save_model_result(
        dataset_id=dataset_id,
        algorithm="IsolationForest",
        algorithm_rationale="scales well",
        applied_feature_engineering=[
            {"column": "a", "status": "used_directly", "output_features": ["a"], "detail": "used as-is"},
            {"column": "b", "status": "dropped", "output_features": [], "detail": "redundant with a"},
        ],
        model_path=str(model_path),
        validation_results=_full_validation_results(),
        confidence=0.7,
        limitations_notes="none"
    )

    missing = compute_missing_model_result_fields(result, brief)
    assert not any("column(s)" in m for m in missing)


def test_compute_missing_model_result_fields_ignores_non_feature_role_columns(db_session, dataset_id, tmp_path):
    brief = _save_brief(
        db_session, dataset_id,
        feature_set=[
            {"column": "a", "role": "feature", "reason": "r"},
            {"column": "id", "role": "identifier", "reason": "grouping key"},
        ]
    )
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"fake")
    result_store = ModelResultStore(db_session)
    result = result_store.save_model_result(
        dataset_id=dataset_id,
        algorithm="IsolationForest",
        algorithm_rationale="scales well",
        applied_feature_engineering=_full_feature_engineering(columns=("a",)),
        model_path=str(model_path),
        validation_results=_full_validation_results(),
        confidence=0.7,
        limitations_notes="none"
    )

    missing = compute_missing_model_result_fields(result, brief)
    assert not any("column(s)" in m for m in missing)


def test_compute_missing_model_result_fields_requires_model_path_to_exist_on_disk(db_session, dataset_id, tmp_path):
    brief = _save_brief(db_session, dataset_id)
    result_store = ModelResultStore(db_session)

    result = result_store.save_model_result(
        dataset_id=dataset_id,
        algorithm="IsolationForest",
        algorithm_rationale="scales well",
        applied_feature_engineering=_full_feature_engineering(),
        model_path=str(tmp_path / "does_not_exist.pkl"),
        validation_results=_full_validation_results(),
        confidence=0.7,
        limitations_notes="none"
    )
    missing = compute_missing_model_result_fields(result, brief)
    assert any("does not exist" in m for m in missing)

    real_path = tmp_path / "model.pkl"
    real_path.write_bytes(b"fake")
    result = result_store.save_model_result(
        dataset_id=dataset_id,
        algorithm="IsolationForest",
        algorithm_rationale="scales well",
        applied_feature_engineering=_full_feature_engineering(),
        model_path=str(real_path),
        validation_results=_full_validation_results(),
        confidence=0.7,
        limitations_notes="none"
    )
    missing = compute_missing_model_result_fields(result, brief)
    assert not any("does not exist" in m for m in missing)


def test_compute_missing_model_result_fields_requires_anchor_validation_when_brief_has_anchor(db_session, dataset_id, tmp_path):
    brief = _save_brief(db_session, dataset_id, has_anchor=True)
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"fake")
    result_store = ModelResultStore(db_session)

    validation_results = _full_validation_results(with_anchor=False)
    result = result_store.save_model_result(
        dataset_id=dataset_id,
        algorithm="IsolationForest",
        algorithm_rationale="scales well",
        applied_feature_engineering=_full_feature_engineering(),
        model_path=str(model_path),
        validation_results=validation_results,
        confidence=0.7,
        limitations_notes="none"
    )
    missing = compute_missing_model_result_fields(result, brief)
    assert any("anchor_validation" in m for m in missing)


def test_compute_missing_model_result_fields_anchor_not_required_without_brief_anchor(db_session, dataset_id, tmp_path):
    brief = _save_brief(db_session, dataset_id, has_anchor=False)
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"fake")
    result_store = ModelResultStore(db_session)

    result = result_store.save_model_result(
        dataset_id=dataset_id,
        algorithm="IsolationForest",
        algorithm_rationale="scales well",
        applied_feature_engineering=_full_feature_engineering(),
        model_path=str(model_path),
        validation_results=_full_validation_results(with_anchor=False),
        confidence=0.7,
        limitations_notes="none"
    )
    missing = compute_missing_model_result_fields(result, brief)
    assert not any("anchor_validation" in m for m in missing)


def test_compute_missing_model_result_fields_requires_temporal_and_cross_entity_performed_flag(db_session, dataset_id, tmp_path):
    brief = _save_brief(db_session, dataset_id, has_anchor=False)
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"fake")
    result_store = ModelResultStore(db_session)

    # Missing entirely
    result = result_store.save_model_result(
        dataset_id=dataset_id,
        algorithm="IsolationForest",
        algorithm_rationale="scales well",
        applied_feature_engineering=_full_feature_engineering(),
        model_path=str(model_path),
        validation_results={},
        confidence=0.7,
        limitations_notes="none"
    )
    missing = compute_missing_model_result_fields(result, brief)
    assert any("temporal_holdout" in m for m in missing)
    assert any("cross_entity_generalization" in m for m in missing)

    # performed=False with no detail - still a gap
    result = result_store.save_model_result(
        dataset_id=dataset_id,
        algorithm="IsolationForest",
        algorithm_rationale="scales well",
        applied_feature_engineering=_full_feature_engineering(),
        model_path=str(model_path),
        validation_results={
            "temporal_holdout": {"performed": False},
            "cross_entity_generalization": {"performed": False},
        },
        confidence=0.7,
        limitations_notes="none"
    )
    missing = compute_missing_model_result_fields(result, brief)
    assert any("temporal_holdout" in m and "detail" in m for m in missing)

    # performed=False with a stated detail - satisfied, a deliberate skip is allowed
    result = result_store.save_model_result(
        dataset_id=dataset_id,
        algorithm="IsolationForest",
        algorithm_rationale="scales well",
        applied_feature_engineering=_full_feature_engineering(),
        model_path=str(model_path),
        validation_results={
            "temporal_holdout": {"performed": False, "detail": "not enough time span in this dataset"},
            "cross_entity_generalization": {"performed": False, "detail": "only one entity present"},
        },
        confidence=0.7,
        limitations_notes="none"
    )
    missing = compute_missing_model_result_fields(result, brief)
    assert not any("temporal_holdout" in m for m in missing)
    assert not any("cross_entity_generalization" in m for m in missing)


def test_model_result_store_upserts_instead_of_duplicating(db_session, dataset_id, tmp_path):
    result_store = ModelResultStore(db_session)
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"fake")

    result_store.save_model_result(
        dataset_id=dataset_id, algorithm="LogisticRegression", algorithm_rationale="first pass",
        applied_feature_engineering=[], model_path=str(model_path), validation_results={},
        confidence=0.4, limitations_notes="first pass"
    )
    result_store.save_model_result(
        dataset_id=dataset_id, algorithm="IsolationForest", algorithm_rationale="second pass",
        applied_feature_engineering=[], model_path=str(model_path), validation_results={},
        confidence=0.8, limitations_notes="second pass"
    )

    saved = result_store.get_model_result(dataset_id)
    assert saved.algorithm == "IsolationForest"
    assert saved.confidence == 0.8


def test_model_result_store_returns_none_when_absent(db_session, dataset_id):
    result_store = ModelResultStore(db_session)
    assert result_store.get_model_result(dataset_id) is None


def test_finalize_model_result_tool_reports_missing_fields_when_incomplete(db_session, dataset_id):
    result_store = ModelResultStore(db_session)
    tool = make_finalize_model_result_tool(result_store, dataset_id)

    result = tool.func(
        algorithm="",
        algorithm_rationale="",
        applied_feature_engineering=[],
        model_path="",
        validation_results={},
        confidence=0.5,
        limitations_notes=""
    )

    assert result["complete"] is False
    assert "algorithm" in result["text"]
    assert "limitations_notes" in result["text"]


def test_finalize_model_result_tool_reports_complete_when_all_fields_present(db_session, dataset_id, tmp_path):
    brief = _save_brief(db_session, dataset_id, has_anchor=True)
    result_store = ModelResultStore(db_session)
    tool = make_finalize_model_result_tool(result_store, dataset_id, brief=brief)

    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"fake")

    result = tool.func(
        algorithm="IsolationForest",
        algorithm_rationale="scales well and needs no labels",
        applied_feature_engineering=_full_feature_engineering(),
        model_path=str(model_path),
        validation_results=_full_validation_results(with_anchor=True),
        confidence=0.85,
        limitations_notes="cross-entity generalization only checked on one held-out entity"
    )

    assert result["complete"] is True
    saved = result_store.get_model_result(dataset_id)
    assert saved.algorithm == "IsolationForest"
    assert compute_missing_model_result_fields(saved, brief) == []


def test_finalize_model_result_tool_upserts_instead_of_raising(db_session, dataset_id, tmp_path):
    result_store = ModelResultStore(db_session)
    tool = make_finalize_model_result_tool(result_store, dataset_id)
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"fake")

    tool.func(
        algorithm="LogisticRegression", algorithm_rationale="first pass",
        applied_feature_engineering=[], model_path=str(model_path),
        validation_results={}, confidence=0.4, limitations_notes="first pass"
    )
    tool.func(
        algorithm="IsolationForest", algorithm_rationale="second pass",
        applied_feature_engineering=[], model_path=str(model_path),
        validation_results={}, confidence=0.9, limitations_notes="second pass"
    )

    saved = result_store.get_model_result(dataset_id)
    assert saved.algorithm == "IsolationForest"
    assert saved.confidence == 0.9
