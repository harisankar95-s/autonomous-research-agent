from src.memory.manager import (
    ModelingBriefStore,
    compute_missing_fields,
    compute_row_coverage_gaps,
    compute_preprocessing_category_gaps,
    compute_validation_anchor_gaps,
)
from src.tools.modeling_brief import make_finalize_modeling_brief_tool


def _full_preprocessing_rules():
    return [
        {"category": "missing_values", "rule": "none needed", "detail": "d"},
        {"category": "placeholder_values", "rule": "none needed", "detail": "d"},
        {"category": "scaling", "rule": "none needed", "detail": "d"},
        {"category": "encoding", "rule": "none needed", "detail": "d"},
    ]


def test_compute_missing_fields_all_missing_when_no_brief():
    missing = compute_missing_fields(None)
    assert len(missing) == 6


def test_compute_missing_fields_flags_uncovered_columns(db_session, dataset_id):
    brief_store = ModelingBriefStore(db_session)
    columns = [
        {"name": "wind_speed", "type": "text"},
        {"name": "rotor_temp", "type": "text"},
        {"name": "gearbox_oil", "type": "text"},
    ]

    brief = brief_store.save_brief(
        dataset_id=dataset_id,
        label_status="absent",
        label_column=None,
        label_notes="no label",
        feature_set=[{"column": "wind_speed", "role": "feature", "reason": "r"}],
        preprocessing_rules=_full_preprocessing_rules(),
        validation_strategy="v",
        confidence=0.8,
        validation_anchor={"has_anchor": False, "reason": "checked, nothing found"}
    )

    missing = compute_missing_fields(brief, columns=columns)

    assert len(missing) == 1
    assert "rotor_temp" in missing[0]
    assert "gearbox_oil" in missing[0]
    assert "wind_speed" not in missing[0].split(":")[1]


def test_compute_missing_fields_satisfied_when_columns_excluded_explicitly(db_session, dataset_id):
    brief_store = ModelingBriefStore(db_session)
    columns = [{"name": "a", "type": "text"}, {"name": "b", "type": "text"}]

    brief = brief_store.save_brief(
        dataset_id=dataset_id,
        label_status="absent",
        label_column=None,
        label_notes="no label",
        feature_set=[
            {"column": "a", "role": "feature", "reason": "r"},
            {"column": "b", "role": "excluded", "reason": "not relevant"},
        ],
        preprocessing_rules=_full_preprocessing_rules(),
        validation_strategy="v",
        confidence=0.8,
        validation_anchor={"has_anchor": False, "reason": "checked, nothing found"}
    )

    assert compute_missing_fields(brief, columns=columns) == []


def test_compute_row_coverage_gaps_flags_missing_count_and_group_by():
    entity_columns = [{"column": "System_Name", "distinct_values": ["T01", "T02"]}]

    gaps = compute_row_coverage_gaps([], entity_columns)

    assert any("COUNT(*)" in g for g in gaps)
    assert any("System_Name" in g for g in gaps)


def test_compute_row_coverage_gaps_satisfied_when_queries_present():
    entity_columns = [{"column": "System_Name", "distinct_values": ["T01", "T02"]}]
    query_log = [
        'SELECT COUNT(*) FROM "turbine_data"',
        'SELECT "System_Name", AVG("ACTIVE_POWER") FROM "turbine_data" GROUP BY "System_Name"',
    ]

    gaps = compute_row_coverage_gaps(query_log, entity_columns)

    assert gaps == []


def test_finalize_modeling_brief_tool_reports_column_and_row_coverage_gaps(db_session, dataset_id):
    brief_store = ModelingBriefStore(db_session)
    columns = [{"name": "wind_speed", "type": "text"}, {"name": "turbine_id", "type": "text"}]
    entity_columns = [{"column": "turbine_id", "distinct_values": ["x", "y"]}]
    query_log = []

    tool = make_finalize_modeling_brief_tool(
        brief_store, dataset_id,
        columns=columns,
        entity_columns=entity_columns,
        query_log=query_log
    )

    result = tool.func(
        label_status="absent",
        label_notes="no label",
        feature_set=[{"column": "wind_speed", "role": "feature", "reason": "r"}],
        preprocessing_rules=[{"rule": "none identified", "detail": "d"}],
        validation_strategy="v",
        confidence=0.8
    )

    assert result["complete"] is False
    assert "turbine_id" in result["text"]
    assert "COUNT(*)" in result["text"]

    # Now cover the missing column and satisfy the row-coverage checks
    query_log.append('SELECT COUNT(*) FROM "t"')
    query_log.append('SELECT "turbine_id" FROM "t" GROUP BY "turbine_id"')

    result = tool.func(
        label_status="absent",
        label_notes="no label",
        feature_set=[
            {"column": "wind_speed", "role": "feature", "reason": "r"},
            {"column": "turbine_id", "role": "identifier", "reason": "grouping key"},
        ],
        preprocessing_rules=_full_preprocessing_rules(),
        validation_strategy="v",
        confidence=0.8,
        entity_heterogeneity_notes="all entities behave consistently",
        validation_anchor={"has_anchor": False, "reason": "checked, nothing found"}
    )

    assert result["complete"] is True


def test_finalize_modeling_brief_tool_reports_missing_fields_when_incomplete(db_session, dataset_id):
    brief_store = ModelingBriefStore(db_session)

    tool = make_finalize_modeling_brief_tool(brief_store, dataset_id)

    result = tool.func(
        label_status="undetermined",
        label_notes="Checked information_schema and the project brief; no label/fault column found.",
        feature_set=[],
        preprocessing_rules=[],
        validation_strategy="",
        confidence=0.5
    )

    assert result["complete"] is False
    assert "feature_set" in result["text"]
    assert "validation_strategy" in result["text"]


def test_finalize_modeling_brief_tool_reports_complete_when_all_fields_present(db_session, dataset_id):
    brief_store = ModelingBriefStore(db_session)

    tool = make_finalize_modeling_brief_tool(brief_store, dataset_id)

    result = tool.func(
        label_status="present",
        label_column="churned",
        label_notes="Explicit binary outcome column confirmed via schema.",
        feature_set=[{"column": "tenure", "role": "feature", "reason": "correlates with outcome"}],
        preprocessing_rules=_full_preprocessing_rules(),
        validation_strategy="time-based holdout, evaluated with AUC",
        confidence=0.85
    )

    assert result["complete"] is True

    saved = brief_store.get_brief(dataset_id)
    assert saved.label_status == "present"
    assert saved.label_column == "churned"
    assert compute_missing_fields(saved) == []


def test_finalize_modeling_brief_tool_upserts_instead_of_raising(db_session, dataset_id):
    brief_store = ModelingBriefStore(db_session)

    tool = make_finalize_modeling_brief_tool(brief_store, dataset_id)

    tool.func(
        label_status="undetermined",
        label_notes="first pass",
        feature_set=[{"column": "a", "role": "feature", "reason": "r"}],
        preprocessing_rules=[{"rule": "none identified", "detail": "d"}],
        validation_strategy="v1",
        confidence=0.4
    )

    tool.func(
        label_status="absent",
        label_notes="confirmed absent after checking source data",
        feature_set=[{"column": "a", "role": "feature", "reason": "r"}],
        preprocessing_rules=[{"rule": "none identified", "detail": "d"}],
        validation_strategy="v2",
        confidence=0.9
    )

    saved = brief_store.get_brief(dataset_id)
    assert saved.label_status == "absent"
    assert saved.validation_strategy == "v2"


def test_save_brief_warns_on_column_coverage_regression(db_session, dataset_id, caplog):
    brief_store = ModelingBriefStore(db_session)

    brief_store.save_brief(
        dataset_id=dataset_id,
        label_status="absent",
        label_column=None,
        label_notes="no label",
        feature_set=[
            {"column": "wind_speed", "role": "feature", "reason": "r"},
            {"column": "rotor_temp", "role": "feature", "reason": "r"},
        ],
        preprocessing_rules=[{"rule": "none identified", "detail": "d"}],
        validation_strategy="v1",
        confidence=0.8
    )

    with caplog.at_level("WARNING"):
        brief_store.save_brief(
            dataset_id=dataset_id,
            label_status="absent",
            label_column=None,
            label_notes="narrower pass",
            feature_set=[{"column": "wind_speed", "role": "feature", "reason": "r"}],
            preprocessing_rules=[{"rule": "none identified", "detail": "d"}],
            validation_strategy="v2",
            confidence=0.9
        )

    assert any("regressing" in record.message.lower() for record in caplog.records)


def test_save_brief_no_warning_when_coverage_grows(db_session, dataset_id, caplog):
    brief_store = ModelingBriefStore(db_session)

    brief_store.save_brief(
        dataset_id=dataset_id,
        label_status="absent",
        label_column=None,
        label_notes="no label",
        feature_set=[{"column": "wind_speed", "role": "feature", "reason": "r"}],
        preprocessing_rules=[{"rule": "none identified", "detail": "d"}],
        validation_strategy="v1",
        confidence=0.8
    )

    with caplog.at_level("WARNING"):
        brief_store.save_brief(
            dataset_id=dataset_id,
            label_status="absent",
            label_column=None,
            label_notes="broader pass",
            feature_set=[
                {"column": "wind_speed", "role": "feature", "reason": "r"},
                {"column": "rotor_temp", "role": "feature", "reason": "r"},
            ],
            preprocessing_rules=[{"rule": "none identified", "detail": "d"}],
            validation_strategy="v2",
            confidence=0.9
        )

    assert not any("regressing" in record.message.lower() for record in caplog.records)


def test_label_column_required_when_label_status_present(db_session, dataset_id):
    brief_store = ModelingBriefStore(db_session)

    tool = make_finalize_modeling_brief_tool(brief_store, dataset_id)

    result = tool.func(
        label_status="present",
        label_notes="found a label but forgot to name it",
        feature_set=[{"column": "a", "role": "feature", "reason": "r"}],
        preprocessing_rules=[{"rule": "none identified", "detail": "d"}],
        validation_strategy="v",
        confidence=0.7
        # label_column omitted
    )

    assert result["complete"] is False
    assert "label_column" in result["text"]


def test_compute_preprocessing_category_gaps_flags_missing_categories():
    gaps = compute_preprocessing_category_gaps([
        {"category": "missing_values", "rule": "none needed", "detail": "d"},
        {"category": "scaling", "rule": "standardize", "detail": "d"},
    ])

    assert any("placeholder_values" in g for g in gaps)
    assert any("encoding" in g for g in gaps)
    assert not any("missing_values" in g for g in gaps)
    assert not any("'scaling'" in g for g in gaps)


def test_compute_preprocessing_category_gaps_satisfied_when_all_four_present():
    assert compute_preprocessing_category_gaps(_full_preprocessing_rules()) == []


def test_compute_preprocessing_category_gaps_all_missing_when_empty():
    gaps = compute_preprocessing_category_gaps([])
    assert len(gaps) == 4


def test_compute_missing_fields_requires_entity_heterogeneity_notes_when_entities_exist(db_session, dataset_id):
    brief_store = ModelingBriefStore(db_session)
    entity_columns = [{"column": "turbine_id", "distinct_values": ["x", "y"]}]

    brief = brief_store.save_brief(
        dataset_id=dataset_id,
        label_status="absent",
        label_column=None,
        label_notes="no label",
        feature_set=[{"column": "a", "role": "feature", "reason": "r"}],
        preprocessing_rules=_full_preprocessing_rules(),
        validation_strategy="v",
        confidence=0.8,
        validation_anchor={"has_anchor": False, "reason": "checked, nothing found"}
    )

    missing = compute_missing_fields(brief, entity_columns=entity_columns)
    assert any("entity_heterogeneity_notes" in m for m in missing)

    brief = brief_store.save_brief(
        dataset_id=dataset_id,
        label_status="absent",
        label_column=None,
        label_notes="no label",
        feature_set=[{"column": "a", "role": "feature", "reason": "r"}],
        preprocessing_rules=_full_preprocessing_rules(),
        validation_strategy="v",
        confidence=0.8,
        entity_heterogeneity_notes="entities behave consistently",
        validation_anchor={"has_anchor": False, "reason": "checked, nothing found"}
    )

    missing = compute_missing_fields(brief, entity_columns=entity_columns)
    assert not any("entity_heterogeneity_notes" in m for m in missing)


def test_compute_missing_fields_does_not_require_entity_heterogeneity_notes_without_entities(db_session, dataset_id):
    brief_store = ModelingBriefStore(db_session)

    brief = brief_store.save_brief(
        dataset_id=dataset_id,
        label_status="absent",
        label_column=None,
        label_notes="no label",
        feature_set=[{"column": "a", "role": "feature", "reason": "r"}],
        preprocessing_rules=_full_preprocessing_rules(),
        validation_strategy="v",
        confidence=0.8,
        validation_anchor={"has_anchor": False, "reason": "checked, nothing found"}
    )

    missing = compute_missing_fields(brief, entity_columns=[])
    assert not any("entity_heterogeneity_notes" in m for m in missing)


def test_compute_validation_anchor_gaps_not_required_when_label_present():
    assert compute_validation_anchor_gaps("present", None) == []


def test_compute_validation_anchor_gaps_required_when_absent_and_missing():
    gaps = compute_validation_anchor_gaps("absent", None)
    assert any("validation_anchor" in g for g in gaps)


def test_compute_validation_anchor_gaps_real_anchor_needs_core_fields():
    gaps = compute_validation_anchor_gaps("absent", {"has_anchor": True, "anchor_entity": "T08"})
    assert any("anchor_condition" in g and "expected_label" in g for g in gaps)


def test_compute_validation_anchor_gaps_satisfied_with_real_anchor():
    gaps = compute_validation_anchor_gaps("absent", {
        "has_anchor": True,
        "anchor_entity": "T08",
        "anchor_condition": "Timestamp >= '2024-11-01'",
        "expected_label": "anomalous"
    })
    assert gaps == []


def test_compute_validation_anchor_gaps_no_anchor_requires_reason():
    gaps = compute_validation_anchor_gaps("absent", {"has_anchor": False})
    assert any("reason" in g for g in gaps)


def test_compute_validation_anchor_gaps_satisfied_with_no_anchor_and_reason():
    gaps = compute_validation_anchor_gaps("absent", {
        "has_anchor": False,
        "reason": "Checked every entity via a broad comparison; nothing deviated from baseline."
    })
    assert gaps == []


def test_finalize_modeling_brief_tool_requires_validation_anchor_when_label_absent(db_session, dataset_id):
    brief_store = ModelingBriefStore(db_session)
    tool = make_finalize_modeling_brief_tool(brief_store, dataset_id)

    result = tool.func(
        label_status="absent",
        label_notes="no label found",
        feature_set=[{"column": "a", "role": "feature", "reason": "r"}],
        preprocessing_rules=_full_preprocessing_rules(),
        validation_strategy="v",
        confidence=0.8
        # validation_anchor omitted
    )

    assert result["complete"] is False
    assert "validation_anchor" in result["text"]

    result = tool.func(
        label_status="absent",
        label_notes="no label found",
        feature_set=[{"column": "a", "role": "feature", "reason": "r"}],
        preprocessing_rules=_full_preprocessing_rules(),
        validation_strategy="v",
        confidence=0.8,
        validation_anchor={"has_anchor": False, "reason": "checked thoroughly, nothing found"}
    )

    assert result["complete"] is True
