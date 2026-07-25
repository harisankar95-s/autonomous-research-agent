from src.memory.manager import ModelingBriefStore, compute_missing_fields
from src.tools.modeling_brief import make_finalize_modeling_brief_tool


def test_compute_missing_fields_all_missing_when_no_brief():
    missing = compute_missing_fields(None)
    assert len(missing) == 6


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
        preprocessing_rules=[{"rule": "none identified", "detail": "no sentinel values or missing data found"}],
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
