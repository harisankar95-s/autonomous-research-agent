from src.tools.base import Tool
from src.memory.manager import ModelingBriefStore, compute_missing_fields
from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_finalize_modeling_brief_tool(
    brief_store: ModelingBriefStore,
    dataset_id: str,
    columns: list | None = None,
    entity_columns: list | None = None,
    query_log: list | None = None
) -> Tool:
    def finalize_modeling_brief(
        label_status: str,
        label_notes: str,
        feature_set: list,
        preprocessing_rules: list,
        validation_strategy: str,
        confidence: float,
        label_column: str = "",
        entity_heterogeneity_notes: str = "",
        validation_anchor: dict | None = None
    ) -> dict:
        brief = brief_store.save_brief(
            dataset_id=dataset_id,
            label_status=label_status,
            label_column=label_column or None,
            label_notes=label_notes,
            feature_set=feature_set,
            preprocessing_rules=preprocessing_rules,
            validation_strategy=validation_strategy,
            confidence=confidence,
            entity_heterogeneity_notes=entity_heterogeneity_notes or None,
            validation_anchor=validation_anchor
        )
        missing = compute_missing_fields(
            brief,
            columns=columns,
            entity_columns=entity_columns,
            query_log=query_log
        )

        if missing:
            text = (
                "Brief saved, but it is still incomplete. Still missing: "
                + "; ".join(missing)
                + ". Call finalize_modeling_brief again with these filled in."
            )
        else:
            text = "Modeling brief finalized - all required fields are complete."

        logger.info(
            f"Modeling brief finalize attempted | dataset_id={dataset_id} | "
            f"complete={not missing}"
        )
        return {"text": text, "complete": not missing}

    return Tool(
        name="finalize_modeling_brief",
        description=(
            "Consolidate your analysis into the single structured modeling "
            "brief a future modeling stage will rely on. Call this once, near "
            "the end, after you've actually done the analysis - not as a "
            "first guess. It must be complete: if any required field is left "
            "empty, this tool tells you exactly what's still missing and you "
            "must call it again with those fields filled in. If you "
            "genuinely found nothing for a field (e.g. no preprocessing is "
            "needed), state that explicitly instead of leaving it empty - an "
            "empty field is treated as unanswered, not as 'nothing found'. "
            "feature_set must account for EVERY real column in the table, "
            "not just the ones you found interesting - include each one as a "
            "feature, identifier, timestamp, redundant, or explicitly "
            "excluded (with a reason), or this tool will tell you exactly "
            "which columns are still unaccounted for. It also checks that "
            "you actually established full-population statistics via SQL "
            "(a COUNT(*), and a GROUP BY per entity if the table has a "
            "grouping dimension like a device or unit id) rather than only "
            "ever working from a sample."
        ),
        parameters={
            "label_status": "one of 'present', 'absent', or 'undetermined' - whether this dataset (or its original source) has any label/outcome/fault column",
            "label_column": "optional - the exact label column name, required if label_status is 'present'",
            "label_notes": "your reasoning for the label_status determination - what you checked and what you found",
            "feature_set": "a non-empty list of objects, each with 'column', 'role' (e.g. 'feature', 'excluded', 'identifier', 'timestamp', 'redundant'), and 'reason' - must cover every real column in the table",
            "preprocessing_rules": "a non-empty list of objects, each with 'category', 'rule', and 'detail'. Every one of these four categories must appear at least once: 'missing_values', 'placeholder_values', 'scaling', 'encoding' - state 'none needed' as the rule if genuinely none, but each category needs its own entry, not one blanket statement covering all of them",
            "validation_strategy": "how a model built on this data should be validated or evaluated, given what you know about labels and structure",
            "confidence": "your overall confidence in this brief, from 0 to 1",
            "entity_heterogeneity_notes": "required if the table has entity/grouping columns - state whether entities behave consistently or differently from each other, and if differently, what that means for preprocessing or validation (e.g. per-entity normalization)",
            "validation_anchor": "required if label_status is 'absent' - either {'has_anchor': true, 'anchor_entity': ..., 'anchor_condition': ..., 'expected_label': ..., 'confidence': ..., 'evidence': [...]} describing a real, well-evidenced case a model should be checked against, or {'has_anchor': false, 'reason': ..., 'evidence': [...]} if genuinely none exists after real investigation. A genuine extreme value that IS a real finding belongs here as the anchor, not in preprocessing_rules as something to clean up - only fake/sentinel/placeholder values belong in preprocessing"
        },
        func=finalize_modeling_brief
    )
