from src.tools.base import Tool
from src.memory.manager import ModelingBriefStore, compute_missing_fields
from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_finalize_modeling_brief_tool(brief_store: ModelingBriefStore, dataset_id: str) -> Tool:
    def finalize_modeling_brief(
        label_status: str,
        label_notes: str,
        feature_set: list,
        preprocessing_rules: list,
        validation_strategy: str,
        confidence: float,
        label_column: str = ""
    ) -> dict:
        brief = brief_store.save_brief(
            dataset_id=dataset_id,
            label_status=label_status,
            label_column=label_column or None,
            label_notes=label_notes,
            feature_set=feature_set,
            preprocessing_rules=preprocessing_rules,
            validation_strategy=validation_strategy,
            confidence=confidence
        )
        missing = compute_missing_fields(brief)

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
            "empty field is treated as unanswered, not as 'nothing found'."
        ),
        parameters={
            "label_status": "one of 'present', 'absent', or 'undetermined' - whether this dataset (or its original source) has any label/outcome/fault column",
            "label_column": "optional - the exact label column name, required if label_status is 'present'",
            "label_notes": "your reasoning for the label_status determination - what you checked and what you found",
            "feature_set": "a non-empty list of objects, each with 'column', 'role' (e.g. 'feature', 'excluded', 'identifier', 'timestamp', 'redundant'), and 'reason'",
            "preprocessing_rules": "a non-empty list of objects, each with 'rule' and 'detail' - write one entry stating 'none identified' if there genuinely are none",
            "validation_strategy": "how a model built on this data should be validated or evaluated, given what you know about labels and structure",
            "confidence": "your overall confidence in this brief, from 0 to 1"
        },
        func=finalize_modeling_brief
    )
