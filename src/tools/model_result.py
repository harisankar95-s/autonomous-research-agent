from src.tools.base import Tool
from src.memory.manager import ModelResultStore, compute_missing_model_result_fields
from src.memory.models import ModelingBrief
from src.utils.logger import get_logger

logger = get_logger(__name__)


def make_finalize_model_result_tool(
    result_store: ModelResultStore,
    dataset_id: str,
    brief: ModelingBrief | None = None
) -> Tool:
    def finalize_model_result(
        algorithm: str,
        algorithm_rationale: str,
        applied_feature_engineering: list,
        model_path: str,
        validation_results: dict,
        confidence: float,
        limitations_notes: str
    ) -> dict:
        result = result_store.save_model_result(
            dataset_id=dataset_id,
            algorithm=algorithm,
            algorithm_rationale=algorithm_rationale,
            applied_feature_engineering=applied_feature_engineering,
            model_path=model_path,
            validation_results=validation_results,
            confidence=confidence,
            limitations_notes=limitations_notes
        )
        missing = compute_missing_model_result_fields(result, brief)

        if missing:
            text = (
                "Model result saved, but it is still incomplete. Still missing: "
                + "; ".join(missing)
                + ". Call finalize_model_result again with these filled in."
            )
        else:
            text = "Model result finalized - all required fields are complete."

        logger.info(
            f"Model result finalize attempted | dataset_id={dataset_id} | "
            f"complete={not missing}"
        )
        return {"text": text, "complete": not missing}

    return Tool(
        name="finalize_model_result",
        description=(
            "Consolidate your feature engineering, trained model, and "
            "validation results into the single structured record a future "
            "stage will rely on. Call this once, near the end, after you "
            "actually have a trained and validated model - not as a plan of "
            "what you intend to do. It must be complete: if any required "
            "field is left empty, this tool tells you exactly what's still "
            "missing and you must call it again with those filled in. "
            "applied_feature_engineering must account for every column the "
            "brief tagged role='feature' - used directly, transformed, or "
            "explicitly dropped with a stated reason - or this tool tells "
            "you exactly which ones are still unaccounted for. model_path "
            "must point to a real, already-persisted model artifact - this "
            "tool checks that the file actually exists, not just that a "
            "path string was given. If the brief defines a validation_anchor, "
            "validation_results must include a non-empty anchor_validation "
            "entry. validation_results must also state whether temporal and "
            "cross-entity validation were performed - if either was "
            "skipped, say why rather than omitting it."
        ),
        parameters={
            "algorithm": "the short name of the algorithm actually trained (e.g. 'IsolationForest', 'LogisticRegression')",
            "algorithm_rationale": "why this algorithm fits this data and the brief - what drove this choice over alternatives",
            "applied_feature_engineering": "a non-empty list of objects, each with 'column' (the brief's column name), 'status' (one of 'used_directly', 'transformed', 'dropped'), 'output_features' (the actual derived feature name(s) produced, empty list if dropped), and 'detail' (what was actually done, or why dropped) - must cover every column the brief tagged role='feature'",
            "model_path": "the exact path returned when execute_python_code persisted your trained model artifact",
            "validation_results": "an object recording what you actually checked and found. 'anchor_validation' is required if the brief has a validation_anchor. 'temporal_holdout' and 'cross_entity_generalization' must each state at least {'performed': true/false, ...} - if performed is false, include a 'detail' explaining why it wasn't applicable or possible. Where performed, include 'result' and 'detail' describing what was found",
            "confidence": "your overall confidence in this trained model and its validation, from 0 to 1",
            "limitations_notes": "an honest account of what this model could not validate, where you're least confident, and why - state explicitly if something could not be checked rather than omitting it"
        },
        func=finalize_model_result
    )
