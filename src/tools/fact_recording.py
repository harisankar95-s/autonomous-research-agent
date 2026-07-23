from src.tools.base import Tool
from src.memory.manager import FactStore
from src.utils.logger import get_logger

logger = get_logger(__name__)

def make_record_facts_tool(fact_store: FactStore, dataset_id: str) -> Tool:
    def record_dataset_facts(
        inferred_task_type: str,
        candidate_targets: list,
        schema_notes: str = ""
    ) -> str:
        fact_store.save_facts(
            dataset_id=dataset_id,
            inferred_task_type=inferred_task_type,
            candidate_targets=candidate_targets,
            schema_notes=schema_notes or None
        )
        logger.info(f"Facts recorded via tool | dataset_id={dataset_id}")
        return "Facts recorded successfully."

    return Tool(
        name="record_dataset_facts",
        description=(
            "Record the structural facts you've determined about this dataset "
            "once you have enough confidence to state them. This includes what "
            "kind of ML task this genuinely is, and any candidate target "
            "variables you've identified, each with your confidence and "
            "reasoning. Call this once you're confident, not on your first "
            "guess - it's meant to capture your settled conclusions. Also use "
            "schema_notes to save yourself and future runs the trouble of "
            "rediscovering table mechanics - once you know the exact column "
            "names (including case) and any quoting SQL needs (e.g. "
            "\"MixedCase_Column\"), write them down here so you never have to "
            "re-derive them again."
        ),
        parameters={
            "inferred_task_type": "the kind of ML problem this is, e.g. 'regression', 'classification', 'exploratory analysis'",
            "candidate_targets": "a list of objects, each with 'column', 'confidence' (0 to 1), and 'reason'",
            "schema_notes": "optional - confirmed column names, their types, and any SQL quoting/casing quirks needed to query this table correctly"
        },
        func=record_dataset_facts
    )