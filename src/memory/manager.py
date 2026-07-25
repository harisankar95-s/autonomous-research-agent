from src.llm.client import BaseLLMClient, BaseEmbeddingClient
from sqlalchemy.orm import Session as DBSession
from src.memory.models import Session,DatasetFacts,Observation,ModelingBrief,AnalysisImage
from src.utils.logger import get_logger
import ast
import base64
import json
import os
import uuid

logger = get_logger(__name__)

VALID_LABEL_STATUSES = ("present", "absent", "undetermined")
ANALYSIS_IMAGES_DIR = os.path.join("data", "analysis_images")


def _as_list(value: list | str) -> list:
    """Gemini's tool-calling sometimes returns array arguments as a
    string instead of a native list - either JSON-encoded, or occasionally
    formatted like a Python literal (single quotes). Normalize at the
    persistence boundary so every caller reading these columns back can
    trust they're real lists."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, list) else []
        except (ValueError, SyntaxError):
            pass
        logger.warning(f"Could not parse stringified list value: {value[:200]}")
        return []
    return []

class MemoryManager:
    def __init__(self,llm_client:BaseLLMClient,embedding_client:BaseEmbeddingClient,db_session: DBSession):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.db_session = db_session
    
    async def summarize_conversation(self, conversation: list[dict]) -> str:
        conversation_text = ""
        for turn in conversation:
            conversation_text += f"Question: {turn['query']}\nAnswer: {turn['answer']}\n\n"
        
        system_prompt = (
            "Summarize the following conversation into a concise paragraph, "
            "capturing the key facts and topics discussed. This summary will be "
            "stored as searchable memory for future reference."
        )
        
        logger.info("Summarizing conversation")
        response = await self.llm_client.send_message(
            message=conversation_text,
            system_prompt=system_prompt
        )
        return response.content
    
    # NOTE: total_tokens is currently passed in as a placeholder (chat.py sends 0),
    # since ReactAgent.run() only returns the final answer string, not the LLMResponse
    # object with actual token usage. Real tracking requires ReactAgent to accumulate
    # and return token counts across its iterations.
    async def save_session(self, summary: str, model: str, total_tokens: int) -> None:
        session_id = str(uuid.uuid4())
        embedding = await self.embedding_client.generate_embedding(summary)
        
        new_session = Session(
            session_id=session_id,
            summary=summary,
            model=model,
            total_tokens=total_tokens,
            embedding=embedding,
            embedding_model=self.embedding_client.model
        )
        
        self.db_session.add(new_session)
        self.db_session.commit()
        
        logger.info(f"Session saved | session_id={session_id}")

    async def get_relevant_context(self, query: str, limit: int = 3) -> list[str]:
        query_embedding = await self.embedding_client.generate_embedding(query)
        
        results = (
            self.db_session.query(Session)
            .order_by(Session.embedding.cosine_distance(query_embedding))
            .limit(limit)
            .all()
        )
        
        logger.info(f"Retrieved {len(results)} relevant sessions")
        return [session.summary for session in results]

class FactStore:
    def __init__(self, db_session: DBSession):
        self.db_session = db_session

    def save_facts(
        self,
        dataset_id: str,
        inferred_task_type: str,
        candidate_targets: list,
        schema_notes: str | None = None
    ) -> None:
        candidate_targets = _as_list(candidate_targets)
        existing = self.get_facts(dataset_id)

        if existing:
            existing.inferred_task_type = inferred_task_type
            existing.candidate_targets = candidate_targets
            if schema_notes:
                existing.schema_notes = schema_notes
            self.db_session.commit()
            logger.info(f"Facts updated | dataset_id={dataset_id}")
        else:
            new_fact = DatasetFacts(
                dataset_id = dataset_id,
                inferred_task_type = inferred_task_type,
                candidate_targets = candidate_targets,
                schema_notes = schema_notes
            )
            self.db_session.add(new_fact)
            self.db_session.commit()
            logger.info(f"Facts saved | dataset_id={dataset_id}")

    def get_facts(self, dataset_id: str) -> DatasetFacts | None:
            result = (
                self.db_session.query(DatasetFacts)
                .filter_by(dataset_id=dataset_id)
                .first()
            )
            logger.info(f"Facts retrieved | dataset_id={dataset_id} | found={result is not None}")
            return result

class KnowledgeStore:
    def __init__(self, embedding_client: BaseEmbeddingClient, db_session: DBSession):
        self.embedding_client = embedding_client
        self.db_session = db_session

    async def save_observation(
            self,
            dataset_id: str,
            content: str,
            confidence_score: float,
            supersedes_id: int | None = None
        ) -> int:
            embedding = await self.embedding_client.generate_embedding(content)

            new_observation = Observation(
                dataset_id=dataset_id,
                content=content,
                embedding=embedding,
                embedding_model=self.embedding_client.model,
                confidence_score=confidence_score,
                supersedes_id=supersedes_id
            )

            self.db_session.add(new_observation)
            self.db_session.commit()

            logger.info(f"Observation saved | dataset_id={dataset_id} | id={new_observation.id}")

            return new_observation.id

    async def get_relevant_observations(self, dataset_id: str, query: str, limit: int = 5) -> list[dict]:
            query_embedding = await self.embedding_client.generate_embedding(query)

            results = (
                self.db_session.query(Observation)
                .filter_by(dataset_id=dataset_id)
                .order_by(Observation.embedding.cosine_distance(query_embedding))
                .limit(limit)
                .all()
            )

            logger.info(f"Retrieved {len(results)} relevant observations | dataset_id={dataset_id}")
            return [{"id": obs.id, "content": obs.content} for obs in results]

class ModelingBriefStore:
    def __init__(self, db_session: DBSession):
        self.db_session = db_session

    def get_brief(self, dataset_id: str) -> ModelingBrief | None:
        result = (
            self.db_session.query(ModelingBrief)
            .filter_by(dataset_id=dataset_id)
            .first()
        )
        logger.info(f"Modeling brief retrieved | dataset_id={dataset_id} | found={result is not None}")
        return result

    def save_brief(
        self,
        dataset_id: str,
        label_status: str,
        label_column: str | None,
        label_notes: str,
        feature_set: list,
        preprocessing_rules: list,
        validation_strategy: str,
        confidence: float
    ) -> ModelingBrief:
        feature_set = _as_list(feature_set)
        preprocessing_rules = _as_list(preprocessing_rules)
        existing = self.get_brief(dataset_id)

        if existing:
            existing.label_status = label_status
            existing.label_column = label_column
            existing.label_notes = label_notes
            existing.feature_set = feature_set
            existing.preprocessing_rules = preprocessing_rules
            existing.validation_strategy = validation_strategy
            existing.confidence = confidence
            self.db_session.commit()
            logger.info(f"Modeling brief updated | dataset_id={dataset_id}")
            return existing
        else:
            new_brief = ModelingBrief(
                dataset_id=dataset_id,
                label_status=label_status,
                label_column=label_column,
                label_notes=label_notes,
                feature_set=feature_set,
                preprocessing_rules=preprocessing_rules,
                validation_strategy=validation_strategy,
                confidence=confidence
            )
            self.db_session.add(new_brief)
            self.db_session.commit()
            logger.info(f"Modeling brief saved | dataset_id={dataset_id}")
            return new_brief


def compute_missing_fields(brief: ModelingBrief | None) -> list[str]:
    if brief is None:
        return [
            "label_status (present/absent/undetermined)",
            "label_notes",
            "feature_set",
            "preprocessing_rules",
            "validation_strategy",
            "confidence",
        ]

    missing = []
    if not brief.label_status or brief.label_status not in VALID_LABEL_STATUSES:
        missing.append("label_status (must be one of present/absent/undetermined)")
    if brief.label_status == "present" and not brief.label_column:
        missing.append("label_column (required since label_status is 'present')")
    if not brief.label_notes:
        missing.append("label_notes (reasoning for the label status determination)")
    if not brief.feature_set:
        missing.append("feature_set (non-empty list of columns with role and reason)")
    if not brief.preprocessing_rules:
        missing.append("preprocessing_rules (non-empty - state 'none identified' if genuinely none)")
    if not brief.validation_strategy:
        missing.append("validation_strategy")
    if brief.confidence is None:
        missing.append("confidence")

    return missing


class ImageStore:
    def __init__(self, db_session: DBSession):
        self.db_session = db_session

    def save_image(self, dataset_id: str, image_b64: str, caption: str = "") -> AnalysisImage:
        dataset_dir = os.path.join(ANALYSIS_IMAGES_DIR, dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)

        filename = f"{uuid.uuid4().hex}.png"
        file_path = os.path.join(dataset_dir, filename)
        with open(file_path, "wb") as f:
            f.write(base64.b64decode(image_b64))

        record = AnalysisImage(
            dataset_id=dataset_id,
            file_path=file_path,
            caption=caption or None
        )
        self.db_session.add(record)
        self.db_session.commit()

        logger.info(f"Analysis image saved | dataset_id={dataset_id} | file_path={file_path}")
        return record