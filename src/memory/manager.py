from src.llm.client import BaseLLMClient, BaseEmbeddingClient
from sqlalchemy.orm import Session as DBSession
from src.memory.models import Session,DatasetFacts,Observation,ModelingBrief,AnalysisImage,UnderstandingRound,ModelResult
from src.utils.logger import get_logger
import ast
import base64
import json
import os
import re
import uuid

logger = get_logger(__name__)

VALID_LABEL_STATUSES = ("present", "absent", "undetermined")
ANALYSIS_IMAGES_DIR = os.path.join("data", "analysis_images")
# Absolute, unlike ANALYSIS_IMAGES_DIR - a model artifact's path is used as
# a Docker volume mount source (to remount a prior artifact into a later
# execute_python_code call), and Docker rejects a relative host path there.
MODEL_ARTIFACTS_DIR = os.path.abspath(os.path.join("data", "model_artifacts"))
DUPLICATE_OBSERVATION_DISTANCE = 0.20


def _mentioned_entity(content: str, entity_values: list[str]) -> str | None:
    """If content mentions exactly one of the known entity values (e.g. one
    specific unit/store/patient id, whatever the dataset's grouping column
    turns out to hold), return it; None if it mentions none or several.
    Used as a guard against merging two findings that are only similar
    because of sentence structure, not because they're the same finding -
    e.g. on a wind-turbine dataset we tested against, two "Unit XX shows a
    developing YYY anomaly" observations about different units landed closer
    in embedding space than genuine restatements of the same finding, so
    distance alone isn't safe here."""
    mentioned = {v for v in entity_values if v and v in content}
    return next(iter(mentioned)) if len(mentioned) == 1 else None


def _mentioned_entities_all(content: str, entity_values: list[str]) -> list[str]:
    """Every entity value found as a substring in content, unlike
    _mentioned_entity which deliberately returns only the singular,
    unambiguous case (for the dedup guard). Storage wants the full list -
    e.g. "show me every finding about T08" should find a sentence that
    mentions T08 alongside other entities too, not just the ones that
    mention it in isolation."""
    return sorted({v for v in entity_values if v and v in content})


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


def _as_dict(value: dict | str | None) -> dict:
    """Same normalization as _as_list, for arguments that should be a
    mapping (e.g. filename -> caption) rather than an array."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            pass
        try:
            parsed = ast.literal_eval(value)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError):
            pass
        logger.warning(f"Could not parse stringified dict value: {value[:200]}")
        return {}
    return {}

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

    def set_columns(
        self,
        dataset_id: str,
        columns: list,
        row_count: int,
        entity_columns: list
    ) -> None:
        """System-captured ground truth (real schema, row count, candidate
        entity columns), not LLM-authored - called deterministically before
        the agent starts, independent of whether it ever calls a tool for
        this. Kept separate from save_facts, which handles the agent's own
        conclusions."""
        existing = self.get_facts(dataset_id)

        if existing:
            existing.columns = columns
            existing.row_count = row_count
            existing.entity_columns = entity_columns
            self.db_session.commit()
            logger.info(f"Schema coverage data updated | dataset_id={dataset_id}")
        else:
            new_fact = DatasetFacts(
                dataset_id=dataset_id,
                inferred_task_type="",
                candidate_targets=[],
                columns=columns,
                row_count=row_count,
                entity_columns=entity_columns
            )
            self.db_session.add(new_fact)
            self.db_session.commit()
            logger.info(f"Schema coverage data saved | dataset_id={dataset_id}")

class KnowledgeStore:
    def __init__(self, embedding_client: BaseEmbeddingClient, db_session: DBSession):
        self.embedding_client = embedding_client
        self.db_session = db_session

    async def find_duplicate(
            self,
            dataset_id: str,
            content: str,
            embedding: list[float],
            entity_values: list[str] | None = None
        ) -> Observation | None:
            """Shared duplicate-detection logic: the closest existing
            observation for this dataset, if it's near enough in embedding
            space AND not plausibly about a different specific entity - e.g.
            a before/after-cleanup pair about the same column can land
            closer than genuine duplicates because they share a subject, not
            a conclusion. Used both to skip inserting a near-duplicate in
            save_observation, and to decide whether a blind-check finding is
            genuinely new before merging it back into the main dataset."""
            closest = (
                self.db_session.query(
                    Observation,
                    Observation.embedding.cosine_distance(embedding).label("distance")
                )
                .filter_by(dataset_id=dataset_id)
                .order_by(Observation.embedding.cosine_distance(embedding))
                .first()
            )
            if closest is None:
                return None
            match, distance = closest
            new_entity = _mentioned_entity(content, entity_values or [])
            match_entity = _mentioned_entity(match.content, entity_values or [])
            comparable_entity = (
                new_entity is None or match_entity is None or new_entity == match_entity
            )
            if distance < DUPLICATE_OBSERVATION_DISTANCE and comparable_entity:
                return match
            return None

    async def save_observation(
            self,
            dataset_id: str,
            content: str,
            confidence_score: float,
            supersedes_id: int | None = None,
            entity_values: list[str] | None = None,
            analysis_type: str | None = None,
            evidence: list | None = None
        ) -> int:
            embedding = await self.embedding_client.generate_embedding(content)

            # An explicit supersedes_id is the caller declaring "this is an
            # intentional update to a prior finding", not a restatement -
            # honor that signal and skip the dedup check entirely rather
            # than silently swallowing it.
            if supersedes_id is None:
                duplicate = await self.find_duplicate(dataset_id, content, embedding, entity_values)
                if duplicate is not None:
                    logger.info(
                        f"Observation skipped as near-duplicate | dataset_id={dataset_id} | "
                        f"existing_id={duplicate.id}"
                    )
                    return duplicate.id

            # entities is auto-derived, not agent-supplied - it's always
            # consistent with the dedup guard's own entity detection rather
            # than relying on the agent to remember to fill it in.
            # analysis_type/evidence are optional agent-supplied enrichment
            # for auditability, not a completion gate.
            new_observation = Observation(
                dataset_id=dataset_id,
                content=content,
                embedding=embedding,
                embedding_model=self.embedding_client.model,
                confidence_score=confidence_score,
                supersedes_id=supersedes_id,
                entities=_mentioned_entities_all(content, entity_values or []),
                analysis_type=analysis_type or None,
                evidence=evidence or None
            )

            self.db_session.add(new_observation)
            self.db_session.commit()

            logger.info(f"Observation saved | dataset_id={dataset_id} | id={new_observation.id}")

            return new_observation.id

    def count_observations(self, dataset_id: str) -> int:
        return self.db_session.query(Observation).filter_by(dataset_id=dataset_id).count()

    def get_observations(self, dataset_id: str) -> list[Observation]:
        """Full rows (including embeddings), not just content - lets a
        caller like the blind-check reconciliation diff reuse each row's
        existing embedding instead of re-embedding its text."""
        return self.db_session.query(Observation).filter_by(dataset_id=dataset_id).all()

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
        confidence: float,
        entity_heterogeneity_notes: str | None = None,
        validation_anchor: dict | None = None
    ) -> ModelingBrief:
        feature_set = _as_list(feature_set)
        preprocessing_rules = _as_list(preprocessing_rules)
        validation_anchor = _as_dict(validation_anchor) if validation_anchor else validation_anchor
        existing = self.get_brief(dataset_id)

        if existing:
            old_covered = len({f.get("column") for f in existing.feature_set if isinstance(f, dict)})
            new_covered = len({f.get("column") for f in feature_set if isinstance(f, dict)})
            if new_covered < old_covered:
                logger.warning(
                    f"Modeling brief for dataset_id={dataset_id} is REGRESSING in "
                    f"coverage: {old_covered} -> {new_covered} columns accounted for. "
                    f"Overwriting anyway - a genuinely narrower brief can be correct, "
                    f"but this is not silent."
                )
            existing.label_status = label_status
            existing.label_column = label_column
            existing.label_notes = label_notes
            existing.feature_set = feature_set
            existing.preprocessing_rules = preprocessing_rules
            existing.validation_strategy = validation_strategy
            existing.confidence = confidence
            existing.entity_heterogeneity_notes = entity_heterogeneity_notes
            existing.validation_anchor = validation_anchor
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
                confidence=confidence,
                entity_heterogeneity_notes=entity_heterogeneity_notes,
                validation_anchor=validation_anchor
            )
            self.db_session.add(new_brief)
            self.db_session.commit()
            logger.info(f"Modeling brief saved | dataset_id={dataset_id}")
            return new_brief


class UnderstandingRoundStore:
    """Audit trail for multi-round convergence - each row is a plain-code
    diff of one round's actual effect (columns/observations/gaps), not the
    model's own self-report of whether it found anything new."""
    def __init__(self, db_session: DBSession):
        self.db_session = db_session

    def record_round(
        self,
        dataset_id: str,
        round_number: int,
        new_columns_covered: list,
        new_observations_count: int,
        row_gaps_remaining: int,
        converged: bool
    ) -> UnderstandingRound:
        row = UnderstandingRound(
            dataset_id=dataset_id,
            round_number=round_number,
            new_columns_covered=new_columns_covered,
            new_observations_count=new_observations_count,
            row_gaps_remaining=row_gaps_remaining,
            converged=converged
        )
        self.db_session.add(row)
        self.db_session.commit()
        logger.info(
            f"Round recorded | dataset_id={dataset_id} | round={round_number} | "
            f"new_columns={len(new_columns_covered)} | new_observations={new_observations_count} | "
            f"row_gaps_remaining={row_gaps_remaining} | converged={converged}"
        )
        return row

    def get_rounds(self, dataset_id: str) -> list[UnderstandingRound]:
        return (
            self.db_session.query(UnderstandingRound)
            .filter_by(dataset_id=dataset_id)
            .order_by(UnderstandingRound.round_number)
            .all()
        )


def compute_row_coverage_gaps(query_log: list[str] | None, entity_columns: list | None) -> list[str]:
    """Plain-code check that the agent actually verified full-population
    statistics via SQL rather than only ever working from a sample - a
    substring heuristic on the queries it ran. Not perfect, but targets the
    real failure mode (conclusions drawn from an unverified sample) instead
    of just bookkeeping column names."""
    query_log = query_log or []
    entity_columns = entity_columns or []
    combined = " ".join(q.lower() for q in query_log)

    gaps = []
    if "count(*)" not in combined and "count (*)" not in combined:
        gaps.append("never ran a COUNT(*) query to establish the true row count")

    for entity in entity_columns:
        col = entity["column"]
        pattern = re.compile(r'group\s+by\s+"?' + re.escape(col.lower()) + r'"?')
        if not pattern.search(combined):
            preview = entity["distinct_values"][:5]
            suffix = "..." if len(entity["distinct_values"]) > 5 else ""
            gaps.append(
                f'never ran a GROUP BY "{col}" query to establish per-entity '
                f"baseline statistics ({len(entity['distinct_values'])} distinct "
                f"values exist, e.g. {preview}{suffix})"
            )
    return gaps


REQUIRED_PREPROCESSING_CATEGORIES = ("missing_values", "placeholder_values", "scaling", "encoding")


def compute_preprocessing_category_gaps(preprocessing_rules: list | None) -> list[str]:
    """preprocessing_rules being merely non-empty doesn't mean every
    universal concern was actually considered - a single blanket "none
    needed" entry satisfies a plain non-emptiness check without anyone
    having looked at missingness, placeholder/sentinel values, scaling, or
    encoding specifically. These four are properties of any tabular
    dataset regardless of industry, so requiring each to be explicitly
    addressed (even as "none needed") is a fixed, generic checklist -
    unlike a broader statistical ontology, this one is justified."""
    preprocessing_rules = preprocessing_rules or []
    present = {r.get("category") for r in preprocessing_rules if isinstance(r, dict)}
    return [
        f"preprocessing_rules has no '{category}' entry - state explicitly "
        f"whether this is needed, even if the answer is none"
        for category in REQUIRED_PREPROCESSING_CATEGORIES
        if category not in present
    ]


def compute_validation_anchor_gaps(label_status: str | None, validation_anchor: dict | None) -> list[str]:
    """When there's no real label/target (label_status == "absent"), a
    modeling agent still needs something concrete to validate against - a
    known, well-evidenced case, or an honest, evidenced conclusion that
    none exists. Structured rather than free text so a modeling agent can
    act on it directly: anchor_entity + anchor_condition is a filter it can
    run, not a sentence it has to parse."""
    if label_status != "absent":
        return []
    if not validation_anchor:
        return [
            "validation_anchor (required since label_status is 'absent' - "
            "either a concrete known case to validate against, or an "
            "explicit, evidenced conclusion that none exists)"
        ]
    if validation_anchor.get("has_anchor"):
        required = ("anchor_entity", "anchor_condition", "expected_label")
        missing_keys = [k for k in required if not validation_anchor.get(k)]
        if missing_keys:
            return [
                f"validation_anchor is missing: {', '.join(missing_keys)} - a "
                f"real anchor needs a specific entity, condition, and expected "
                f"label a modeling agent can act on directly"
            ]
        return []
    if not validation_anchor.get("reason"):
        return [
            "validation_anchor has has_anchor=false but no 'reason' - explain "
            "what was actually checked before concluding no anchor exists"
        ]
    return []


def compute_missing_fields(
    brief: ModelingBrief | None,
    columns: list | None = None,
    entity_columns: list | None = None,
    query_log: list[str] | None = None
) -> list[str]:
    if brief is None:
        missing = [
            "label_status (present/absent/undetermined)",
            "label_notes",
            "feature_set",
            "preprocessing_rules (must address missing_values, placeholder_values, scaling, and encoding)",
            "validation_strategy",
            "confidence",
        ]
        if entity_columns:
            missing.append(
                "entity_heterogeneity_notes (required since this dataset has entity/grouping columns)"
            )
        feature_set = []
    else:
        missing = []
        if not brief.label_status or brief.label_status not in VALID_LABEL_STATUSES:
            missing.append("label_status (must be one of present/absent/undetermined)")
        if brief.label_status == "present" and not brief.label_column:
            missing.append("label_column (required since label_status is 'present')")
        if not brief.label_notes:
            missing.append("label_notes (reasoning for the label status determination)")
        if not brief.feature_set:
            missing.append("feature_set (non-empty list of columns with role and reason)")
        missing.extend(compute_preprocessing_category_gaps(brief.preprocessing_rules))
        if not brief.validation_strategy:
            missing.append("validation_strategy")
        if brief.confidence is None:
            missing.append("confidence")
        if entity_columns and not brief.entity_heterogeneity_notes:
            missing.append(
                "entity_heterogeneity_notes (required since this dataset has entity/grouping "
                "columns) - state whether entities behave consistently or differently, and if "
                "differently, what that means for preprocessing or validation"
            )
        missing.extend(compute_validation_anchor_gaps(brief.label_status, brief.validation_anchor))
        feature_set = brief.feature_set or []

    if columns:
        covered = {f.get("column") for f in feature_set if isinstance(f, dict)}
        real_columns = {c["name"] for c in columns}
        uncovered = sorted(real_columns - covered)
        if uncovered:
            missing.append(
                f"feature_set does not account for {len(uncovered)} column(s): "
                f"{', '.join(uncovered)} - include each as a feature or explicitly excluded"
            )

    if entity_columns is not None or query_log is not None:
        missing.extend(compute_row_coverage_gaps(query_log, entity_columns))

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


def save_model_artifact(dataset_id: str, artifact_bytes: bytes) -> str:
    """Persists the single current trained-model artifact for a dataset,
    overwriting any prior one - cardinality here is one current artifact
    per dataset, not many like AnalysisImage, so the path is stable (not a
    uuid) and a later execute_python_code call can remount the exact same
    path without the agent tracking a changing filename."""
    dataset_dir = os.path.join(MODEL_ARTIFACTS_DIR, dataset_id)
    os.makedirs(dataset_dir, exist_ok=True)
    file_path = os.path.join(dataset_dir, "model.pkl")
    with open(file_path, "wb") as f:
        f.write(artifact_bytes)
    logger.info(f"Model artifact saved | dataset_id={dataset_id} | file_path={file_path}")
    return file_path


class ModelResultStore:
    def __init__(self, db_session: DBSession):
        self.db_session = db_session

    def get_model_result(self, dataset_id: str) -> ModelResult | None:
        result = (
            self.db_session.query(ModelResult)
            .filter_by(dataset_id=dataset_id)
            .first()
        )
        logger.info(f"Model result retrieved | dataset_id={dataset_id} | found={result is not None}")
        return result

    def save_model_result(
        self,
        dataset_id: str,
        algorithm: str,
        algorithm_rationale: str,
        applied_feature_engineering: list,
        model_path: str,
        validation_results: dict,
        confidence: float,
        limitations_notes: str
    ) -> ModelResult:
        applied_feature_engineering = _as_list(applied_feature_engineering)
        validation_results = _as_dict(validation_results) if validation_results else validation_results
        existing = self.get_model_result(dataset_id)

        if existing:
            existing.algorithm = algorithm
            existing.algorithm_rationale = algorithm_rationale
            existing.applied_feature_engineering = applied_feature_engineering
            existing.model_path = model_path
            existing.validation_results = validation_results
            existing.confidence = confidence
            existing.limitations_notes = limitations_notes
            self.db_session.commit()
            logger.info(f"Model result updated | dataset_id={dataset_id}")
            return existing
        else:
            new_result = ModelResult(
                dataset_id=dataset_id,
                algorithm=algorithm,
                algorithm_rationale=algorithm_rationale,
                applied_feature_engineering=applied_feature_engineering,
                model_path=model_path,
                validation_results=validation_results,
                confidence=confidence,
                limitations_notes=limitations_notes
            )
            self.db_session.add(new_result)
            self.db_session.commit()
            logger.info(f"Model result saved | dataset_id={dataset_id}")
            return new_result


def _has_numeric_metrics(leg_result: dict) -> bool:
    """A validation leg's own prose 'detail' is not enough to make its
    numbers queryable later - this checks for a non-empty 'metrics' dict
    where every value is a real number, not text describing one. Excludes
    bool explicitly since bool is a subclass of int in Python and would
    otherwise silently pass as a 'metric'."""
    metrics = leg_result.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        return False
    return all(
        isinstance(v, (int, float)) and not isinstance(v, bool)
        for v in metrics.values()
    )


def compute_missing_model_result_fields(
    result: ModelResult | None,
    brief: ModelingBrief | None
) -> list[str]:
    """Mirrors compute_missing_fields' plain-code philosophy: every
    requirement here is checked against real, structural state (the brief's
    own recorded columns/validation_anchor, the actual filesystem) rather
    than trusting the agent's self-report."""
    if result is None:
        return [
            "algorithm", "algorithm_rationale", "applied_feature_engineering",
            "model_path", "validation_results", "confidence", "limitations_notes",
        ]

    missing = []
    if not result.algorithm:
        missing.append("algorithm")
    if not result.algorithm_rationale:
        missing.append("algorithm_rationale (why this algorithm was chosen)")
    if not result.model_path:
        missing.append("model_path")
    elif not os.path.exists(result.model_path):
        missing.append(f"model_path points to '{result.model_path}', which does not exist on disk")
    if result.confidence is None:
        missing.append("confidence")
    if not result.limitations_notes:
        missing.append("limitations_notes")

    if brief is not None:
        feature_columns = {
            f.get("column") for f in (brief.feature_set or [])
            if isinstance(f, dict) and f.get("role") == "feature"
        }
        addressed = {
            e.get("column") for e in (result.applied_feature_engineering or [])
            if isinstance(e, dict) and e.get("column") and e.get("status") and e.get("detail")
        }
        unaddressed = sorted(feature_columns - addressed)
        if unaddressed:
            missing.append(
                f"applied_feature_engineering does not address {len(unaddressed)} "
                f"brief feature column(s): {', '.join(unaddressed)} - state whether "
                f"each was used directly, transformed, or dropped, and why"
            )

        validation_results = result.validation_results or {}
        if brief.validation_anchor and brief.validation_anchor.get("has_anchor"):
            anchor_result = validation_results.get("anchor_validation")
            if not anchor_result:
                missing.append(
                    "validation_results.anchor_validation (required because the "
                    "modeling brief defines a validation_anchor) - check the model "
                    "against the brief's anchor_entity/anchor_condition/expected_label "
                    "and record what happened"
                )
            elif not _has_numeric_metrics(anchor_result):
                missing.append(
                    "validation_results.anchor_validation.metrics - a non-empty "
                    "object of real numbers (e.g. the anchor's own score and what "
                    "it's being compared against), not just a prose description"
                )
        for leg in ("temporal_holdout", "cross_entity_generalization"):
            leg_result = validation_results.get(leg)
            if not isinstance(leg_result, dict) or "performed" not in leg_result:
                missing.append(
                    f"validation_results.{leg} - state whether this was performed "
                    f"(true/false) and either what you found or why it wasn't "
                    f"applicable or possible"
                )
            elif not leg_result.get("performed") and not leg_result.get("detail"):
                missing.append(
                    f"validation_results.{leg} has performed=false but no 'detail' "
                    f"explaining why"
                )
            elif leg_result.get("performed") and not _has_numeric_metrics(leg_result):
                missing.append(
                    f"validation_results.{leg}.metrics - a non-empty object of "
                    f"real numbers backing up what was found, not just a prose "
                    f"description"
                )

    return missing