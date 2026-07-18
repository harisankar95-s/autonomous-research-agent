from src.llm.client import BaseLLMClient, BaseEmbeddingClient
from sqlalchemy.orm import Session as DBSession
from src.memory.models import Session,DatasetFacts,Observation
from src.utils.logger import get_logger
import uuid

logger = get_logger(__name__)

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

    def save_facts(self, dataset_id: str, inferred_task_type: str, candidate_targets: list) -> None:

        new_fact = DatasetFacts(
            dataset_id = dataset_id,
            inferred_task_type = inferred_task_type,
            candidate_targets = candidate_targets
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