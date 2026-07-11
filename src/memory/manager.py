from src.llm.client import BaseLLMClient, BaseEmbeddingClient
from sqlalchemy.orm import Session as DBSession
from src.memory.models import Session
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