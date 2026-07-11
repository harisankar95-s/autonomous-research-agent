from datetime import datetime, timezone
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector

class Base(DeclarativeBase):
    pass

class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(unique=True)
    query: Mapped[str] = mapped_column()
    answer: Mapped[str] = mapped_column()
    model: Mapped[str] = mapped_column()
    total_tokens: Mapped[int] = mapped_column()
    embedding: Mapped[list[float]] = mapped_column(Vector(3072))
    embedding_model: Mapped[str] = mapped_column()
    confidence_score: Mapped[float] = mapped_column(default=0.5)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))