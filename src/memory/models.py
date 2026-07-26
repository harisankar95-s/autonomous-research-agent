from datetime import datetime, timezone
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON

class Base(DeclarativeBase):
    pass

class Session(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(unique=True)
    summary: Mapped[str] = mapped_column()
    model: Mapped[str] = mapped_column()
    total_tokens: Mapped[int] = mapped_column()
    embedding: Mapped[list[float]] = mapped_column(Vector(3072))
    embedding_model: Mapped[str] = mapped_column()
    confidence_score: Mapped[float] = mapped_column(default=0.5)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

class DatasetFacts(Base):
    __tablename__ = "dataset_facts"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[str] = mapped_column(unique=True)
    inferred_task_type: Mapped[str] = mapped_column()
    candidate_targets: Mapped[list] = mapped_column(JSON)
    schema_notes: Mapped[str | None] = mapped_column(default=None)
    columns: Mapped[list | None] = mapped_column(JSON, default=None)
    row_count: Mapped[int | None] = mapped_column(default=None)
    entity_columns: Mapped[list | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

class ModelingBrief(Base):
    __tablename__ = "modeling_briefs"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[str] = mapped_column(unique=True)
    label_status: Mapped[str] = mapped_column()
    label_column: Mapped[str | None] = mapped_column(default=None)
    label_notes: Mapped[str] = mapped_column()
    feature_set: Mapped[list] = mapped_column(JSON)
    preprocessing_rules: Mapped[list] = mapped_column(JSON)
    validation_strategy: Mapped[str] = mapped_column()
    confidence: Mapped[float] = mapped_column()
    entity_heterogeneity_notes: Mapped[str | None] = mapped_column(default=None)
    validation_anchor: Mapped[dict | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc)
    )

class Observation(Base):
    __tablename__ = "observations"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[str] = mapped_column()
    content: Mapped[str] = mapped_column()
    embedding: Mapped[list[float]] = mapped_column(Vector(3072))
    embedding_model: Mapped[str] = mapped_column()
    confidence_score: Mapped[float] = mapped_column()
    supersedes_id: Mapped[int | None] = mapped_column(default=None)
    entities: Mapped[list | None] = mapped_column(JSON, default=None)
    analysis_type: Mapped[str | None] = mapped_column(default=None)
    evidence: Mapped[list | None] = mapped_column(JSON, default=None)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))

class AnalysisImage(Base):
    __tablename__ = "analysis_images"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[str] = mapped_column()
    file_path: Mapped[str] = mapped_column()
    caption: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))