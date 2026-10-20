"""A planned campaign scenario (typed, from the copilot planner)."""

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..db.base import Base


class Scenario(Base):
    __tablename__ = "scenarios"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str] = mapped_column(String(32), nullable=False)
    budget_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    cost_per_contact_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    depth: Mapped[float | None] = mapped_column(Float, nullable=True)
    #: Which ranking model answered and how it was chosen (audit trail).
    model_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_selector: Mapped[str | None] = mapped_column(String(16), nullable=True)
    score_kind: Mapped[str | None] = mapped_column(String(40), nullable=True)
    research_mode: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    assumptions_used: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )