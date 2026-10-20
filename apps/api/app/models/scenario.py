"""A planned campaign scenario (typed, from the copilot planner)."""

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
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
    assumptions_used: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )