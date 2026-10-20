"""A generated campaign brief with its validation report."""

from datetime import UTC, datetime

from sqlalchemy import DateTime, ForeignKey, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..db.base import Base


class Brief(Base):
    __tablename__ = "briefs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scenario_id: Mapped[int | None] = mapped_column(
        ForeignKey("scenarios.id"), nullable=True
    )
    prose: Mapped[str] = mapped_column(Text, nullable=False)
    validation_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    citations_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    scenario: Mapped["Scenario | None"] = relationship()  # noqa: F821