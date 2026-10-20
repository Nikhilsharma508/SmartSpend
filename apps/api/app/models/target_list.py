"""A recommended targeting list (output of optimize_budget)."""

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from ..db.base import Base


class TargetList(Base):
    __tablename__ = "target_lists"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    scenario_id: Mapped[int | None] = mapped_column(
        ForeignKey("scenarios.id"), nullable=True
    )
    target_list_id: Mapped[str] = mapped_column(String(64), nullable=False)
    n_targeted: Mapped[int] = mapped_column(Integer, nullable=False)
    expected_profit: Mapped[float] = mapped_column(Float, nullable=False)
    target_ids_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )