from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class AgentContext:
    """Shared state passed to every tool invocation."""

    dataframe: pd.DataFrame
    dataset_name: str
    column_names: list[str] = field(default_factory=list)
    shape: tuple[int, int] = (0, 0)

    # Citation registry — populated by citation_manager tool
    citation_store: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Step-by-step analysis audit trail — populated by log_analysis_step tool
    analysis_log: list[dict[str, Any]] = field(default_factory=list)

    # Deduplication cache: "tool_name::sorted_args_json" -> result_str
    _call_cache: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.column_names = list(self.dataframe.columns)
        self.shape = (self.dataframe.shape[0], self.dataframe.shape[1])
