from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class AgentContext:
    """Shared state passed to every tool invocation."""

    dataframe: pd.DataFrame
    dataset_name: str
    column_names: list[str] = field(default_factory=list)
    shape: tuple[int, int] = (0, 0)

    def __post_init__(self) -> None:
        self.column_names = list(self.dataframe.columns)
        self.shape = (self.dataframe.shape[0], self.dataframe.shape[1])
