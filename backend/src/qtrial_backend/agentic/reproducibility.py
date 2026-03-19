"""
Reproducibility Log Builder.

Collects audit data incrementally during a pipeline run and finalises it into
a ReproducibilityLog that is serialised to outputs/{run_id}_reproducibility.json.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from qtrial_backend.agentic.schemas import (
    ClinicalSearchTerm,
    LiteratureQueryRecord,
    LLMCallRecord,
    ReproducibilityLog,
    SynthesisQualityScore,
    ToolCallRecord,
)

_OUTPUTS_DIR = Path("outputs")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class ReproducibilityLogBuilder:
    """
    Incrementally collects audit records during a single pipeline run.

    Usage:
        builder = ReproducibilityLogBuilder(run_id, study_context, seed)
        builder.add_llm_call(...)
        builder.add_literature_query(...)
        log = builder.finalise(synthesis_quality_score)
    """

    def __init__(self, run_id: str, study_context: str, seed: int) -> None:
        self.run_id = run_id
        self.study_context = study_context
        self.seed = seed
        self._llm_calls: list[LLMCallRecord] = []
        self._literature_queries: list[LiteratureQueryRecord] = []
        self._csts: list[ClinicalSearchTerm] = []
        self._tool_calls: list[ToolCallRecord] = []
        self._call_counter = 0

    def add_llm_call(
        self,
        stage: str,
        model: str,
        temperature: float,
        prompt: str,
        response: str,
        seed: int | None = None,
    ) -> None:
        self._call_counter += 1
        self._llm_calls.append(
            LLMCallRecord(
                call_id=f"{self.run_id}_llm_{self._call_counter:03d}",
                stage=stage,
                model=model,
                temperature=temperature,
                seed=seed,
                prompt_hash=_sha256(prompt),
                response_hash=_sha256(response),
            )
        )

    def add_literature_query(self, record: LiteratureQueryRecord) -> None:
        self._literature_queries.append(record)

    def add_literature_queries(self, records: list[LiteratureQueryRecord]) -> None:
        self._literature_queries.extend(records)

    def add_cst(self, cst: ClinicalSearchTerm) -> None:
        self._csts.append(cst)

    def add_csts(self, csts: list[ClinicalSearchTerm]) -> None:
        self._csts.extend(csts)

    def add_tool_call(self, record: ToolCallRecord) -> None:
        self._tool_calls.append(record)

    def add_tool_calls(self, records: list[ToolCallRecord]) -> None:
        self._tool_calls.extend(records)

    def finalise(
        self, synthesis_quality_score: SynthesisQualityScore | None = None
    ) -> ReproducibilityLog:
        """Build and persist the ReproducibilityLog."""
        log = ReproducibilityLog(
            run_id=self.run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            study_context=self.study_context,
            seed=self.seed,
            llm_calls=list(self._llm_calls),
            literature_queries=list(self._literature_queries),
            clinical_search_terms=list(self._csts),
            tool_call_log=list(self._tool_calls),
            synthesis_quality_score=synthesis_quality_score,
        )
        self._persist(log)
        return log

    def _persist(self, log: ReproducibilityLog) -> None:
        try:
            _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            path = _OUTPUTS_DIR / f"{self.run_id}_reproducibility.json"
            path.write_text(
                json.dumps(log.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception:
            # Persistence failure must never abort the pipeline
            pass
