from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


@dataclass
class BenchmarkRecord:
    """Standardized representation of a benchmark run."""

    identifier: str
    execution_passed: bool
    visual_success: bool
    overall_score: float
    meta: Dict[str, Any]

    @classmethod
    def from_result(cls, result_path: Path) -> "BenchmarkRecord":
        data = json.loads(result_path.read_text(encoding="utf-8"))
        outputs = data.get("outputs", {})

        visual_path = outputs.get("visual_evaluator") or outputs.get("visual_evaluator_json")
        evaluation: Dict[str, Any] = {}
        if visual_path and Path(visual_path).exists():
            try:
                evaluation = json.loads(Path(visual_path).read_text(encoding="utf-8"))
            except Exception:
                evaluation = {}

        semantic_block = evaluation.get("semantic_accuracy", {})
        metric = semantic_block.get("specification_adherence_score")
        try:
            metric_value = float(metric)
        except Exception:
            metric_value = 0.0

        recommendation = str(evaluation.get("final_recommendation", "")).lower()
        visual_success = recommendation == "accept"

        code_path = outputs.get("generated_plot_code")
        execution_passed = bool(code_path and Path(code_path).exists())

        identifier = data.get("query") or data.get("plot_type") or result_path.stem

        return cls(
            identifier=str(identifier),
            execution_passed=execution_passed,
            visual_success=visual_success,
            overall_score=metric_value,
            meta={
                "result_path": str(result_path),
                "evaluation_path": visual_path,
                "recommendation": recommendation,
            },
        )


def compute_metrics(records: Iterable[BenchmarkRecord]) -> Dict[str, float]:
    rec_list = list(records)
    if not rec_list:
        return {"EPR": 0.0, "VSR": 0.0, "OS": 0.0}

    total = len(rec_list)
    epr = sum(1 for r in rec_list if r.execution_passed) / total
    vsr = sum(1 for r in rec_list if r.visual_success) / total
    os_value = sum(r.overall_score for r in rec_list) / total
    return {"EPR": round(epr, 4), "VSR": round(vsr, 4), "OS": round(os_value, 4)}


def summarize_runs(result_paths: Iterable[Path]) -> Dict[str, Any]:
    records = [BenchmarkRecord.from_result(path) for path in result_paths]
    metrics = compute_metrics(records)
    return {
        "metrics": metrics,
        "records": [
            {
                "id": r.identifier,
                "execution_passed": r.execution_passed,
                "visual_success": r.visual_success,
                "overall_score": r.overall_score,
                **r.meta,
            }
            for r in records
        ],
    }
