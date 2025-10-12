from __future__ import annotations

import copy
import json
from typing import Any, Dict, Optional

from .llm_orchestrator import _parse_json_block
from utils.spec_adapter import figure_spec_to_paper_schema, paper_schema_summary
from utils.llm_client import LLMClient


def llm_aesthetic_stylist(
    client: LLMClient,
    query: str,
    design_spec: Dict[str, Any],
    mapping: Dict[str, Any],
    data_context: Dict[str, Any],
    *,
    policies: Optional[Dict[str, Any]] = None,
    search_context: Optional[Dict[str, Any]] = None,
    iteration: int = 1,
    feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    stage = "initial" if iteration == 1 else f"self_reflection_round_{iteration}"
    policies_json = json.dumps(policies or {}, ensure_ascii=False)
    design_summary = paper_schema_summary(figure_spec_to_paper_schema(mapping))
    feedback_json = json.dumps(feedback or {}, ensure_ascii=False)
    gallery_refs = json.dumps(
        (search_context or {}).get("references", []),
        ensure_ascii=False,
    )
    skeleton = {
        "style_spec_version": "1.0",
        "palette": {
            "mode": "categorical",
            "name": "tab10",
            "n_colors": 10,
            "colorblind_safe": True,
            "colors": [],
            "color_mapping": {},
            "contrast_ratio_min": 4.5,
        },
        "transparency": {
            "global_alpha": 0.85,
            "by_layer": {"points": 0.35, "bars": 0.65, "lines": 0.8},
            "overplotting_strategy": "auto_alpha",
        },
        "stat_annotations": {
            "enable_sample_size": True,
            "enable_confidence_interval": True,
            "confidence_level": 0.95,
            "trend": None,
            "group_comparison": None,
        },
        "layout_tuning": {
            "plt_style": "seaborn-v0_8-whitegrid",
            "grid": {"show": True, "which": "major", "alpha": 0.25},
            "ticks": {"rotation_x": 0, "format_y": "plain"},
            "constrained_layout": True,
        },
        "accessibility": {
            "font_min_px": 12,
            "legend_position": "best",
            "legend_framealpha": 0.85,
            "wcag_level": "AA",
        },
        "patches": {"ops": []},
        "notes": [],
    }
    prompt = f"""
You are Dr. Mei Tan, an aesthetic stylist specialising in Matplotlib and CoDA-style pipelines.
Stage: {stage}. Your task is to propose a structured StyleSpec JSON that harmonises the current design intent with global policies and accessibility needs.

Requirements:
- Honour the provided policies (transparency, color, statistics) where possible.
- Ensure palettes are colorblind-safe and annotate expectations for sample size / confidence intervals.
- Suggest overplotting mitigation strategies if density is high.
- Return JSON matching this skeleton (keep all keys, fill in values): {json.dumps(skeleton, ensure_ascii=False, indent=2)}
- Enumerate any additional comments in "notes".
- Include patch suggestions in patches.ops when specific adjustments are required.

Context:
Query: {query}
DesignSpec: {json.dumps(design_spec, ensure_ascii=False)}
FigureSpec Summary: {design_summary}
Policies: {policies_json}
Data Context: {json.dumps(data_context, ensure_ascii=False)}
Gallery References: {gallery_refs}
Feedback: {feedback_json}
"""
    out = client.chat([{"role": "user", "content": prompt}])
    return _parse_json_block(out)


def llm_aesthetic_stylist_refine(
    client: LLMClient,
    previous_stylespec: Dict[str, Any],
    *,
    feedback: Optional[Dict[str, Any]] = None,
    iteration: int = 1,
) -> Dict[str, Any]:
    stage = "initial" if iteration == 1 else f"self_reflection_round_{iteration}"
    skeleton = copy.deepcopy(previous_stylespec)
    skeleton.setdefault("patches", {"ops": []})
    prompt = f"""
You are Dr. Mei Tan revisiting a StyleSpec for iteration {iteration} ({stage}).
Refine the existing spec in response to the feedback.
Keep the original structure and update only the necessary fields. When applying targeted fixes, add entries to patches.ops using JSON Patch semantics ("add"/"replace"/"remove").

Existing StyleSpec: {json.dumps(previous_stylespec, ensure_ascii=False)}
Feedback: {json.dumps(feedback or {}, ensure_ascii=False)}

Return updated StyleSpec JSON only.
"""
    out = client.chat([{"role": "user", "content": prompt}])
    return _parse_json_block(out)
