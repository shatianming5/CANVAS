from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _coalesce_layers(mapping: Dict[str, Any]) -> List[Dict[str, Any]]:
    layers = mapping.get("layers", [])
    if isinstance(layers, list):
        return [lyr for lyr in layers if isinstance(lyr, dict)]
    return []


def _collect_layer_field(layer: Dict[str, Any], field: str, default: Any = None) -> Any:
    value = layer.get(field)
    if value is not None:
        return value
    roles = layer.get("roles") or {}
    if isinstance(roles, dict):
        return roles.get(field, default)
    return default


def _collect_transforms(layer: Dict[str, Any]) -> Dict[str, Any]:
    transform = layer.get("transform") or {}
    if not isinstance(transform, dict):
        return {"filters": [], "aggregations": {}, "derivations": []}
    filters = transform.get("filter") if isinstance(transform.get("filter"), list) else []
    derivations = transform.get("derive") if isinstance(transform.get("derive"), list) else []
    aggregations = transform.get("aggregate") if isinstance(transform.get("aggregate"), dict) else {}
    return {
        "filters": filters,
        "aggregations": aggregations,
        "derivations": derivations,
    }


def _merge_styling_hints(mapping: Dict[str, Any], design: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    hints = mapping.get("styling_hints")
    if isinstance(hints, dict):
        merged.update(hints)
    style = design.get("style")
    if isinstance(style, dict):
        merged.setdefault("global_style", {}).update(style)  # type: ignore[arg-type]
    figure = design.get("figure")
    if isinstance(figure, dict):
        merged.setdefault("figure", figure)
    axes_style = design.get("axes_style")
    if isinstance(axes_style, dict):
        merged.setdefault("axes_style", axes_style)
    legend_cfg = design.get("legend")
    if isinstance(legend_cfg, dict):
        merged.setdefault("legend", legend_cfg)
    return merged


def figure_spec_to_paper_schema(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Convert FigureSpec v1.2-like mapping into paper schema."""
    mapping = spec.get("mapping") or {}
    design = spec.get("design") or {}
    layers = _coalesce_layers(mapping)

    chart_type = mapping.get("chart_type")
    if not chart_type and layers:
        chart_type = layers[0].get("geom")

    data_mappings: Dict[str, Any] = {}
    dm = mapping.get("data_mappings") or {}
    if isinstance(dm, dict):
        data_mappings.update(dm)
    if layers:
        primary = layers[0].get("select") or {}
        if isinstance(primary, dict):
            data_mappings.setdefault("x", primary.get("x"))
            data_mappings.setdefault("y", primary.get("y"))
            data_mappings.setdefault("color", primary.get("color"))
            data_mappings.setdefault("size", primary.get("size"))
            data_mappings.setdefault("facet", primary.get("facet"))

    aggregations: Dict[str, Any] = {}
    filters: List[Any] = []
    derives: List[Any] = []
    goals: List[str] = []
    rationales: List[str] = []
    confidences: List[float] = []

    for layer in layers:
        transforms = _collect_transforms(layer)
        if transforms["aggregations"]:
            aggregations[layer.get("id", f"layer_{len(aggregations)}")] = transforms["aggregations"]
        if transforms["filters"]:
            filters.extend(transforms["filters"])
        if transforms["derivations"]:
            derives.extend(transforms["derivations"])
        purpose = layer.get("purpose") or {}
        if isinstance(purpose, dict):
            intent = purpose.get("intent")
            rationale = purpose.get("rationale")
            confidence = purpose.get("confidence")
            if isinstance(intent, str):
                goals.append(intent)
            if isinstance(rationale, str):
                rationales.append(rationale)
            if isinstance(confidence, (int, float)):
                confidences.append(float(confidence))

    styling_hints = _merge_styling_hints(mapping, design if isinstance(design, dict) else {})

    confidence = None
    if confidences:
        confidence = sum(confidences) / len(confidences)
    elif isinstance(mapping.get("confidence"), (int, float)):
        confidence = float(mapping["confidence"])

    return {
        "chart_type": chart_type,
        "data_mappings": data_mappings,
        "aggregations": aggregations,
        "filters": filters,
        "styling_hints": styling_hints,
        "transformations": {
            "derivations": derives,
        },
        "goal": goals,
        "rationale": rationales,
        "confidence": confidence,
    }


def paper_schema_summary(schema: Dict[str, Any]) -> str:
    parts: List[str] = []
    chart_type = schema.get("chart_type")
    if chart_type:
        parts.append(f"chart_type={chart_type}")
    dm = schema.get("data_mappings") or {}
    if dm:
        pairs = [f"{k}:{v}" for k, v in dm.items() if v]
        if pairs:
            parts.append("mappings[" + ", ".join(pairs) + "]")
    goals = schema.get("goal") or []
    if goals:
        parts.append("goals=" + ", ".join(goals if isinstance(goals, list) else [str(goals)]))
    return "; ".join(parts)
