from __future__ import annotations

import json
import copy
from typing import Any, Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
from utils.llm_client import LLMClient
from utils.spec_adapter import figure_spec_to_paper_schema, paper_schema_summary

# === PATCH 1: helpers for v1.2 spec, registry, validation, and transforms ===
ALLOWED_GEOMS = {"bar", "line", "point", "area", "rect", "pie", "histogram", "box", "errorbar"}
ALLOWED_AXIS = {"primary", "secondary"}


def upgrade_to_v1_2(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Upgrade incoming mapping/spec to v1.2 with safe defaults."""
    sp = copy.deepcopy(spec)
    sp.setdefault("spec_version", "1.2")
    sp.setdefault("datasets", {})
    mapping = sp.setdefault("mapping", {})
    layers = mapping.setdefault("layers", [])

    for lyr in layers:
        lyr.setdefault("id", f"layer_{abs(hash(str(lyr))) % 10_000_000}")
        lyr.setdefault("geom", "line")
        if lyr["geom"] not in ALLOWED_GEOMS:
            raise ValueError(f"Unsupported geom: {lyr['geom']}")
        lyr.setdefault("alpha", 1.0)
        lyr.setdefault("axis", "primary")
        if lyr["axis"] not in ALLOWED_AXIS:
            raise ValueError(f"axis must be primary|secondary, got {lyr['axis']}")
        lyr.setdefault("source", "main")
        sel = lyr.setdefault("select", {})
        if "x" in lyr:
            sel.setdefault("x", lyr.pop("x"))
        if "y" in lyr:
            sel.setdefault("y", lyr.pop("y"))
        lyr.setdefault("roles", {})
        lyr.setdefault("transform", {"filter": [], "derive": [], "groupby": [], "aggregate": {}, "sort": []})
        lyr.setdefault("semantics", {"unit_y": None, "scale_y": "linear", "domain_y": None})
        lyr.setdefault("purpose", {"intent": "trend", "rationale": ""})
        lyr.setdefault("style", {})

    axes = mapping.setdefault("axes", {})
    axes.setdefault("x", {"type": "quantitative", "label": ""})
    axes.setdefault("y", {"type": "quantitative", "label": ""})
    axes.setdefault("secondary_y", {"enabled": False, "label": ""})

    design = sp.setdefault("design", {})
    design.setdefault("figure", {"width": 800, "height": 500, "dpi": 300, "grid": "y-major"})
    design.setdefault("axes_style", {"gap_mark": "none", "tick_size": 3.0, "line_width": 1.0, "ratio": [3, 1], "hspace": 0.05})
    design.setdefault("legend", {"loc": "best", "frameon": False})
    style = design.setdefault("style", {})
    style.setdefault("alpha", 1.0)
    style.setdefault("palette", {"name": "tab10", "reverse": False})
    style.setdefault("cmap", None)
    style.setdefault("linewidth", 1.5)
    style.setdefault("markersize", 36)
    style.setdefault("edgecolor", None)
    return sp


def reconcile_mapping_with_registry(
    spec: Dict[str, Any], dataset_registry: Optional[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[str]]:
    """Patch select fields to match registry columns when possible."""
    sp = copy.deepcopy(spec)
    warnings: List[str] = []
    if not dataset_registry:
        return sp, warnings

    per_src_cols: Dict[str, set] = {}
    for src, meta in dataset_registry.items():
        cols = meta.get("columns", {}) if isinstance(meta, dict) else {}
        per_src_cols[src] = set(cols.keys())

    for idx, lyr in enumerate(sp.get("mapping", {}).get("layers", [])):
        src = lyr.get("source", "main")
        sel = lyr.get("select", {}) or {}
        cols = per_src_cols.get(src, set())
        for key in ("x", "y", "color", "size", "facet"):
            col = sel.get(key)
            if col and col not in cols:
                warnings.append(f"layers[{idx}].select.{key}='{col}' not in dataset '{src}'")
    return sp, warnings


def validate_spec_vs_registry(spec: Dict[str, Any], dataset_registry: Optional[Dict[str, Any]]) -> None:
    """Strict validation before rendering; raise ValueError on mismatch."""
    if not dataset_registry:
        return
    errors: List[str] = []
    dsmap = dataset_registry
    for idx, lyr in enumerate(spec.get("mapping", {}).get("layers", [])):
        src = lyr.get("source", "main")
        sel = lyr.get("select", {}) or {}
        if src not in dsmap:
            errors.append(f"layers[{idx}].source '{src}' not found in dataset_registry")
            continue
        cols = (dsmap[src] or {}).get("columns", {})
        for key in ("x", "y", "color", "size", "facet"):
            col = sel.get(key)
            if col and col not in cols:
                errors.append(f"layers[{idx}].select.{key}='{col}' not in dataset '{src}' columns")
        ycol = sel.get("y")
        if ycol:
            dtype = str((cols.get(ycol, {}) or {}).get("dtype", ""))
            role_y = (lyr.get("roles", {}) or {}).get("y")
            if role_y == "measure" and not (dtype.startswith("float") or dtype.startswith("int")):
                errors.append(f"layers[{idx}].select.y='{ycol}' not numeric for measure role (dtype={dtype})")
        if lyr.get("axis") == "secondary":
            sec = spec.get("mapping", {}).get("axes", {}).get("secondary_y", {})
            if not (isinstance(sec, dict) and sec.get("label")):
                errors.append("secondary axis used but mapping.axes.secondary_y.label is missing")
    if errors:
        raise ValueError("Spec validation failed:\n- " + "\n- ".join(errors))


def _apply_filter(df: pd.DataFrame, filt: Dict[str, Any]) -> pd.DataFrame:
    expr = filt.get("expr")
    if expr:
        return df.query(expr)
    return df


def _apply_derive(df: pd.DataFrame, deriv: Dict[str, Any]) -> pd.DataFrame:
    name, expr = deriv.get("name"), deriv.get("expr")
    if not name or not expr:
        return df
    df = df.copy()
    df[name] = pd.eval(expr, engine="python", parser="pandas", target=df)
    return df


def apply_layer_transform(df: pd.DataFrame, transform: Dict[str, Any], cols_needed: List[str]) -> pd.DataFrame:
    """Execute minimal layer transforms for validation or rendering helpers."""
    if cols_needed:
        keep = [col for col in cols_needed if col in df.columns]
        if keep:
            df = df[keep].copy()
    for filt in transform.get("filter", []):
        df = _apply_filter(df, filt)
    for deriv in transform.get("derive", []):
        df = _apply_derive(df, deriv)
    groupby = transform.get("groupby", []) or []
    aggregate = transform.get("aggregate", {}) or {}
    if groupby and aggregate:
        df = df.groupby(groupby, dropna=False).agg(aggregate).reset_index()
    for sort in transform.get("sort", []):
        by = sort.get("by")
        if by and by in df.columns:
            df = df.sort_values(by=by, ascending=bool(sort.get("ascending", True)))
    return df


def _parse_json_block(text: str) -> Dict[str, Any]:
    # Extract the last JSON block in text
    last = None
    try:
        # Try direct parse
        return json.loads(text)
    except Exception:
        pass
    # Fallback: find code fences
    import re
    matches = re.findall(r"```(?:json)?\n(.*?)\n```", text, flags=re.S)
    for m in matches:
        try:
            last = json.loads(m)
        except Exception:
            continue
    if last is not None:
        return last
    # As a last resort, try to bracket substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    raise ValueError("Failed to parse JSON from LLM output")


def _parse_code_block(text: str) -> str:
    import re
    matches = re.findall(r"```python\n(.*?)\n```", text, flags=re.S)
    if matches:
        return matches[-1]
    # fallback to raw
    return text


def llm_query_analyzer(client: LLMClient, query: str) -> Dict[str, Any]:
    prompt = f"""
You are Dr. Sarah Chen, a visualization query expert. Analyze this query and create a master TODO list.

USER QUERY: "{query}"

Respond only with concise JSON matching this schema:
{{
  "interpreted_intent": "...",
  "visualization_type": "scatter|bar|line|histogram|box|heatmap|pie|area",
  "plotting_key_points": ["..."],
  "implementation_plan": [
    {{"step": 1, "action": "Load and prepare data", "details": "...", "functions": ["..."]}},
    {{"step": 2, "action": "Create base plot", "details": "...", "functions": ["..."]}},
    {{"step": 3, "action": "Apply formatting", "details": "...", "functions": ["..."]}},
    {{"step": 4, "action": "Finalize and save", "details": "...", "functions": ["..."]}}
  ],
  "global_todo_list": [
    {{"id": "todo_1", "task": "...", "agent": "data_processor|design_explorer|code_generator|debug_agent|visual_evaluator", "status": "pending", "priority": "high|medium|low"}}
  ],
  "success_criteria": ["..."]
}}
"""
    out = client.chat([{"role": "user", "content": prompt}])
    return _parse_json_block(out)


def llm_data_processor(client: LLMClient, query: str, plot_type: str, csv_path: str | None) -> Tuple[Dict[str, Any], pd.DataFrame]:
    data_section = (
        "A CSV is provided at path; load it with pandas.read_csv"
        if csv_path
        else "No data files provided. Create synthetic data suitable for the visualization."
    )
    prompt = f"""
You are Prof. Marcus Rodriguez, an expert in statistical data preparation for visualization.
Return only JSON plus one separate python code block (constructing a DataFrame named df) if needed.

Context:
- Query: "{query}"
- Target plot type hint: "{plot_type}"
- Data source: "{'csv' if csv_path else 'synthetic'}"

Tasks:
1) Identify required columns and minimal transformations.
2) Provide a compact data summary (dtypes, cardinalities, quantiles for numeric).
3) Detect extremes/outliers and, if appropriate, suggest a broken-axis range [[l1,h1],[l2,h2]].
4) Build dataset/field registry for mapping.

Respond only with JSON:
{{
  "spec_version": "1.2",
  "processing_steps": ["..."],
  "insights": {{
    "key_columns": ["..."],
    "aggregations_needed": ["..."],
    "quality_issues": ["..."],
    "data_summary": {{
      "columns": {{"col": {{"dtype": "...", "unique": 0}}}},
      "numeric_overview": {{"col": {{"min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0}}}}
    }},
    "break_suggestion": null
  }},
  "dataset_registry": {{
    "main": {{"columns": {{"column_name": {{"dtype": "string|float|int|datetime", "role": "dimension|measure|temporal", "unit": null, "agg_default": "sum|mean|none"}}}}}}
  }},
  "field_registry": {{"alias": "main.column_name"}},
  "visualization_hint": "{plot_type}",
  "_warnings": []
}}

Additionally, write Python code that constructs a pandas DataFrame variable named df.
Return that code in a separate python code block (```python ... ```).
"""
    out = client.chat([{"role": "user", "content": prompt}])
    try:
        meta = _parse_json_block(out)
    except Exception:
        meta = {
            "spec_version": "1.2",
            "processing_steps": [],
            "insights": {
                "key_columns": [],
                "aggregations_needed": [],
                "quality_issues": [],
                "data_summary": {},
                "break_suggestion": None,
            },
            "dataset_registry": {},
            "field_registry": {},
            "visualization_hint": plot_type,
            "_warnings": ["failed to parse JSON, using fallback"],
        }
    code = _parse_code_block(out)

    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        local_vars: Dict[str, Any] = {"pd": pd, "np": np}
        try:
            exec(code, local_vars, local_vars)
            df = local_vars.get("df")
            if df is None or not isinstance(df, pd.DataFrame):
                raise RuntimeError("LLM code did not define a DataFrame variable df")
        except Exception as exc:
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            regions = ["North", "South", "East", "West"]
            rng = np.random.default_rng(42)
            rows = []
            for month in months:
                for region in regions:
                    rows.append({
                        "month": month,
                        "region": region,
                        "sales": float(rng.normal(100, 20)),
                        "profit": float(rng.normal(30, 10)),
                    })
            df = pd.DataFrame(rows)
            meta.setdefault("_warnings", []).append(f"LLM code failed: {exc}")

    meta.setdefault("spec_version", "1.2")
    meta.setdefault("processing_steps", [])
    insights = meta.setdefault("insights", {})
    insights.setdefault("key_columns", [])
    insights.setdefault("aggregations_needed", [])
    insights.setdefault("quality_issues", [])
    insights.setdefault("data_summary", {})
    insights.setdefault("break_suggestion", None)
    meta.setdefault("dataset_registry", {})
    meta.setdefault("field_registry", {})
    meta.setdefault("visualization_hint", plot_type)
    meta.setdefault("_warnings", [])
    meta["generated_code"] = code
    return meta, df

def llm_viz_mapping(client: LLMClient, query: str, df: pd.DataFrame, plot_type: str) -> Dict[str, Any]:
    sample = df.head(2).to_dict(orient="records")
    shape = f"{df.shape[0]} rows and {df.shape[1]} columns"
    cols = list(map(str, df.columns.tolist()))
    prompt = f"""
You are Dr. Sarah Kim, a mapping planner for FigureSpec v1.2 (layers-first).
Return only JSON. Every layer MUST include source, select, roles, transform, semantics, purpose.
Always populate design.style (alpha, palette/cmap, linewidth, markersize, edgecolor) and per-layer style overrides when helpful.
Add mapping.styling_hints summarizing axis labels/titles/colors and provide mapping.confidence between 0 and 1.
If multi-geometry or secondary axis is required, encode it in layers and axes.secondary_y.
If a broken axis is appropriate, set mapping.axes.y.breaks = [[l1,h1],[l2,h2]].

USER QUERY: "{query}"

AVAILABLE DATA (preview):
- Shape: {shape}
- Columns: {cols}
- Sample (first 2 rows): {json.dumps(sample, ensure_ascii=False, default=str)}

Respond only with JSON:
{{
  "spec_version": "1.2",
  "mapping": {{
    "layers": [
      {{
        "id": "layer_sales",
        "geom": "bar|line|point|area|rect|pie|histogram|box|errorbar",
        "source": "main",
        "select": {{"x": "...", "y": "...", "color": "..."}},
        "roles": {{"x": "temporal|dimension", "y": "measure"}},
        "transform": {{"filter": [], "derive": [], "groupby": ["..."], "aggregate": {{"ycol": "sum"}}, "sort": []}},
        "semantics": {{"unit_y": "USD|null", "scale_y": "linear|log", "domain_y": null}},
        "purpose": {{"intent": "trend|comparison|reference|threshold|uncertainty|context", "rationale": "..."}},
        "alpha": 1.0,
        "legend": "...",
        "axis": "primary|secondary",
        "style": {{
          "alpha": 0.8,
          "palette": {{"name": "Set2", "reverse": false}},
          "cmap": null,
          "linewidth": 1.5,
          "markersize": 32,
          "edgecolor": null
        }}
      }}
    ],
    "encodings": {{"color_by": null, "facet": null}},
    "axes": {{
      "x": {{"type": "temporal|quantitative|nominal", "label": "..."}},
      "y": {{"type": "quantitative", "label": "...", "breaks": null}},
      "secondary_y": {{"enabled": false, "label": ""}}
    }},
    "chart_type": "{plot_type}",
    "data_mappings": {{"x_axis": "...", "y_axis": "...", "color": "..."}}
  }},
  "design": {{
    "figure": {{"width": 800, "height": 500, "dpi": 300, "grid": "y-major"}},
    "legend": {{"loc": "best", "frameon": false}},
    "axes_style": {{"tick_size": 3.0, "line_width": 1.0, "hspace": 0.05}},
    "style": {{
      "alpha": 0.85,
      "palette": {{"name": "tab10", "reverse": false}},
      "cmap": null,
      "linewidth": 1.5,
      "markersize": 36,
      "edgecolor": null
    }}
  }},
  "_warnings": []
}}
"""
    out = client.chat([{"role": "user", "content": prompt}])
    resp = _parse_json_block(out)
    return upgrade_to_v1_2(resp)

def llm_search_agent(client: LLMClient, plot_type: str) -> Dict[str, Any]:
    prompt = f"""
You are Dr. Elaine Porter, the Search Agent responsible for gathering authoritative Matplotlib examples.
For the plot type "{plot_type}", consult the official Matplotlib Gallery / Plot Types documentation and optionally Python Graph Gallery.
Return JSON with two keys:
{{
  "references": [
    {{"title": "Official Example Name", "url": "https://matplotlib.org/...", "reason": "Why it is relevant"}}
  ],
  "example_code": "```python\\n...matplotlib example that can run locally...\\n```"
}}
Rules:
- Include at least one official Matplotlib documentation URL.
- Do not invent new Matplotlib APIs.
- The example code must be executable with matplotlib.pyplot imported as plt and conclude by saving or showing the figure.
"""
    out = client.chat([{"role": "user", "content": prompt}])
    data = _parse_json_block(out)
    code = data.get("example_code", "")
    data["example_code"] = _parse_code_block(code if isinstance(code, str) else "")
    refs = data.get("references")
    if not isinstance(refs, list):
        data["references"] = []
    return data


def llm_design_explorer(
    client: LLMClient,
    original_query: str,
    interpreted_intent: str,
    visualization_type: str,
    data_characteristics_json: Dict[str, Any],
    extras: Optional[Dict[str, Any]] = None,
    feedback: Optional[Dict[str, Any]] = None,
    iteration: int = 1,
) -> Dict[str, Any]:
    stage = "initial" if iteration == 1 else "self_reflection"
    extras_json = json.dumps(extras or {}, ensure_ascii=False)
    feedback_json = json.dumps(feedback or {}, ensure_ascii=False)
    prompt = f"""
You are Isabella Nakamura, create comprehensive design specifications.

Query Analysis:
* Original Query: "{original_query}"
* Interpreted Intent: "{interpreted_intent}"
* Visualization Type: "{visualization_type}"

Data Characteristics:
{json.dumps(data_characteristics_json, ensure_ascii=False)}

Pipeline Context:
- Iteration Stage: "{stage}"
- Extras: {extras_json}
- Feedback from evaluator/debug/quality monitors: {feedback_json}

Ensure the color_strategy section names concrete palette/cmap choices, desired alpha ranges, and marker/line sizing guidance.
Include self_reflection_notes when feedback is provided, and propose adjustments for code_generator_actions.
Respond only with JSON with keys: design_objectives, target_audience, visual_hierarchy, color_strategy, layout_principles, typography_requirements, interaction_design, technical_constraints, innovation_opportunities, design_confidence, self_reflection_notes, code_generator_actions.
"""
    out = client.chat([{"role": "user", "content": prompt}])
    return _parse_json_block(out)


def llm_code_generator(
    client: LLMClient,
    context_json: Dict[str, Any],
    design_spec_json: Dict[str, Any],
    n_rows: int,
    n_cols: int,
    data_columns: List[str],
    data_quality_score: float,
    style_spec: Dict[str, Any],
    mapping: Dict[str, Any],
    paper_schema: Dict[str, Any],
    search_context: Dict[str, Any],
    iteration: int = 1,
    feedback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    stage = "initial" if iteration == 1 else f"self_reflection_round_{iteration}"
    search_refs = json.dumps(search_context.get("references", []), ensure_ascii=False)
    example_code = search_context.get("example_code") or ""
    feedback_json = json.dumps(feedback or {}, ensure_ascii=False)
    style_spec_json = json.dumps(style_spec or {}, ensure_ascii=False)
    prompt = f"""
You are Alex Thompson. Produce a concise, actionable matplotlib code generation plan based on FigureSpec v1.2.
Current pipeline stage: {stage}. If feedback highlights issues, address them explicitly.
Reference official patterns from these sources: {search_refs}.
Incorporate insights from the provided gallery snippet when useful.
Return only JSON. Focus on layered rendering order, secondary axis routing, broken-axis helper usage, and per-layer data_plan. Highlight how global design.style, StyleSpec directives (palette/transparency/stat_annotations/layout/accessibility), and layer.style values (alpha/palette/cmap/linewidth/markersize/edgecolor) should feed into the plotting helpers. Include actions that respond to code_generator_actions from Design Explorer.

Context: {json.dumps(context_json, ensure_ascii=False)}
MappingSpec: {json.dumps(mapping, ensure_ascii=False)}
DesignSpec: {json.dumps(design_spec_json, ensure_ascii=False)}
PaperSchema: {json.dumps(paper_schema, ensure_ascii=False)}
StyleSpec: {style_spec_json}
Data: shape=({n_rows},{n_cols}), columns={data_columns}, quality_score={data_quality_score}
Feedback: {feedback_json}
ExampleSnippet:
```python
{example_code}
```

Respond only with JSON:
{{
  "spec_version": "1.2",
  "code_architecture": ["import/rcparams", "figure/axes init", "broken-axis setup if needed", "layered drawing", "labels/legend", "export"],
  "data_plan": [{{"layer_id": "...", "source": "main", "steps": [{{"op": "select", "cols": ["..."]}}]}}],
  "matplotlib_approach": {{"theme": "...", "broken_axis": "...", "secondary_axis": "..."}},
  "rendering_order": [{{"layer_index": 0, "geom": "bar|line|point|area|rect|pie|histogram|box|errorbar", "target_axis": "primary|secondary", "alpha": 1.0, "legend": "...", "style": {{"alpha": "...", "palette": "...", "cmap": "...", "linewidth": "...", "markersize": "...", "edgecolor": "..."}}}}],
  "helpers_needed": ["apply_layer_transform", "draw_layers", "broken_axis_y"],
  "quality_requirements": ["no legend overlap", "secondary axis labeled", "tight layout", "style directives applied"],
  "_warnings": []
}}
"""
    out = client.chat([{"role": "user", "content": prompt}])
    return _parse_json_block(out)

def llm_debug_agent(client: LLMClient, code: str, error_message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ctx = json.dumps(context or {}, ensure_ascii=False)
    prompt = f"""
You are Jordan Martinez, a debugging specialist. Fix this Python matplotlib code.

CURRENT CODE:\n```python\n{code}\n```
ERROR MESSAGE:\n{error_message}

Contextual signals (quality issues, search references, iteration notes): {ctx}

Respond only with JSON:
{{
  "error_type": "visual_overlap|syntax|runtime|import|logic",
  "root_cause": "...",
  "overlapping_elements": ["..."],
  "missing_requirements": "...",
  "error_location": "...",
  "fixed_code": "your fixed matplotlib code",
  "confidence": 0.0
}}
"""
    out = client.chat([{"role": "user", "content": prompt}])
    return _parse_json_block(out)


def llm_visual_evaluator(
    client: LLMClient,
    query: str,
    key_points: List[str],
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    paper_schema: Dict[str, Any],
    design_spec: Dict[str, Any],
    image_payload: Dict[str, Any],
    iteration: int,
) -> Dict[str, Any]:
    data_ctx = {
        "shape": f"{df.shape[0]}x{df.shape[1]}",
        "columns": list(map(str, df.columns.tolist())),
        "dtypes": {str(col): str(df[col].dtype) for col in df.columns},
    }
    iter_label = "initial" if iteration == 1 else f"iteration_{iteration}"
    prompt = f"""
You are Dr. Elena Vasquez. Evaluate the visualization against the CoDA paper spec, using the provided figure (base64-encoded preview) and metadata.
Verify semantic accuracy, mapping compliance, layout integrity, accessibility, and whether design recommendations were followed.
Return JSON with the following structure:
{{
  "semantic_accuracy": {{
    "data_query_match": 0.0,
    "mathematical_correctness": 0.0,
    "visual_element_compliance": 0.0,
    "layout_structure_match": 0.0,
    "specification_adherence_score": 0.0
  }},
  "quality_assessment": {{"strengths": ["..."], "risks": ["..."]}},
  "requirement_analysis": {{
    "key_points_missing": ["..."],
    "blocking_issues": ["..."],
    "repair_suggestions": ["..."]
  }},
  "accessibility_check": {{"color_contrast": "pass|warn|fail", "annotations": "pass|warn|fail", "notes": ["..."]}},
  "final_recommendation": "accept|iterate|reject",
  "_warnings": []
}}
Scores must be floats between 0 and 1. Reference iteration label "{iter_label}" when proposing follow-up actions.

Query: {query}
Key Points: {json.dumps(key_points, ensure_ascii=False)}
Data Context: {json.dumps(data_ctx, ensure_ascii=False)}
FigureSpec: {json.dumps(mapping, ensure_ascii=False)}
PaperSchema: {json.dumps(paper_schema, ensure_ascii=False)}
DesignSpec: {json.dumps(design_spec, ensure_ascii=False)}
Image Payload: {json.dumps(image_payload)}
"""
    out = client.chat([{"role": "user", "content": prompt}])
    return _parse_json_block(out)


def llm_staged_visual_evaluator(
    client: LLMClient,
    query: str,
    key_points: List[str],
    df: pd.DataFrame,
    mapping: Dict[str, Any],
    paper_schema: Dict[str, Any],
    design_spec: Dict[str, Any],
    image_payload: Dict[str, Any],
    iteration: int,
    *,
    policies: Optional[Dict[str, Any]] = None,
    todo: Optional[Dict[str, Any]] = None,
    quality_notes: Optional[List[str]] = None,
    style_spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    data_ctx = {
        "shape": f"{df.shape[0]}x{df.shape[1]}",
        "columns": list(map(str, df.columns.tolist())),
        "dtypes": {str(col): str(df[col].dtype) for col in df.columns},
    }
    iter_label = "initial" if iteration == 1 else f"iteration_{iteration}"
    stage_brief = {
        "L1": "Orchestration: global styling policies, legend strategy, run-level coherence, respectful of declared transparency/color/statistics policies.",
        "L2": "Composition: subplot layout, axis pairing, facet structure, secondary axes labeling, avoiding overcrowding.",
        "L3": "Calibration: data-to-visual mapping fidelity, statistical summaries, color scales (centering, ranges), reproducibility of transformations.",
        "L4": "Polish: tick formatting, annotations placement, contrast, alpha fine tuning, clarity of callouts and captions.",
    }
    skeleton = {
        "overall": {
            "specification_adherence_score": 0.0,
            "target_stage": "L1",
            "rationale": "",
            "next_actions": [],
            "risks": [],
        },
        "stages": {
            "L1": {
                "score": 0.0,
                "risk": "low",
                "issues": [],
                "patch_suggestions": {"ops": []},
                "notes": [],
            },
            "L2": {
                "score": 0.0,
                "risk": "low",
                "issues": [],
                "patch_suggestions": {"ops": []},
                "notes": [],
            },
            "L3": {
                "score": 0.0,
                "risk": "low",
                "issues": [],
                "patch_suggestions": {"ops": []},
                "notes": [],
            },
            "L4": {
                "score": 0.0,
                "risk": "low",
                "issues": [],
                "patch_suggestions": {"ops": []},
                "notes": [],
            },
        },
        "semantic_accuracy": {
            "data_query_match": 0.0,
            "mathematical_correctness": 0.0,
            "visual_element_compliance": 0.0,
            "layout_structure_match": 0.0,
            "specification_adherence_score": 0.0,
        },
        "accessibility": {"color_contrast": "pass", "annotations": "pass", "notes": []},
        "image_diagnostics": {"warnings": [], "observations": []},
    }
    prompt = f"""
You are Dr. Elena Vasquez. Perform a four-layer review of the visualization according to the CoDA staged evaluator:
- L1 Orchestration: {stage_brief['L1']}
- L2 Composition: {stage_brief['L2']}
- L3 Calibration: {stage_brief['L3']}
- L4 Polish: {stage_brief['L4']}

Return JSON matching exactly this schema (fill in values; keep all keys):
{json.dumps(skeleton, ensure_ascii=False, indent=2)}

Guidelines:
- Scores must be floats between 0 and 1.
- risk must be one of "low", "medium", or "high".
- patch_suggestions.ops should be a JSON Patch style list (can be empty) tailored to the specific stage.
- \"target_stage\" should reference the layer with the most critical issues this iteration.
- Always align recommendations with the stated policies and TODO plan when present.
- Reference iteration label "{iter_label}" when discussing improvements.

Context:
Query: {query}
Key Points: {json.dumps(key_points, ensure_ascii=False)}
Policies: {json.dumps(policies or {}, ensure_ascii=False)}
TODO Plan: {json.dumps(todo or {}, ensure_ascii=False)}
Quality Notes: {json.dumps(quality_notes or [], ensure_ascii=False)}
StyleSpec: {json.dumps(style_spec or {}, ensure_ascii=False)}
Data Context: {json.dumps(data_ctx, ensure_ascii=False)}
FigureSpec: {json.dumps(mapping, ensure_ascii=False)}
PaperSchema: {json.dumps(paper_schema, ensure_ascii=False)}
DesignSpec: {json.dumps(design_spec, ensure_ascii=False)}
Image Payload: {json.dumps(image_payload)}
"""
    out = client.chat([{"role": "user", "content": prompt}])
    return _parse_json_block(out)
