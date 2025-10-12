from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd
from PIL import Image

from utils.io_utils import ensure_dir, write_json, write_text
from utils.config import apply_env_defaults
from utils.patch_ops import apply_patch_ops
from agents.aesthetic_stylist import (
    llm_aesthetic_stylist,
    llm_aesthetic_stylist_refine,
)
from agents.llm_orchestrator import (
    llm_query_analyzer,
    llm_data_processor,
    llm_viz_mapping,
    llm_search_agent,
    llm_design_explorer,
    llm_code_generator,
    llm_debug_agent,
    llm_staged_visual_evaluator,
    upgrade_to_v1_2,
    reconcile_mapping_with_registry,
    validate_spec_vs_registry,
)
from utils.llm_client import LLMClient
from agents.report_builder import build_html_report
from utils.spec_adapter import figure_spec_to_paper_schema, paper_schema_summary
from utils.web_search import official_references
from utils.style_rules import auto_tune_alpha, adjust_local_contrast, sanitize_backgrounds, prune_insets
from utils.stats_utils import annotate_sample_size, add_errorbars
from utils.style_spec import StyleSpec




def _infer_role_and_defaults(series: pd.Series) -> tuple[str, str | None]:
    dtype = series.dtype
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "temporal", "none"
    if pd.api.types.is_numeric_dtype(dtype):
        return "measure", "sum"
    return "dimension", "none"


def _ensure_registry_from_df(df: pd.DataFrame, registry: Dict[str, Any]) -> Dict[str, Any]:
    registry = registry.copy() if isinstance(registry, dict) else {}
    if not registry:
        registry["main"] = {"columns": {}}
    if "main" not in registry or not isinstance(registry["main"], dict):
        registry["main"] = {"columns": {}}
    registry["main"].setdefault("columns", {})
    cols_meta = registry["main"]["columns"]
    for col in df.columns:
        if col in cols_meta:
            continue
        role, agg_default = _infer_role_and_defaults(df[col])
        cols_meta[col] = {
            "dtype": str(df[col].dtype),
            "role": role,
            "unit": None,
            "agg_default": agg_default,
        }
    return registry



HEATMAP_CELL_ORDER = ["skmel19", "wm266_4", "skmel28", "uacc62", "wm983b", "g361", "wm88"]
HEATMAP_DRUG_ORDER = ["plx4720", "azd6244", "plx_azd", "vtx_11e"]


def _order_heatmap_columns(columns: list[str]) -> list[str]:
    def _order_key(col: str) -> tuple[int, int, str]:
        base = col.replace("_rel_zc", "")
        parts = base.split("_", 1)
        cell = parts[0]
        drug = parts[1] if len(parts) > 1 else ""
        cell_idx = HEATMAP_CELL_ORDER.index(cell) if cell in HEATMAP_CELL_ORDER else len(HEATMAP_CELL_ORDER)
        drug_idx = HEATMAP_DRUG_ORDER.index(drug) if drug in HEATMAP_DRUG_ORDER else len(HEATMAP_DRUG_ORDER)
        return (cell_idx, drug_idx, base)

    return sorted(columns, key=_order_key)


def _format_heatmap_condition(col: str) -> str:
    base = col.replace("_rel_zc", "")
    parts = base.split("_", 1)
    if len(parts) == 1:
        return parts[0].upper()
    cell, drug = parts
    cell_label = cell.upper().replace("SKMEL", "SKMEL").replace("WM", "WM")
    drug_map = {
        "plx4720": "PLX4720",
        "azd6244": "AZD6244",
        "plx_azd": "PLX+AZD",
        "vtx_11e": "VTX-11E",
    }
    drug_label = drug_map.get(drug, drug.replace("_", "+").upper())
    return f"{cell_label} - {drug_label}"


def _prepare_heatmap_dataframe(df: pd.DataFrame) -> pd.DataFrame | None:
    if "symbol" not in df.columns:
        return None
    condition_cols = [c for c in df.columns if c.endswith("_rel_zc")]
    if not condition_cols:
        return None
    ordered_cols = _order_heatmap_columns(condition_cols)
    base_cols = [col for col in ["symbol", "orf_class"] if col in df.columns]
    melt_df = df.melt(
        id_vars=base_cols,
        value_vars=ordered_cols,
        var_name="condition_key",
        value_name="z_score",
    )
    melt_df = melt_df.dropna(subset=["z_score"])
    counts = (df[ordered_cols] >= 4).sum(axis=1)
    symbol_counts = dict(zip(df["symbol"], counts))
    melt_df["condition_count"] = melt_df["symbol"].map(symbol_counts)
    melt_df = melt_df[melt_df["condition_count"] >= 2]
    if melt_df.empty:
        return melt_df
    label_categories = [_format_heatmap_condition(c) for c in ordered_cols]
    melt_df["condition"] = melt_df["condition_key"].map(_format_heatmap_condition)
    melt_df["condition"] = pd.Categorical(melt_df["condition"], categories=label_categories, ordered=True)
    melt_df["max_z_score"] = melt_df.groupby("symbol")["z_score"].transform("max")
    melt_df = melt_df.sort_values(["max_z_score", "symbol", "condition"], ascending=[False, True, True]).reset_index(drop=True)
    keep_cols = ["symbol", "condition", "z_score", "condition_count", "max_z_score"]
    if "orf_class" in melt_df.columns:
        keep_cols.append("orf_class")
    return melt_df[keep_cols]

_DF_REASSIGN_PATTERN = re.compile(r"\bdf\s*=\s*pd\.DataFrame", re.IGNORECASE)
_SAMPLE_PHRASE_PATTERN = re.compile(r"Sample DataFrame", re.IGNORECASE)


def _validate_generated_code(code: str) -> None:
    """Raise ValueError if generated code violates basic constraints (e.g., recreating df)."""
    if _DF_REASSIGN_PATTERN.search(code):
        raise ValueError(
            "Generated code redefines 'df' via pd.DataFrame; please use the provided dataframe from the pipeline."
        )
    if _SAMPLE_PHRASE_PATTERN.search(code):
        raise ValueError(
            "Generated code contains sample DataFrame placeholders; please operate directly on the provided dataframe."
        )
    if _RANDOM_DATA_PATTERN.search(code):
        raise ValueError("Generated code uses np.random*; do not fabricate data. Use the provided df.")
    if _INSET_PATTERN.search(code):
        raise ValueError("Generated code adds inset_axes; avoid insets unless explicitly requested.")


def _merge_styles(global_style: Dict[str, Any] | None, layer_style: Dict[str, Any] | None) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(global_style or {})
    if layer_style:
        for key, value in layer_style.items():
            if value is not None:
                merged[key] = value
    return merged


def _extract_style_context(mapping_spec: Dict[str, Any]) -> Dict[str, Any]:
    design = mapping_spec.get("design", {}) or {}
    if not design:
        nested_design = mapping_spec.get("mapping", {}).get("design")
        if isinstance(nested_design, dict):
            design = nested_design
    global_style = design.get("style", {}) or {}
    layers_styles: Dict[str, Dict[str, Any]] = {}
    for layer in mapping_spec.get("mapping", {}).get("layers", []):
        lid = layer.get("id") or f"layer_{len(layers_styles)}"
        lid = str(lid)
        layer_style = layer.get("style", {}) or {}
        layers_styles[lid] = _merge_styles(global_style, layer_style)
    return {"global": global_style, "layers": layers_styles}


def _resolve_palette(values, palette_cfg: Dict[str, Any] | None):
    if not palette_cfg:
        return None
    if isinstance(palette_cfg, dict):
        explicit_map = palette_cfg.get("map")
        if isinstance(explicit_map, dict):
            return explicit_map
        name = palette_cfg.get("name")
        reverse = palette_cfg.get("reverse", False)
        if name:
            import matplotlib.pyplot as plt  # noqa: PLC0415

            cmap = plt.get_cmap(name)
            if reverse and hasattr(cmap, "reversed"):
                cmap = cmap.reversed()
            uniques = pd.Series(list(values)).dropna().unique().tolist()  # type: ignore[arg-type]
            if not uniques:
                return {}
            if len(uniques) == 1:
                return {uniques[0]: cmap(0.5)}
            colors = [cmap(i / (len(uniques) - 1)) for i in range(len(uniques))]
            return dict(zip(uniques, colors))
    return None


def _resolve_cmap(cmap_cfg: Any, *, center: float | None = None):
    if not cmap_cfg:
        return {"cmap": None, "norm": None}
    from matplotlib import cm, colors  # noqa: PLC0415

    name = None
    reverse = False
    explicit_center = center
    if isinstance(cmap_cfg, str):
        name = cmap_cfg
    elif isinstance(cmap_cfg, dict):
        name = cmap_cfg.get("name") or cmap_cfg.get("value") or cmap_cfg.get("id")
        reverse = bool(cmap_cfg.get("reverse", False))
        explicit_center = cmap_cfg.get("center", center)
    cmap = cm.get_cmap(name) if name else cm.get_cmap("viridis")
    if reverse and hasattr(cmap, "reversed"):
        cmap = cmap.reversed()
    norm = None
    if explicit_center is not None:
        norm = colors.TwoSlopeNorm(vcenter=explicit_center)
    return {"cmap": cmap, "norm": norm}


def _encode_image_payload(image_path: Path, max_size: tuple[int, int] = (640, 640)) -> Dict[str, Any]:
    """Downsample and base64-encode the generated figure for the evaluator."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGBA")
            img.thumbnail(max_size, Image.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            payload = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return {
                "path": str(image_path.resolve()),
                "format": "png",
                "base64": payload,
            }
    except Exception:
        try:
            raw = image_path.read_bytes()
            payload = base64.b64encode(raw).decode("utf-8")
            return {
                "path": str(image_path.resolve()),
                "format": image_path.suffix.lstrip(".") or "png",
                "base64": payload,
            }
        except Exception:
            return {"path": str(image_path.resolve()), "error": "encode_failed"}


def _parse_todo_plan(todo_list: Any) -> Dict[str, List[Dict[str, Any]]]:
    plan: Dict[str, List[Dict[str, Any]]] = {}
    if not isinstance(todo_list, list):
        return plan
    priority_map = {"high": 0, "medium": 1, "low": 2}
    for item in todo_list:
        if not isinstance(item, dict):
            continue
        agent = str(item.get("agent") or "").strip().lower()
        if not agent:
            continue
        priority = str(item.get("priority") or "medium").lower()
        score = priority_map.get(priority, 1)
        entry = dict(item)
        entry["_priority_score"] = score
        plan.setdefault(agent, []).append(entry)
    for agent, items in plan.items():
        items.sort(key=lambda x: (x.get("_priority_score", 1), x.get("id", "")))
    return plan


def _compute_data_quality_score(quality_issues: List[str]) -> float:
    if not quality_issues:
        return 1.0
    unique = len(set(q for q in quality_issues if isinstance(q, str)))
    return max(0.2, 1.0 - 0.1 * unique)


def _extract_code_block(text: str) -> str:
    import re

    matches = re.findall(r"```python\s+(.*?)\s+```", text, flags=re.S | re.I)
    if matches:
        return matches[-1]
    matches = re.findall(r"```(.*?)```", text, flags=re.S)
    if matches:
        return matches[-1]
    return text
def main():
    parser = argparse.ArgumentParser(description="Run the VizFlow pipeline (CLI args override .env)")
    parser.add_argument("--query", required=False, default=None, help="Visualization request in natural language (or set QUERY in .env)")
    parser.add_argument("--data", required=False, default=None, help="Optional path to input CSV (or set DATA in .env)")
    parser.add_argument("--sheet", required=False, default=None, help="Excel sheet name or index for .xlsx/.xls files (or set SHEET)")
    parser.add_argument("--out", default=None, help="Output directory (or set OUT in .env; default outputs)")
    parser.add_argument("--out-name", default=None, help="Base filename for outputs (without extension) (or set OUT_NAME; default plot)")
    parser.add_argument("--format", default=None, choices=["png", "svg", "pdf"], help="Output image format (or set FORMAT; default png)")
    parser.add_argument("--dpi", default=None, type=int, help="Output image DPI (or set DPI; default 160)")
    parser.add_argument("--width", default=None, type=float, help="Figure width in inches (or set WIDTH; default 8.0)")
    parser.add_argument("--height", default=None, type=float, help="Figure height in inches (or set HEIGHT; default 5.0)")
    parser.add_argument("--style", default=None, help="Matplotlib style to apply (e.g., seaborn, ggplot) (or set STYLE)")
    # Advanced overrides
    parser.add_argument("--chart", default=None, choices=["bar", "line", "scatter", "histogram", "box", "heatmap", "area"], help="Override detected chart type")
    parser.add_argument("--x", dest="x_col", default=None, help="Override x-axis column")
    parser.add_argument("--y", dest="y_col", default=None, help="Override y-axis column")
    parser.add_argument("--color", dest="color_col", default=None, help="Override color/group column")
    parser.add_argument("--facet", dest="facet_col", default=None, help="Facet/small-multiples by this column")
    parser.add_argument("--no-refine", action="store_true", help="Disable design refinement stage")
    # Timestamped run-dir options
    parser.add_argument("--run-dir-root", default=None, help="If set, create timestamped run subdir here (e.g., D:\\nvAgent-main\\logs\\review_runs)")
    parser.add_argument("--run-prefix", default=None, help="Run subdir prefix (default review_batch_)")
    parser.add_argument("--auto-run-dir", action="store_true", help="Enable timestamped run subdir under --run-dir-root or --out")
    parser.add_argument("--max-iter", type=int, default=None, help="Maximum refinement iterations (default 3; env MAX_ITER)")
    parser.add_argument("--quality-threshold", type=float, default=None, help="Quality threshold for stopping criteria (default 0.85; env QUALITY_THRESHOLD)")
    parser.add_argument("--quality-metric", default=None, help="Metric key from evaluator JSON used for threshold (default specification_adherence_score)")
    parser.add_argument("--llm-provider", default=None, help="LLM provider strategy (openai-compatible|google|anthropic)")
    args = parser.parse_args()
    # Apply .env defaults (D:\\hope_last\\.env preferred)
    apply_env_defaults(args)
    if not args.out:
        args.out = "outputs"
    if not args.out_name:
        args.out_name = "plot"
    if not args.format:
        args.format = "png"
    if args.dpi is None:
        args.dpi = 160
    if args.width is None:
        args.width = 8.0
    if args.height is None:
        args.height = 5.0
    if not args.query:
        parser.error("Missing --query and QUERY in .env")
    if args.max_iter is None or args.max_iter <= 0:
        args.max_iter = 3
    if args.quality_threshold is None or args.quality_threshold <= 0:
        args.quality_threshold = 0.85
    if not getattr(args, "quality_metric", None):
        args.quality_metric = "specification_adherence_score"
    if not getattr(args, "llm_provider", None):
        args.llm_provider = "openai-compatible"

    # Decide output directory: optionally use timestamped subdir
    base_out = Path(args.out or "outputs")
    if getattr(args, "auto_run_dir", False) or getattr(args, "run_dir_root", None):
        base_root = Path(args.run_dir_root) if args.run_dir_root else base_out
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"{args.run_prefix or 'review_batch_'}{ts}"
        out_dir = ensure_dir(base_root / run_name)
    else:
        out_dir = ensure_dir(base_out)
    client = None
    llm_errors = []
    # Local agents removed: require LLM client
    try:
        client = LLMClient.from_env(args.llm_provider)
        # Try to list models for a helpful hint if configured model is not allowed
        try:
            models = client.list_models()
            if models and client.model not in models:
                llm_errors.append({"stage": "init", "warning": f"Configured model '{client.model}' not in allowed models: {models[:10]}..."})
        except Exception as e:
            llm_errors.append({"stage": "init", "warning": f"List models failed: {e}"})
    except Exception as e:
        raise SystemExit(
            "LLM API client initialization failed. Ensure BASE_URL and API_KEY (or OPENAI_API_KEY) are set in .env or environment.\n"
            f"Details: {e}"
        )

    def _with_model_fallback(call_fn, stage_name: str):
        if not client:
            raise RuntimeError("LLM client not initialized")
        try:
            return call_fn()
        except Exception as e:
            msg = str(e)
            # On model permission error, try fallback model from list
            if any(k in msg for k in ["没有权限", "no permission", "permission", "权限"]):
                try:
                    models = client.list_models()
                    if models:
                        client.model = models[0]
                        llm_errors.append({"stage": stage_name, "info": f"Retry with fallback model '{client.model}'"})
                        return call_fn()
                except Exception as e2:
                    llm_errors.append({"stage": stage_name, "error": f"fallback failed: {e2}"})
            raise

    def _record_llm_error(stage_name: str, exc: Exception) -> None:
        detail = {"stage": stage_name, "error": str(exc)}
        try:
            from utils.llm_client import LLMAPIError
            if isinstance(exc, LLMAPIError):
                detail.update({
                    "endpoint": exc.endpoint,
                    "status": exc.status,
                    "url": exc.url,
                    "response_snippet": (exc.response_text[:200] if exc.response_text else None),
                    "response_json": exc.response_json,
                })
        except Exception:
            pass
        llm_errors.append(detail)

    # 1) Query Analyzer (LLM only)
    try:
        qa = _with_model_fallback(lambda: llm_query_analyzer(client, args.query), "query_analyzer")
    except Exception as e:
        detail = {"stage": "query_analyzer", "error": str(e)}
        try:
            from utils.llm_client import LLMAPIError
            if isinstance(e, LLMAPIError):
                detail.update({"endpoint": e.endpoint, "status": e.status, "url": e.url, "response_snippet": (e.response_text[:200] if e.response_text else None), "response_json": e.response_json})
        except Exception:
            pass
        llm_errors.append(detail)
        raise
    write_json(qa, out_dir / "query_analyzer.json")
    try:
        print(f"[1/8] Query Analyzer -> {out_dir / 'query_analyzer.json'}", flush=True)
        print(json.dumps({
            "visualization_type": qa.get("visualization_type"),
            "interpreted_intent": qa.get("interpreted_intent")
        }, ensure_ascii=False), flush=True)
    except Exception:
        pass

    plot_type = args.chart or qa.get("visualization_type", "bar")
    todo_plan = _parse_todo_plan(qa.get("global_todo_list"))

    # 2) Data Processor (LLM only)
    try:
        data_result, df = _with_model_fallback(lambda: llm_data_processor(client, args.query, plot_type, args.data), "data_processor")
    except Exception as e:
        detail = {"stage": "data_processor", "error": str(e)}
        try:
            from utils.llm_client import LLMAPIError
            if isinstance(e, LLMAPIError):
                detail.update({"endpoint": e.endpoint, "status": e.status, "url": e.url, "response_snippet": (e.response_text[:200] if e.response_text else None), "response_json": e.response_json})
        except Exception:
            pass
        llm_errors.append(detail)
        raise
    dataset_registry = data_result.get("dataset_registry") or {}
    field_registry = data_result.get("field_registry") or {}
    dataset_registry = _ensure_registry_from_df(df, dataset_registry)
    insights_section = data_result.get("insights") or {}
    quality_issues = insights_section.get("quality_issues") or []
    if not isinstance(quality_issues, list):
        quality_issues = [str(quality_issues)]
    data_quality_score = _compute_data_quality_score(quality_issues)
    break_suggestion = insights_section.get("break_suggestion")
    write_json(data_result, out_dir / "data_processor.json")
    try:
        print(f"[2/8] Data Processor -> {out_dir / 'data_processor.json'} | df shape: {tuple(df.shape)}", flush=True)
    except Exception:
        print(f"[2/8] Data Processor -> {out_dir / 'data_processor.json'}", flush=True)

    # 3) Viz Mapping (LLM only)
    try:
        raw_mapping = _with_model_fallback(
            lambda: llm_viz_mapping(client, args.query, df, plot_type),
            "viz_mapping",
        )
    except Exception as e:
        detail = {"stage": "viz_mapping", "error": str(e)}
        try:
            from utils.llm_client import LLMAPIError
            if isinstance(e, LLMAPIError):
                detail.update({
                    "endpoint": e.endpoint,
                    "status": e.status,
                    "url": e.url,
                    "response_snippet": (e.response_text[:200] if e.response_text else None),
                    "response_json": e.response_json,
                })
        except Exception:
            pass
        llm_errors.append(detail)
        raise

    mapping_spec = upgrade_to_v1_2(raw_mapping)
    mapping_spec.setdefault("mapping", {}).setdefault(
        "data_mappings", mapping_spec.get("data_mappings", {})
    )
    mapping_spec["data_mappings"] = mapping_spec["mapping"].get("data_mappings", {})

    if break_suggestion and not (
        mapping_spec["mapping"].get("axes", {}).get("y", {}).get("breaks")
    ):
        mapping_spec["mapping"].setdefault("axes", {}).setdefault("y", {}).setdefault(
            "breaks", break_suggestion
        )

    layers = mapping_spec["mapping"].get("layers", [])
    registry_keys = list(dataset_registry.keys()) if dataset_registry else []
    default_source = registry_keys[0] if len(registry_keys) == 1 else None
    if layers:
        for idx, layer in enumerate(layers):
            if not layer.get("source") and default_source:
                layer["source"] = default_source
            elif layer.get("source") not in dataset_registry and default_source:
                layer["source"] = default_source
            elif layer.get("source") not in dataset_registry:
                layer["source"] = registry_keys[0] if registry_keys else layer.get("source", "main")
            sel = layer.setdefault("select", {})
            if args.x_col:
                sel["x"] = args.x_col
            if args.y_col and (idx == 0 or "y" in sel):
                sel["y"] = args.y_col
            if args.color_col and (idx == 0 or "color" in sel):
                sel["color"] = args.color_col
            if args.facet_col and (idx == 0 or "facet" in sel):
                sel["facet"] = args.facet_col

    dm = mapping_spec["mapping"].get("data_mappings", {})
    if args.x_col:
        dm["x_axis"] = args.x_col
    if args.y_col:
        dm["y_axis"] = args.y_col
    if args.color_col:
        dm["color"] = args.color_col
    if args.facet_col:
        dm["facet"] = args.facet_col
    mapping_spec["mapping"]["data_mappings"] = dm
    mapping_spec["data_mappings"] = dm

    sh = mapping_spec.get("styling_hints", raw_mapping.get("styling_hints", {})) or {}
    if args.x_col:
        sh["xlabel"] = args.x_col
    if args.y_col:
        sh["ylabel"] = args.y_col
    mapping_spec["styling_hints"] = sh

    def _coerce_mapping_to_df(df_use: pd.DataFrame, mp: Dict[str, str]) -> Dict[str, str]:
        import pandas as _pd

        cols = list(df_use.columns)
        cat_cols = [
            c
            for c in cols
            if df_use[c].dtype == "object" or str(df_use[c].dtype).startswith("category")
        ]
        num_cols = [c for c in cols if _pd.api.types.is_numeric_dtype(df_use[c])]
        x = mp.get("x_axis") or (cat_cols[0] if cat_cols else (cols[0] if cols else "x"))
        y = mp.get("y_axis") or (num_cols[0] if num_cols else (cols[1] if len(cols) > 1 else "y"))
        color = mp.get("color")
        if x not in cols:
            x = cat_cols[0] if cat_cols else (cols[0] if cols else x)
        if y not in cols:
            y = num_cols[0] if num_cols else (cols[1] if len(cols) > 1 else y)
        if (not color) or (color not in cols) or (color == x):
            color = None
            for candidate in cat_cols:
                if candidate != x:
                    color = candidate
                    break
        mp2 = dict(mp)
        mp2["x_axis"] = x
        mp2["y_axis"] = y
        if color:
            mp2["color"] = color
        return mp2

    before_dm = dict(mapping_spec.get("data_mappings", {}))
    coerced_dm = _coerce_mapping_to_df(df, before_dm)
    if coerced_dm != before_dm:
        mapping_spec["mapping"]["data_mappings"] = coerced_dm
        mapping_spec["data_mappings"] = coerced_dm
        sh["xlabel"] = coerced_dm.get("x_axis", sh.get("xlabel"))
        sh["ylabel"] = coerced_dm.get("y_axis", sh.get("ylabel"))
        mapping_spec["styling_hints"] = sh
        if layers:
            base_select = layers[0].setdefault("select", {})
            if coerced_dm.get("x_axis"):
                base_select["x"] = coerced_dm.get("x_axis")
            if coerced_dm.get("y_axis"):
                base_select["y"] = coerced_dm.get("y_axis")
            if coerced_dm.get("color"):
                base_select["color"] = coerced_dm.get("color")

    valid_cols = set(df.columns)
    sanitized_layers: list[Dict[str, Any]] = []
    for layer in layers:
        select = layer.get("select", {}) or {}
        invalid_layer = False
        for key in ("x", "y", "color", "size", "facet"):
            val = select.get(key)
            if isinstance(val, str) and val not in valid_cols:
                if "kde(" in val.lower() or "density" in val.lower():
                    invalid_layer = True
                    break
        if not invalid_layer:
            sanitized_layers.append(layer)
    if len(sanitized_layers) != len(layers):
        mapping_spec["mapping"]["layers"] = sanitized_layers
        layers = sanitized_layers

    rect_layers = [lyr for lyr in layers if str(lyr.get("geom")).lower() == "rect"]
    if rect_layers:
        prepared_df = _prepare_heatmap_dataframe(df)
        if prepared_df is not None and not prepared_df.empty:
            df = prepared_df
            dataset_registry = _ensure_registry_from_df(df, dataset_registry)
            dm = mapping_spec["mapping"].setdefault("data_mappings", {})
            dm.update({"x_axis": "condition", "y_axis": "symbol", "color": "z_score"})
            mapping_spec["data_mappings"] = dm
            for layer in rect_layers:
                sel = layer.setdefault("select", {})
                sel["x"] = "condition"
                sel["y"] = "symbol"
                sel["color"] = "z_score"
                layer["roles"] = {"x": "dimension", "y": "dimension"}
                layer["transform"] = {"filter": [], "derive": [], "groupby": [], "aggregate": {}, "sort": []}
                layer.setdefault("legend", "Viability z-score relative to control")
            enc = mapping_spec["mapping"].setdefault("encodings", {})
            prev_color = enc.get("color_by") if isinstance(enc.get("color_by"), dict) else {}
            prev_scale = prev_color.get("scale") if isinstance(prev_color, dict) else None
            enc["color_by"] = {
                "field": "z_score",
                "type": "quantitative",
                "scale": prev_scale or {"type": "diverging"},
            }
            axes_cfg = mapping_spec["mapping"].setdefault("axes", {})
            axes_cfg.setdefault("x", {}).update({"type": "nominal", "label": "Condition"})
            axes_cfg.setdefault("y", {}).update({"type": "nominal", "label": "Gene Symbol"})
            mapping_spec["styling_hints"] = {
                **(mapping_spec.get("styling_hints") or {}),
                "xlabel": "Condition",
                "ylabel": "Gene Symbol",
            }

    mapping_spec, reconcile_warnings = reconcile_mapping_with_registry(mapping_spec, dataset_registry)
    if reconcile_warnings:
        mapping_spec.setdefault("_warnings", []).extend(reconcile_warnings)
        llm_errors.append({"stage": "viz_mapping", "warning": "; ".join(reconcile_warnings)})

    try:
        validate_spec_vs_registry(mapping_spec, dataset_registry)
    except ValueError as err:
        write_text(str(err), out_dir / "spec_validation_error.txt")
        raise

    style_context = _extract_style_context(mapping_spec)

    paper_schema = figure_spec_to_paper_schema(mapping_spec)

    write_json(mapping_spec, out_dir / "viz_mapping.json")
    write_json(paper_schema, out_dir / "viz_mapping_paper.json")
    try:
        print(f"[3/8] Viz Mapping -> {out_dir / 'viz_mapping.json'}", flush=True)
        print(
            json.dumps(
                {
                    "chart_type": mapping_spec.get("mapping", {}).get("chart_type"),
                    "data_mappings": mapping_spec.get("mapping", {}).get("data_mappings", {}),
                    "paper_schema": paper_schema_summary(paper_schema),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    except Exception:
        pass
    # 4) Search Agent (Matplotlib example via LLM)
    try:
        search_context = _with_model_fallback(lambda: llm_search_agent(client, plot_type), "search_agent")
    except Exception as e:
        detail = {"stage": "search_agent", "error": str(e)}
        try:
            from utils.llm_client import LLMAPIError
            if isinstance(e, LLMAPIError):
                detail.update({"endpoint": e.endpoint, "status": e.status, "url": e.url, "response_snippet": (e.response_text[:200] if e.response_text else None), "response_json": e.response_json})
        except Exception:
            pass
        llm_errors.append(detail)
        search_context = {"references": [], "example_code": ""}
    example_code = search_context.get("example_code") or ""
    official_refs = official_references(plot_type)
    existing_refs = search_context.get("references", []) or []
    existing_urls = {str(ref.get("url")) for ref in existing_refs if isinstance(ref, dict)}
    for ref in official_refs:
        if isinstance(ref, dict):
            url = str(ref.get("url"))
            if url and url not in existing_urls:
                existing_refs.append(ref)
                existing_urls.add(url)
    search_context["references"] = existing_refs
    write_text(example_code, out_dir / "search_agent_example.py")
    try:
        write_json({"references": search_context.get("references", [])}, out_dir / "search_agent_references.json")
    except Exception:
        pass
    try:
        print(f"[4/8] Search Agent example -> {out_dir / 'search_agent_example.py'}", flush=True)
    except Exception:
        pass


    # 5) Iterative Design, Generation, and Evaluation (Self-Reflection Loop)
    data_char = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns": list(map(str, df.columns.tolist())),
        "dtypes": {str(c): str(df[c].dtype) for c in df.columns},
    }
    design_base_extras = copy.deepcopy(qa.get("_extras", {})) if isinstance(qa.get("_extras"), dict) else {}
    if args.style:
        design_base_extras["style"] = args.style
    policies = design_base_extras.get("policies")
    if not isinstance(policies, dict):
        policies = {}
    policies.setdefault("transparency", "auto")
    policies.setdefault("color", "colorblind_safe")
    policies.setdefault("stats", "n_and_ci")
    design_base_extras["policies"] = policies
    style_spec_state = StyleSpec.default()
    design_base_extras["style_spec_seed"] = style_spec_state.to_dict()
    design_base_extras["figsize"] = [args.width, args.height]
    if getattr(args, "_env_extras", None):
        for k, v in args._env_extras.items():
            design_base_extras[k] = v
    facet_count = None
    if dm.get("facet") and dm["facet"] in df.columns:
        try:
            facet_count = int(df[dm["facet"]].nunique())
        except Exception:
            facet_count = None
    design_base_extras["facet_count"] = facet_count
    design_base_extras["facet_col"] = dm.get("facet")
    design_base_extras["style_global"] = style_context.get("global")
    design_base_extras["layer_styles"] = style_context.get("layers")
    design_base_extras["quality_issues"] = quality_issues
    design_base_extras["todo_plan"] = todo_plan
    design_base_extras["paper_schema_summary"] = paper_schema_summary(paper_schema)
    design_base_extras["data_context"] = data_char

    search_context.setdefault("references", [])
    search_context.setdefault("example_code", "")

    quality_metric_key = args.quality_metric or "specification_adherence_score"
    iteration_results: List[Dict[str, Any]] = []
    iteration_feedback: Optional[Dict[str, Any]] = None
    best_iteration: Optional[Dict[str, Any]] = None
    best_metric_value = float("-inf")

    def _execute_iteration(iter_idx: int, feedback_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        nonlocal style_spec_state
        stage_name = f"iteration_{iter_idx}"
        extras = copy.deepcopy(design_base_extras)
        extras["style_spec"] = style_spec_state.to_dict()
        extras["iteration"] = iter_idx
        if feedback_payload:
            extras["feedback"] = feedback_payload

        try:
            design_spec_iter = _with_model_fallback(
                lambda: llm_design_explorer(
                    client,
                    args.query,
                    qa.get("interpreted_intent", ""),
                    plot_type,
                    data_char,
                    extras=extras,
                    feedback=feedback_payload,
                    iteration=iter_idx,
                ),
                stage_name + "_design",
            )
        except Exception as e:
            _record_llm_error(stage_name + "_design", e)
            raise

        design_spec_iter["_mapping_styling"] = mapping_spec.get("styling_hints", {})
        design_spec_iter["_extras"] = extras
        design_spec_iter.setdefault("self_reflection_notes", [])
        design_spec_iter.setdefault("code_generator_actions", [])
        design_spec_iter["_previous_style_spec"] = style_spec_state.to_dict()

        design_path = out_dir / f"design_explorer_iter_{iter_idx}.json"

        stylist_stage_name = stage_name + "_stylist"
        style_spec_raw: Dict[str, Any] = {}
        stylist_data_context = {
            "shape": data_char.get("shape"),
            "columns": data_char.get("columns"),
            "dtypes": data_char.get("dtypes"),
            "quality_issues": quality_issues,
            "plot_type": plot_type,
        }
        try:
            if iter_idx > 1 and feedback_payload:
                style_spec_raw = _with_model_fallback(
                    lambda: llm_aesthetic_stylist_refine(
                        client,
                        style_spec_state.to_dict(),
                        feedback=feedback_payload,
                        iteration=iter_idx,
                    ),
                    stylist_stage_name,
                )
            else:
                style_spec_raw = _with_model_fallback(
                    lambda: llm_aesthetic_stylist(
                        client,
                        args.query,
                        design_spec_iter,
                        mapping_spec,
                        stylist_data_context,
                        policies=policies,
                        search_context=search_context,
                        iteration=iter_idx,
                        feedback=feedback_payload,
                    ),
                    stylist_stage_name,
                )
        except Exception as e:
            _record_llm_error(stylist_stage_name, e)
            style_spec_raw = style_spec_state.to_dict()

        style_spec_candidate = StyleSpec(style_spec_state.to_dict())
        if isinstance(style_spec_raw, dict):
            style_spec_candidate.merge(style_spec_raw)
        stage_patches = {}
        if isinstance(feedback_payload, dict):
            stage_patches = feedback_payload.get("stage_patches", {}) or {}
        if isinstance(stage_patches, dict):
            for key in ("L1", "L3", "L4"):
                patch_payload = stage_patches.get(key)
                if isinstance(patch_payload, dict):
                    style_spec_candidate.update_from_feedback({"patch_suggestions": patch_payload})
        style_spec_state = style_spec_candidate
        style_spec_dict = style_spec_state.to_dict()
        design_spec_iter["_style_spec"] = style_spec_dict
        write_json(design_spec_iter, design_path)
        try:
            print(f"[5.{iter_idx}] Design Explorer -> {design_path}", flush=True)
        except Exception:
            pass
        style_spec_path = out_dir / f"style_spec_iter_{iter_idx}.json"
        write_json(style_spec_dict, style_spec_path)
        try:
            print(f"[5.{iter_idx}] Aesthetic Stylist -> {style_spec_path}", flush=True)
        except Exception:
            pass

        try:
            code_plan_iter = _with_model_fallback(
                lambda: llm_code_generator(
                    client,
                    qa,
                    design_spec_iter,
                    int(df.shape[0]),
                    int(df.shape[1]),
                    list(map(str, df.columns.tolist())),
                    data_quality_score,
                    style_spec_dict,
                    mapping_spec,
                    paper_schema,
                    search_context,
                    iteration=iter_idx,
                    feedback=feedback_payload,
                ),
                stage_name + "_code_plan",
            )
        except Exception as e:
            _record_llm_error(stage_name + "_code_plan", e)
            code_plan_iter = {}

        plan_path = out_dir / f"code_generator_plan_iter_{iter_idx}.json"
        write_json(code_plan_iter, plan_path)

        df_columns_list = list(map(str, df.columns.tolist()))
        heatmap_layers = [
            lyr
            for lyr in mapping_spec.get("mapping", {}).get("layers", [])
            if str(lyr.get("geom")).lower() == "rect"
        ]
        pie_layers = [
            lyr
            for lyr in mapping_spec.get("mapping", {}).get("layers", [])
            if str(lyr.get("geom")).lower() == "pie"
        ]
        histogram_layers = [
            lyr
            for lyr in mapping_spec.get("mapping", {}).get("layers", [])
            if str(lyr.get("geom")).lower() == "histogram"
        ]
        box_layers = [
            lyr
            for lyr in mapping_spec.get("mapping", {}).get("layers", [])
            if str(lyr.get("geom")).lower() == "box"
        ]
        cg_guidance = (
            "Existing DataFrame columns: "
            + json.dumps(df_columns_list, ensure_ascii=False)
            + ". Use in-memory DataFrame 'df'; do not assume disk IO."
        )
        if heatmap_layers:
            cg_guidance += (
                " For heatmap layers (geom='rect'), use tidy columns from DataFrame (e.g., 'condition', 'symbol', 'z_score') "
                "and ensure the colormap is centered with a labeled colorbar."
            )
        if pie_layers:
            cg_guidance += (
                " For pie layers (geom='pie'), aggregate df to one value per category, call plt.pie with autopct, "
                "and enforce equal aspect via plt.axis('equal')."
            )
        if histogram_layers:
            cg_guidance += (
                " For histogram layers, use matplotlib.pyplot.hist with explicit bins and labeled axes."
            )
        if box_layers:
            cg_guidance += (
                " For box plot layers, rely on matplotlib.pyplot.boxplot or seaborn.boxplot referencing the mapped columns."
            )
        cg_guidance += (
            " Style helpers are injected as style_global, layer_styles, merge_styles(...), resolve_palette(...), resolve_cmap(...); "
            "apply alpha/palette/cmap/linewidth/markersize/edgecolor accordingly."
        )
        cg_guidance += (
            " Additional helpers auto_tune_alpha(fig), adjust_local_contrast(fig), annotate_sample_size(ax, df, x, y), add_errorbars(ax, df, x, y) "
            "are available to mitigate overplotting and surface sample sizes with confidence intervals when appropriate."
        )

        reference_lines = "\n".join(
            f"- {ref.get('title', 'reference')}: {ref.get('url')}"
            for ref in search_context.get("references", [])
            if isinstance(ref, dict)
        )
        if not reference_lines:
            reference_lines = "- Matplotlib API Reference: https://matplotlib.org/stable/index.html"
        plan_summary = json.dumps(code_plan_iter, ensure_ascii=False)
        feedback_summary = json.dumps(feedback_payload or {}, ensure_ascii=False)
        design_actions = json.dumps(design_spec_iter.get("code_generator_actions", []), ensure_ascii=False)
        helper_notes = json.dumps(design_spec_iter.get("self_reflection_notes", []), ensure_ascii=False)

        target_basename = f"{args.out_name}_iter_{iter_idx}"
        image_target = out_dir / f"{target_basename}.{args.format}"

        cg_prompt_lines = [
            f"You are a senior Matplotlib engineer executing iteration {iter_idx} of a self-reflective pipeline.",
            "Follow FigureSpec mapping and design directives to regenerate the plot.",
            f"FigureSpec: {json.dumps(mapping_spec, ensure_ascii=False)}",
            f"PaperSchemaSummary: {design_base_extras['paper_schema_summary']}",
            f"DesignSelfReflectionNotes: {helper_notes}",
            f"CodeGeneratorActions: {design_actions}",
            f"IterationFeedback: {feedback_summary}",
            f"PlanningJSON: {plan_summary}",
            "Authoritative references:",
            reference_lines,
            "Reference snippet:",
            "```python",
            search_context.get('example_code', ''),
            "```",
            cg_guidance,
            f"Save the final image to '{image_target.as_posix()}' and assign the matplotlib figure to a variable named fig.",
            "Use a non-interactive backend and finish by calling fig.savefig.",
            "Respond only with python code in a code block.",
        ]
        cg_prompt = "\n".join(line for line in cg_prompt_lines if line)
        try:
            code_response = _with_model_fallback(
                lambda: client.chat([{"role": "user", "content": cg_prompt}], temperature=0.12),
                stage_name + "_code_snippet",
            )
            generated_plot_code = _extract_code_block(code_response)
        except Exception as e:
            _record_llm_error(stage_name + "_code_snippet", e)
            generated_plot_code = ""
        if not generated_plot_code.strip():
            generated_plot_code = search_context.get("example_code", "")

        code_path = out_dir / f"generated_plot_iter_{iter_idx}.py"
        write_text(generated_plot_code, code_path)

        code_plan_out = {
            "iteration": iter_idx,
            "plan": code_plan_iter,
            "generated_code": generated_plot_code,
            "stage": stage_name,
        }
        code_plan_path = out_dir / f"code_generator_iter_{iter_idx}.json"
        write_json(code_plan_out, code_plan_path)

        ensure_dir("outputs")

        def _normalize_plot_code_paths(code: str) -> str:
            try:
                code = code.replace("outputs/", out_dir.as_posix() + "/")
                code = code.replace("outputs\\", str(out_dir) + "\\")
                target_posix = image_target.as_posix()
                replacements = [
                    ("'plot.png'", f"'{target_posix}'"),
                    ('"plot.png"', f'"{target_posix}"'),
                    (f"'{args.out_name}.{args.format}'", f"'{target_posix}'"),
                    (f'"{args.out_name}.{args.format}"', f'"{target_posix}"'),
                ]
                for old, new in replacements:
                    code = code.replace(old, new)
            except Exception:
                pass
            return code

        def _build_exec_locals() -> Dict[str, Any]:
            local_vars: Dict[str, Any] = {
                "pd": pd,
                "np": __import__("numpy"),
                "plt": __import__("matplotlib.pyplot"),
            }
            local_vars["df"] = df.copy(deep=True)
            local_vars["style_global"] = style_context.get("global", {})
            local_vars["layer_styles"] = style_context.get("layers", {})
            local_vars["merge_styles"] = _merge_styles
            local_vars["resolve_palette"] = _resolve_palette
            local_vars["resolve_cmap"] = _resolve_cmap
            local_vars["auto_tune_alpha"] = auto_tune_alpha
            local_vars["adjust_local_contrast"] = adjust_local_contrast
            local_vars["annotate_sample_size"] = annotate_sample_size
            local_vars["add_errorbars"] = add_errorbars
            return local_vars

        executed = False
        last_exec_locals: Optional[Dict[str, Any]] = None
        debug_payload: Dict[str, Any] = {}
        debug_path: Optional[Path] = None
        exec_code = _normalize_plot_code_paths(generated_plot_code)
        try:
            _validate_generated_code(generated_plot_code)
            local_vars = _build_exec_locals()
            exec(exec_code, local_vars, local_vars)
            last_exec_locals = local_vars
            executed = True
            try:
                print(f"[6.{iter_idx}] Plot executed successfully.", flush=True)
            except Exception:
                pass
        except Exception as e:
            debug_context = {
                "quality_issues": quality_issues,
                "references": search_context.get("references", []),
                "iteration": iter_idx,
                "stage": stage_name,
            }
            try:
                debug_payload = _with_model_fallback(
                    lambda: llm_debug_agent(client, exec_code, str(e), context=debug_context),
                    stage_name + "_debug",
                )
            except Exception as dbg_exc:
                _record_llm_error(stage_name + "_debug", dbg_exc)
                debug_payload = {}
            if debug_payload:
                debug_path = out_dir / f"debug_agent_iter_{iter_idx}.json"
                write_json(debug_payload, debug_path)
                fixed = debug_payload.get("fixed_code") or generated_plot_code
                exec_code = _normalize_plot_code_paths(fixed)
                write_text(fixed, code_path)
                try:
                    _validate_generated_code(fixed)
                    local_vars = _build_exec_locals()
                    exec(exec_code, local_vars, local_vars)
                    last_exec_locals = local_vars
                    executed = True
                    try:
                        print(f"[6.{iter_idx}] Debug Agent produced a working fix.", flush=True)
                    except Exception:
                        pass
                except Exception as e2:
                    error_path = out_dir / f"plot_error_iter_{iter_idx}.txt"
                    write_text(f"Plotting failed after debug fix: {e2}", error_path)
                    executed = False

        if executed and last_exec_locals is not None:
            fig = last_exec_locals.get("fig")
            if fig is not None:
                try:
                    auto_tune_alpha(fig)
                except Exception:
                    pass
                try:
                    adjust_local_contrast(fig)
                except Exception:
                    pass
                try:
                    sanitize_backgrounds(fig)
                except Exception:
                    pass
                try:
                    prune_insets(
                        fig,
                        allow_insets=(
                            getattr(args, "allow_insets", False)
                            or bool(style_spec_dict.get("layout_tuning", {}).get("allow_insets"))
                        ),
                    )
                except Exception:
                    pass
                try:
                    primary_ax = fig.axes[0] if fig.axes else None
                    x_field = None
                    y_field = None
                    if isinstance(dm, dict):
                        x_field = dm.get("x_axis") or dm.get("x") or dm.get("dimension")
                        y_field = dm.get("y_axis") or dm.get("y") or dm.get("measure")
                    if (
                        primary_ax is not None
                        and x_field
                        and y_field
                        and x_field in df.columns
                        and y_field in df.columns
                        and plot_type in {"bar", "line", "area", "histogram"}
                    ):
                        unique_count = df[x_field].nunique(dropna=True)
                        if 0 < unique_count <= 20:
                            try:
                                annotate_sample_size(primary_ax, df, x_field, y_field)
                            except Exception:
                                pass
                            try:
                                add_errorbars(primary_ax, df, x_field, y_field)
                            except Exception:
                                pass
                except Exception:
                    pass
                try:
                    fig.savefig(image_target, dpi=args.dpi)
                except Exception:
                    pass

        possible_images = [
            image_target,
            out_dir / f"{args.out_name}.{args.format}",
        ]
        for ext in ["png", "svg", "pdf"]:
            possible_images.append(out_dir / f"{target_basename}.{ext}")
            possible_images.append(out_dir / f"{args.out_name}.{ext}")

        final_image_path: Optional[Path] = None
        for candidate in possible_images:
            if candidate.exists():
                final_image_path = candidate
                break
        if final_image_path and final_image_path != image_target:
            try:
                image_target.write_bytes(final_image_path.read_bytes())
                final_image_path = image_target
            except Exception:
                pass
        if final_image_path is None:
            final_image_path = image_target

        image_payload = _encode_image_payload(final_image_path)
        try:
            evaluation_json = _with_model_fallback(
                lambda: llm_staged_visual_evaluator(
                    client,
                    args.query,
                    qa.get("plotting_key_points", []),
                    df,
                    mapping_spec,
                    paper_schema,
                    design_spec_iter,
                    image_payload,
                    iter_idx,
                    policies=design_base_extras.get("policies"),
                    todo=todo_plan,
                    quality_notes=quality_issues,
                    style_spec=style_spec_dict,
                ),
                stage_name + "_evaluate",
            )
        except Exception as e:
            _record_llm_error(stage_name + "_evaluate", e)
            evaluation_json = {}

        visual_path = out_dir / f"visual_evaluator_iter_{iter_idx}.json"
        write_json(evaluation_json, visual_path)
        try:
            print(f"[7.{iter_idx}] Visual Evaluator -> {visual_path}", flush=True)
        except Exception:
            pass

        metric_value = None
        target_stage = None
        stage_scores: Dict[str, float] = {}
        stage_patches: Dict[str, Any] = {}
        base_metric = None
        if isinstance(evaluation_json, dict):
            overall_block = evaluation_json.get("overall")
            if isinstance(overall_block, dict):
                target_stage = overall_block.get("target_stage")
                base_metric = overall_block.get(quality_metric_key)
            stages_block = evaluation_json.get("stages")
            if isinstance(stages_block, dict):
                for s_name, payload in stages_block.items():
                    if not isinstance(payload, dict):
                        continue
                    sc = payload.get("score")
                    if isinstance(sc, (int, float)):
                        stage_scores[s_name] = float(sc)
                    patch = payload.get("patch_suggestions")
                    if patch:
                        stage_patches[s_name] = patch
            if base_metric is None:
                semantic_block = evaluation_json.get("semantic_accuracy")
                if isinstance(semantic_block, dict):
                    base_metric = semantic_block.get(quality_metric_key)
        if getattr(args, "quality_metric", None) == "staged_weighted_score" and stage_scores:
            weights = getattr(args, "stage_weights", {"L1": 0.4, "L2": 0.3, "L3": 0.2, "L4": 0.1})
            metric_value = sum(stage_scores.get(k, 0.0) * float(weights.get(k, 0.0)) for k in ("L1", "L2", "L3", "L4"))
        elif base_metric is not None:
            try:
                metric_value = float(base_metric)
            except Exception:
                metric_value = 0.0
        else:
            try:
                metric_value = float(metric_value)
            except Exception:
                metric_value = 0.0

        floors = getattr(args, "stage_min_floors", None) or {}
        floors_failed = [k for k, v in floors.items() if stage_scores and stage_scores.get(k, 1.0) < float(v)]
        if floors_failed:
            try:
                metric_value = min(float(metric_value), float(args.quality_threshold) - 1e-3)
            except Exception:
                pass

        feedback_for_next = {
            "visual_evaluator": evaluation_json,
            "debug_agent": debug_payload,
            "quality_issues": quality_issues,
            "iteration": iter_idx,
            "metric": metric_value,
            "metric_key": quality_metric_key,
            "style_spec": style_spec_dict,
        }
        if target_stage:
            feedback_for_next["target_stage"] = target_stage
        if stage_scores:
            feedback_for_next["stage_scores"] = stage_scores
        if stage_patches:
            feedback_for_next["stage_patches"] = stage_patches

        all_ops = []
        for p in (stage_patches or {}).values():
            ops = p.get("ops") if isinstance(p, dict) else None
            if isinstance(ops, list):
                all_ops.extend(ops)
        if all_ops:
            try:
                mapping_spec, style_spec_dict_after, _notes = apply_patch_ops(mapping_spec, style_spec_dict, all_ops)
                style_spec_state = StyleSpec(style_spec_dict_after)
                applied_log = {"ops": all_ops, "notes": _notes, "style_after": style_spec_dict_after, "mapping_after": mapping_spec}
                write_json(applied_log, out_dir / f"patch_ops_applied_iter_{iter_idx}.json")
            except Exception as _e:
                write_text(f"apply_patch_ops failed: {_e}", out_dir / f"patch_ops_error_iter_{iter_idx}.txt")
        style_spec_dict = style_spec_state.to_dict()

        return {
            "iteration_index": iter_idx,
            "design_spec": design_spec_iter,
            "design_path": design_path,
            "code_plan": code_plan_iter,
            "code_plan_path": code_plan_path,
            "generated_code": generated_plot_code,
            "code_path": code_path,
            "debug_path": debug_path,
            "evaluation_path": visual_path,
            "evaluation_json": evaluation_json,
            "metric_value": metric_value,
            "feedback_for_next": feedback_for_next,
            "image_path": final_image_path,
            "style_spec": style_spec_dict,
            "style_spec_path": style_spec_path,
        }

    for iter_idx in range(1, args.max_iter + 1):
        iteration_result = _execute_iteration(iter_idx, iteration_feedback)
        iteration_results.append(iteration_result)
        metric_value = iteration_result.get("metric_value")
        try:
            metric_value = float(metric_value)
        except Exception:
            metric_value = 0.0
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_iteration = iteration_result
        iteration_feedback = iteration_result.get("feedback_for_next")
        if metric_value >= args.quality_threshold:
            try:
                print(
                    f"[Self-Reflection] Stopping after iteration {iter_idx} (metric {metric_value:.3f} >= threshold {args.quality_threshold}).",
                    flush=True,
                )
            except Exception:
                pass
            break
    else:
        try:
            print(
                f"[Self-Reflection] Completed {args.max_iter} iterations (best {best_metric_value:.3f}).",
                flush=True,
            )
        except Exception:
            pass

    if best_iteration is None and iteration_results:
        best_iteration = iteration_results[-1]

    final_iteration = best_iteration or {}
    final_design_spec = final_iteration.get("design_spec") or {}
    final_code_plan = final_iteration.get("code_plan") or {}
    final_generated_code = final_iteration.get("generated_code") or ""
    final_design_path = final_iteration.get("design_path")
    final_code_plan_path = final_iteration.get("code_plan_path")
    final_code_path = final_iteration.get("code_path")
    final_debug_path = final_iteration.get("debug_path")
    final_evaluation_path = final_iteration.get("evaluation_path")
    final_evaluation_json = final_iteration.get("evaluation_json") or {}
    final_image_path = final_iteration.get("image_path")
    final_iteration_index = final_iteration.get("iteration_index", 1)
    final_style_spec = final_iteration.get("style_spec") or {}
    final_style_spec_path = final_iteration.get("style_spec_path")

    if final_design_spec:
        write_json(final_design_spec, out_dir / "design_explorer.json")
        if final_iteration_index > 1:
            write_json(final_design_spec, out_dir / "design_explorer_refined.json")
    if final_code_plan or final_generated_code:
        write_json(
            {
                "iteration": final_iteration_index,
                "plan": final_code_plan,
                "generated_code": final_generated_code,
            },
            out_dir / "code_generator.json",
        )
    if final_generated_code:
        write_text(final_generated_code, out_dir / "generated_plot.py")
    if final_evaluation_json:
        write_json(final_evaluation_json, out_dir / "visual_evaluator.json")
    if final_style_spec:
        write_json(final_style_spec, out_dir / "style_spec.json")
        if final_iteration_index > 1:
            write_json(final_style_spec, out_dir / "style_spec_refined.json")
    if final_debug_path and isinstance(final_debug_path, Path) and final_debug_path.exists():
        try:
            (out_dir / "debug_agent.json").write_bytes(final_debug_path.read_bytes())
        except Exception:
            pass

    canonical_image_path = out_dir / f"{args.out_name}.{args.format}"
    if isinstance(final_image_path, Path) and final_image_path.exists():
        if final_image_path != canonical_image_path:
            try:
                canonical_image_path.write_bytes(final_image_path.read_bytes())
            except Exception:
                canonical_image_path = final_image_path
    elif isinstance(final_image_path, Path):
        canonical_image_path = final_image_path

    visual_evaluator_path = out_dir / "visual_evaluator.json"
    if not visual_evaluator_path.exists() and final_evaluation_json:
        write_json(final_evaluation_json, visual_evaluator_path)

    # Save processed dataframe for transparency
    try:
        df.to_csv(out_dir / "processed_data.csv", index=False)
    except Exception:
        pass

    # Aggregate a high-level results index
    index: Dict[str, Any] = {
        "query": args.query,
        "plot_type": plot_type,
        "outputs": {
            "image": str((out_dir / f"{args.out_name}.{args.format}").resolve()),
            "query_analyzer": str((out_dir / "query_analyzer.json").resolve()),
            "data_processor": str((out_dir / "data_processor.json").resolve()),
            "viz_mapping": str((out_dir / "viz_mapping.json").resolve()),
            "viz_mapping_paper": str((out_dir / "viz_mapping_paper.json").resolve()) if (out_dir / "viz_mapping_paper.json").exists() else None,
            "design_explorer": str((out_dir / "design_explorer.json").resolve()),
            "design_explorer_refined": str((out_dir / "design_explorer_refined.json").resolve()) if (out_dir / "design_explorer_refined.json").exists() else None,
            "code_generator": str((out_dir / "code_generator.json").resolve()),
            "search_agent_example": str((out_dir / "search_agent_example.py").resolve()),
            "search_agent_references": str((out_dir / "search_agent_references.json").resolve()) if (out_dir / "search_agent_references.json").exists() else None,
            "generated_plot_code": str((out_dir / "generated_plot.py").resolve()),
            "visual_evaluator": str((out_dir / "visual_evaluator.json").resolve()) if (out_dir / "visual_evaluator.json").exists() else None,
            "processed_data": str((out_dir / "processed_data.csv").resolve()) if (out_dir / "processed_data.csv").exists() else None,
            "debug_agent": str((out_dir / "debug_agent.json").resolve()) if (out_dir / "debug_agent.json").exists() else None,
            "style_spec": str((out_dir / "style_spec.json").resolve()) if (out_dir / "style_spec.json").exists() else None,
        },
        "best_iteration": final_iteration_index,
        "best_metric_value": best_metric_value,
        "iterations": [
            {
                "iteration": result.get("iteration_index"),
                "metric_value": result.get("metric_value"),
                "design_path": str(result.get("design_path")) if isinstance(result.get("design_path"), Path) else result.get("design_path"),
                "code_path": str(result.get("code_path")) if isinstance(result.get("code_path"), Path) else result.get("code_path"),
                "evaluation_path": str(result.get("evaluation_path")) if isinstance(result.get("evaluation_path"), Path) else result.get("evaluation_path"),
                "style_spec_path": str(result.get("style_spec_path")) if isinstance(result.get("style_spec_path"), Path) else result.get("style_spec_path"),
            }
            for result in iteration_results
        ],
    }
    if getattr(args, "_env_path", None):
        index["env_path"] = args._env_path
    index["llm_provider"] = args.llm_provider
    write_json(index, out_dir / "results.json")

    # Build an HTML report for convenience
    try:
        report_html = build_html_report(out_dir)
        (out_dir / "report.html").write_text(report_html, encoding="utf-8")
    except Exception:
        pass

    # Save any LLM errors collected
    if llm_errors:
        write_json({"llm_errors": llm_errors}, out_dir / "llm_errors.json")
        try:
            print(f"[LLM] Some stages used fallbacks. Details -> {out_dir / 'llm_errors.json'}", flush=True)
        except Exception:
            pass

    print(f"Pipeline complete. Outputs are in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
