from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


def _parse_bool(val: Optional[str]) -> Optional[bool]:
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def _manual_read_env_file(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return env


def load_env_map() -> tuple[Dict[str, str], Optional[str]]:
    # Prefer explicit ENV_PATH, then D:\hope_last\.env, then ./.env
    candidates = []
    explicit = os.environ.get("ENV_PATH")
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(Path(r"D:\hope_last\.env"))
    candidates.append(Path.cwd() / ".env")

    env_map: Dict[str, str] = {}
    used: Optional[str] = None
    tried_any = False
    for p in candidates:
        if p.exists():
            tried_any = True
            try:
                # Try python-dotenv if available
                try:
                    from dotenv import dotenv_values  # type: ignore
                    env_map.update({k: str(v) for k, v in dotenv_values(str(p)).items() if v is not None})
                except Exception:
                    env_map.update(_manual_read_env_file(p))
                used = str(p)
            except Exception:
                continue
            break
    # If none exists, still return empty map
    return env_map, used


def apply_env_defaults(args) -> None:
    env, used_path = load_env_map()

    def g(key: str, default: Any = None) -> Any:
        return env.get(key, os.environ.get(key, default))

    # String fields
    if getattr(args, "query", None) in (None, ""):
        args.query = g("QUERY")
    if getattr(args, "data", None) in (None, ""):
        args.data = g("DATA")
    if getattr(args, "sheet", None) in (None, ""):
        args.sheet = g("SHEET") or g("SHEET_NAME")
    if getattr(args, "out", None) in (None, ""):
        args.out = g("OUT", "outputs")
    if getattr(args, "out_name", None) in (None, ""):
        args.out_name = g("OUT_NAME", "plot")
    if getattr(args, "format", None) in (None, ""):
        args.format = (g("FORMAT", "png") or "png").lower()
    if getattr(args, "style", None) in (None, ""):
        args.style = g("STYLE")
    if getattr(args, "chart", None) in (None, ""):
        args.chart = g("CHART")
    if getattr(args, "x_col", None) in (None, ""):
        args.x_col = g("X")
    if getattr(args, "y_col", None) in (None, ""):
        args.y_col = g("Y")
    if getattr(args, "color_col", None) in (None, ""):
        args.color_col = g("COLOR")
    if getattr(args, "facet_col", None) in (None, ""):
        args.facet_col = g("FACET")

    # Numeric fields
    if getattr(args, "dpi", None) in (None, ""):
        v = g("DPI")
        try:
            args.dpi = int(v) if v is not None else 160
        except Exception:
            args.dpi = 160
    if getattr(args, "width", None) in (None, ""):
        v = g("WIDTH")
        try:
            args.width = float(v) if v is not None else 8.0
        except Exception:
            args.width = 8.0
    if getattr(args, "height", None) in (None, ""):
        v = g("HEIGHT")
        try:
            args.height = float(v) if v is not None else 5.0
        except Exception:
            args.height = 5.0

    # Booleans
    if getattr(args, "no_refine", None) in (None, ""):
        b = _parse_bool(g("NO_REFINE"))
        args.no_refine = bool(b) if b is not None else False
    # LLM toggle
    if getattr(args, "llm", None) in (None, ""):
        b = _parse_bool(g("USE_LLM"))
        # default True if unspecified
        args.llm = True if b is None else bool(b)

    # LLM controls
    if getattr(args, "llm_timeout", None) in (None, ""):
        # Support formats like "60" or "10,120" (connect,read)
        tval = g("LLM_TIMEOUT")
        timeout_parsed = None
        if tval:
            try:
                if "," in tval:
                    parts = [p.strip() for p in tval.split(",")]
                    timeout_parsed = (float(parts[0]), float(parts[1]))
                else:
                    timeout_parsed = float(tval)
            except Exception:
                timeout_parsed = None
        args.llm_timeout = timeout_parsed or 60.0
    if getattr(args, "llm_max_output_tokens", None) in (None, ""):
        mt = g("LLM_MAX_OUTPUT_TOKENS")
        try:
            args.llm_max_output_tokens = int(mt) if mt is not None else None
        except Exception:
            args.llm_max_output_tokens = None

    # Run directory naming (timestamped)
    if getattr(args, "run_dir_root", None) in (None, ""):
        args.run_dir_root = g("RUN_DIR_ROOT")  # e.g., D:\nvAgent-main\logs\review_runs
    if getattr(args, "run_prefix", None) in (None, ""):
        args.run_prefix = g("RUN_PREFIX", "review_batch_")
    if getattr(args, "auto_run_dir", None) in (None, ""):
        b = _parse_bool(g("AUTO_RUN_DIR"))
        args.auto_run_dir = bool(b) if b is not None else False

    # Extras overrides (optional): allow enabling annotate/stacked/error via env
    # These flags are interpreted later in Query Analyzer, but we also pass via STYLE extras
    # They will be merged within run.py when building design spec.
    args._env_extras = {
        k: v
        for k, v in {
            "annotate_bars": _parse_bool(g("ANNOTATE_BARS")),
            "annotate_totals": _parse_bool(g("ANNOTATE_TOTALS")),
            "stacked": _parse_bool(g("STACKED")),
            "error_bars": _parse_bool(g("ERROR_BARS")),
        }.items()
        if v is not None
    }
    args._env_path = used_path

    # Iterative refinement controls
    if getattr(args, "max_iter", None) in (None, ""):
        v = g("MAX_ITER")
        try:
            args.max_iter = int(v) if v is not None else 3
        except Exception:
            args.max_iter = 3
    if getattr(args, "quality_threshold", None) in (None, ""):
        v = g("QUALITY_THRESHOLD")
        try:
            args.quality_threshold = float(v) if v is not None else 0.85
        except Exception:
            args.quality_threshold = 0.85
    if getattr(args, "quality_metric", None) in (None, ""):
        args.quality_metric = g("QUALITY_METRIC", "specification_adherence_score")

    # LLM provider defaults (openai-compatible fallback)
    if getattr(args, "llm_provider", None) in (None, ""):
        args.llm_provider = g("LLM_PROVIDER", "openai-compatible")
