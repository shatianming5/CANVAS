from __future__ import annotations

from typing import Any, Dict, List, Tuple


def _ensure(mapping: Dict[str, Any], path: List[str], default: Any) -> Any:
    node = mapping
    for i, key in enumerate(path):
        if i == len(path) - 1:
            if key not in node or node.get(key) is None:
                node[key] = default
            return node[key]
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    return node


def apply_patch_ops(
    mapping_spec: Dict[str, Any],
    style_spec: Dict[str, Any],
    patch_ops: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """Apply evaluator patch ops to mapping_spec/style_spec conservatively.

    Returns: (new_mapping, new_style, notes)
    """
    notes: List[str] = []
    mapping_out = dict(mapping_spec or {})
    style_out = dict(style_spec or {})

    for op in patch_ops or []:
        if not isinstance(op, dict):
            continue
        optype = str(op.get("type") or "").lower()
        target = str(op.get("target") or "").lower()
        style = op.get("style") or {}

        if optype in {"update_color_scheme", "color_scheme", "palette_update"}:
            pal = style.get("palette") or style.get("name") or style.get("value")
            if pal:
                _ensure(style_out, ["palette"], {})["name"] = pal
                # mark as colorblind safe when common safe palettes are chosen
                safe_names = {"tab10", "tab20", "tableau", "colorblind_safe", "okabe_ito"}
                _ensure(style_out, ["palette"], {})["colorblind_safe"] = pal in safe_names
                notes.append(f"palette -> {pal}")

        elif optype in {"enhance_legend", "legend_clarify"}:
            pos = style.get("position") or style.get("loc")
            framealpha = style.get("framealpha")
            acc = _ensure(style_out, ["accessibility"], {})
            if pos:
                acc["legend_position"] = pos
            if isinstance(framealpha, (int, float)):
                acc["legend_framealpha"] = float(framealpha)
            notes.append("legend tuned")

        elif optype in {"rearrange_subplots", "layout_spacing"}:
            sp = style.get("spacing") or style.get("wspace")
            lay = _ensure(style_out, ["layout_tuning"], {})
            if sp == "increased":
                lay["wspace"] = 0.3
                lay["hspace"] = 0.35
            elif isinstance(sp, (int, float)):
                lay["wspace"] = float(sp)
                lay["hspace"] = float(sp)
            notes.append("layout spacing adjusted")

        elif optype in {"standardize_axis_labels", "axis_labels"}:
            if "x_label" in style or "xlabel" in style:
                _ensure(mapping_out, ["mapping", "axes", "x"], {})["label"] = style.get("x_label") or style.get("xlabel")
            if "y_label" in style or "ylabel" in style:
                _ensure(mapping_out, ["mapping", "axes", "y"], {})["label"] = style.get("y_label") or style.get("ylabel")
            notes.append("axis labels synchronized")

        elif optype in {"center_color_scale", "calibrate_color"}:
            center = style.get("center")
            enc = _ensure(mapping_out, ["mapping", "encodings"], {})
            color_by = enc.get("color_by") if isinstance(enc.get("color_by"), dict) else {}
            prev_scale = color_by.get("scale") if isinstance(color_by, dict) else None
            if not isinstance(prev_scale, dict):
                prev_scale = {}
            if center is not None:
                try:
                    cval = float(center) if isinstance(center, (int, float, str)) else None
                except Exception:
                    cval = None
                prev_scale["center"] = cval if cval is not None else "mean"
            prev_scale.setdefault("type", "diverging")
            enc["color_by"] = {"field": color_by.get("field", "z_score"), "type": "quantitative", "scale": prev_scale}
            notes.append("color scale centered")

        elif optype in {"add_statistical_summary", "error_bars"}:
            sa = _ensure(style_out, ["stat_annotations"], {})
            if style.get("mean"):
                sa["trend"] = "mean"
            if style.get("sd") or style.get("std") or style.get("stderr"):
                sa["enable_confidence_interval"] = True
            notes.append("stat annotations enabled")

        elif optype in {"standardize_tick_formatting", "tick_format"}:
            fmt = style.get("format") or "plain"
            lay = _ensure(style_out, ["layout_tuning", "ticks"], {})
            lay["format_y"] = fmt
            lay["format_x"] = style.get("format_x", lay.get("format_x", "plain"))
            notes.append("tick format standardized")

        elif optype in {"optimize_annotation_placement", "annotation_placement"}:
            _ensure(style_out, ["accessibility"], {})["avoid_overlap"] = True
            notes.append("annotation placement optimized")

        elif optype in {"change_plot_type"}:
            # rarely used; if requested, mutate primary layer geom
            new_geom = str(style.get("geom") or style.get("type") or "").lower()
            try:
                layers = _ensure(mapping_out, ["mapping", "layers"], [])
                if isinstance(layers, list) and layers:
                    layers[0]["geom"] = new_geom or layers[0].get("geom")
                    notes.append(f"geom -> {new_geom}")
            except Exception:
                pass

    return mapping_out, style_out, notes
