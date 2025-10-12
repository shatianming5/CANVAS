from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _deep_get(container: Dict[str, Any], path: List[str]) -> Any:
    node: Any = container
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _deep_set(container: Dict[str, Any], path: List[str], value: Any, *, create_missing: bool = True) -> None:
    node: Any = container
    for idx, part in enumerate(path):
        is_last = idx == len(path) - 1
        if is_last:
            node[part] = value
            return
        if part not in node:
            if not create_missing:
                raise KeyError(f"path missing segment: {'/'.join(path[:idx + 1])}")
            node[part] = {}
        if not isinstance(node[part], dict):
            if not create_missing:
                raise TypeError(f"cannot descend into non-dict at {'/'.join(path[:idx + 1])}")
            node[part] = {}
        node = node[part]


def _deep_remove(container: Dict[str, Any], path: List[str]) -> None:
    if not path:
        return
    node: Any = container
    for idx, part in enumerate(path):
        is_last = idx == len(path) - 1
        if part not in node:
            return
        if is_last:
            node.pop(part, None)
            return
        node = node.get(part)
        if not isinstance(node, dict):
            return


@dataclass
class StyleSpec:
    """Structured style payload exchanged between agents."""

    payload: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "StyleSpec":
        return cls(
            {
                "style_spec_version": "1.0",
                "palette": {
                    "mode": "categorical",
                    "name": "tab10",
                    "n_colors": 10,
                    "colorblind_safe": True,
                    "contrast_ratio_min": 4.5,
                    "colors": [],
                    "color_mapping": {},
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
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.payload)

    def merge(self, other: Dict[str, Any]) -> None:
        def _merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
            for key, value in src.items():
                if isinstance(value, dict) and isinstance(dst.get(key), dict):
                    _merge(dst[key], value)  # type: ignore[arg-type]
                else:
                    dst[key] = copy.deepcopy(value)

        if isinstance(other, dict):
            _merge(self.payload, other)

    def apply_patch(self, ops: Iterable[Dict[str, Any]]) -> None:
        for op in ops:
            if not isinstance(op, dict):
                continue
            path_raw = op.get("path")
            if not isinstance(path_raw, str) or not path_raw.startswith("/"):
                continue
            path = [segment for segment in path_raw.split("/") if segment]
            operation = (op.get("op") or "").lower()
            if operation in {"add", "replace", "set", "update"}:
                _deep_set(self.payload, path, copy.deepcopy(op.get("value")))
            elif operation == "remove":
                _deep_remove(self.payload, path)

    def update_from_feedback(self, feedback: Optional[Dict[str, Any]]) -> None:
        if not feedback:
            return
        patch_ops = (
            feedback.get("patch_suggestions", {}).get("ops")
            if isinstance(feedback, dict)
            else None
        )
        if isinstance(patch_ops, list):
            self.apply_patch(patch_ops)


def load_style_spec(path: Path) -> StyleSpec:
    if not path.exists():
        return StyleSpec.default()
    return StyleSpec(json_load(path))


def json_load(path: Path) -> Dict[str, Any]:
    import json

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def json_dump(data: Dict[str, Any], path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
