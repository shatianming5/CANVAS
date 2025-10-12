from __future__ import annotations

from typing import Iterable, Optional

from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.collections import Collection, PathCollection
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


def _iter_axes(fig: Figure) -> Iterable[Axes]:
    axes = getattr(fig, "axes", [])
    for ax in axes:
        if isinstance(ax, Axes):
            yield ax


def _relative_luminance(rgba: Iterable[float]) -> float:
    r, g, b, _ = rgba
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def auto_tune_alpha(fig: Figure) -> None:
    """Heuristically lower alpha for dense plots to reduce overplotting."""
    if not isinstance(fig, Figure):
        return
    width, height = fig.get_size_inches()
    area = max(width * height, 1e-6)

    for ax in _iter_axes(fig):
        point_count = 0
        for coll in ax.collections:
            if isinstance(coll, PathCollection):
                offsets = coll.get_offsets()
                try:
                    point_count += len(offsets)
                except TypeError:
                    continue
        line_vertices = sum(len(line.get_xdata()) for line in ax.lines if isinstance(line, Line2D))
        bar_count = sum(
            1
            for patch in ax.patches
            if isinstance(patch, Rectangle) and patch.get_width() > 0 and patch.get_height() > 0
        )
        density = (point_count + bar_count * 2 + line_vertices * 0.25) / area

        if density > 120:
            scatter_alpha = 0.35
            bar_alpha = 0.55
            line_alpha = 0.65
        elif density > 40:
            scatter_alpha = 0.5
            bar_alpha = 0.65
            line_alpha = 0.75
        else:
            scatter_alpha = 0.85
            bar_alpha = 0.85
            line_alpha = 0.9

        for coll in ax.collections:
            if not isinstance(coll, Collection):
                continue
            current = coll.get_alpha()
            target = scatter_alpha
            if current is None or current > target:
                coll.set_alpha(target)

        for patch in ax.patches:
            if not isinstance(patch, Rectangle):
                continue
            current = patch.get_alpha()
            target = bar_alpha
            if current is None or current > target:
                patch.set_alpha(target)

        for line in ax.lines:
            if not isinstance(line, Line2D):
                continue
            current = line.get_alpha()
            target = line_alpha
            if current is None or current > target:
                line.set_alpha(target)

        for image in ax.images:
            if not isinstance(image, AxesImage):
                continue
            current = image.get_alpha()
            target = scatter_alpha if density > 40 else 0.9
            if current is None or current > target:
                image.set_alpha(target)


def _is_dark(color: Optional[Iterable[float]]) -> bool:
    if color is None:
        return False
    try:
        rgba = mcolors.to_rgba(color)
    except (ValueError, TypeError):
        return False
    return _relative_luminance(rgba) < 0.3


def adjust_local_contrast(fig: Figure) -> None:
    """Strengthen edges/text for extremely dark fills to preserve readability."""
    if not isinstance(fig, Figure):
        return

    for ax in _iter_axes(fig):
        for patch in ax.patches:
            if not isinstance(patch, Patch):
                continue
            face = patch.get_facecolor()
            if not face:
                continue
            try:
                rgba = mcolors.to_rgba(face)
            except (ValueError, TypeError):
                continue
            if rgba[3] == 0:
                continue
            luminance = _relative_luminance(rgba)
            if luminance < 0.2:
                patch.set_edgecolor("#f5f5f5")
                if (patch.get_linewidth() or 0) < 0.8:
                    patch.set_linewidth(0.8)
            elif luminance > 0.92:
                patch.set_edgecolor("#424242")
                if (patch.get_linewidth() or 0) < 0.6:
                    patch.set_linewidth(0.6)

        for text in ax.texts:
            color = text.get_color()
            try:
                rgba = mcolors.to_rgba(color)
            except (ValueError, TypeError):
                continue
            luminance = _relative_luminance(rgba)
            if 0.4 < luminance < 0.65:
                text.set_color("#202020")
            elif 0.2 < luminance <= 0.4:
                text.set_color("#f4f4f4")
            if text.get_fontsize() < 11:
                text.set_fontsize(max(text.get_fontsize(), 11))

        for spine in ax.spines.values():
            spine.set_alpha(0.7)
