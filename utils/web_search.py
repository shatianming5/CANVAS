from __future__ import annotations

from typing import Dict, List


_MATPLOTLIB_DOCS: Dict[str, List[Dict[str, str]]] = {
    "bar": [
        {
            "title": "Matplotlib Bar Charts",
            "url": "https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_chart.html",
            "reason": "Official gallery example covering grouped and stacked bar charts.",
        }
    ],
    "line": [
        {
            "title": "Matplotlib Line Demo",
            "url": "https://matplotlib.org/stable/gallery/lines_bars_and_markers/simple_plot.html",
            "reason": "Canonical multi-line plotting example.",
        }
    ],
    "scatter": [
        {
            "title": "Matplotlib Scatter",
            "url": "https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter.html",
            "reason": "Official scatter plot reference including colormap usage.",
        }
    ],
    "histogram": [
        {
            "title": "Matplotlib Histogram",
            "url": "https://matplotlib.org/stable/gallery/statistics/hist.html",
            "reason": "Histogram basics with bins and normalization options.",
        }
    ],
    "box": [
        {
            "title": "Matplotlib Boxplot",
            "url": "https://matplotlib.org/stable/gallery/statistics/boxplot.html",
            "reason": "Official guide to box-and-whisker plots.",
        }
    ],
    "heatmap": [
        {
            "title": "Matplotlib Heatmap",
            "url": "https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html",
            "reason": "Annotated heatmap example with colorbar configuration.",
        }
    ],
    "pie": [
        {
            "title": "Matplotlib Pie Charts",
            "url": "https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html",
            "reason": "Official pie chart reference with autopct usage.",
        }
    ],
}


def official_references(plot_type: str) -> List[Dict[str, str]]:
    """Return curated Matplotlib documentation links for a given plot type."""
    key = (plot_type or "").strip().lower()
    return _MATPLOTLIB_DOCS.get(key, [])
