from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import json


def build_html_report(outputs_dir: Path) -> str:
    p = outputs_dir
    def jread(name: str) -> Any:
        f = p / name
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    qa = jread("query_analyzer.json")
    dp = jread("data_processor.json")
    vm = jread("viz_mapping.json")
    vmp = jread("viz_mapping_paper.json")
    de = jread("design_explorer.json")
    der = jread("design_explorer_refined.json")
    cg = jread("code_generator.json")
    ve = jread("visual_evaluator.json")
    ss = jread("style_spec.json")
    ssr = jread("style_spec_refined.json")
    res = jread("results.json")

    # Detect image file (prefer results.json path if present)
    img_src = None
    if res and res.get("outputs", {}).get("image"):
        img_src = res["outputs"]["image"]
    else:
        for ext in ("png", "svg", "pdf"):
            f = p / f"plot.{ext}"
            if f.exists():
                img_src = str(f)
                break

    def section(title: str, content: str) -> str:
        return f"<h2>{title}</h2>\n<div style='padding-left:8px'>{content}</div>"

    html = [
        "<html><head><meta charset='utf-8'><title>VizFlow Report</title>",
        "<style>body{font-family:Segoe UI,Arial,sans-serif;padding:16px;} pre{background:#f7f7f7;padding:8px;border-radius:4px;overflow:auto;} img{max-width:100%;height:auto;border:1px solid #ddd;} .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;}</style>",
        "</head><body>",
        "<h1>VizFlow Report</h1>",
    ]

    if qa:
        html.append(section("Query Analyzer", f"<pre>{json.dumps(qa, ensure_ascii=False, indent=2)}</pre>"))
    if dp:
        html.append(section("Data Processor", f"<pre>{json.dumps(dp, ensure_ascii=False, indent=2)}</pre>"))
    if vm:
        html.append(section("Viz Mapping", f"<pre>{json.dumps(vm, ensure_ascii=False, indent=2)}</pre>"))
    if vmp:
        html.append(section("Viz Mapping (Paper Schema)", f"<pre>{json.dumps(vmp, ensure_ascii=False, indent=2)}</pre>"))
    if de:
        html.append(section("Design Explorer", f"<pre>{json.dumps(de, ensure_ascii=False, indent=2)}</pre>"))
    if der:
        html.append(section("Design Refined", f"<pre>{json.dumps(der, ensure_ascii=False, indent=2)}</pre>"))
    if ss:
        html.append(section("Style Spec", f"<pre>{json.dumps(ss, ensure_ascii=False, indent=2)}</pre>"))
    if ssr:
        html.append(section("Style Spec Refined", f"<pre>{json.dumps(ssr, ensure_ascii=False, indent=2)}</pre>"))
    if cg:
        html.append(section("Code Generator", f"<pre>{json.dumps(cg, ensure_ascii=False, indent=2)}</pre>"))
    if ve:
        html.append(section("Visual Evaluator", f"<pre>{json.dumps(ve, ensure_ascii=False, indent=2)}</pre>"))

    if img_src:
        html.append(section("Visualization", f"<img src='{img_src}' alt='plot'>"))

    html.append("</body></html>")
    return "\n".join(html)

