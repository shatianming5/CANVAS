# VizFlow: End-to-End Visualization Agent Pipeline

This project implements a complete, runnable multi-agent pipeline based on the specification in `task.txt`.

Note: the local (non-LLM) implementation has been removed. The pipeline now uses an LLM HTTP API exclusively for all reasoning stages.

It orchestrates the following agents (via API):

- Query Analyzer
- Data Processor
- VizMapping Agent
- Search Agent (Matplotlib example generator)
- Design Explorer
- Code Generator
- Debug Agent
- Visual Evaluator

The pipeline accepts a natural language query and optional CSV data, plans a visualization, generates a Matplotlib plot, evaluates it, and emits structured JSON artifacts for each stage.

## Quick Start

Requirements: Python 3.9+

```
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: . .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

Run the full pipeline (uses synthetic sample data if no CSV in `data/`):

```
python run.py --query "Plot monthly sales by region as a grouped bar chart; color by region; annotate totals." --format png --dpi 160 --width 8 --height 5 --style seaborn --out-name myplot
```

Outputs are written to `outputs/`:

- `query_analyzer.json`
- `data_processor.json`
- `viz_mapping.json`
- `viz_mapping_paper.json`
- `search_agent_example.py`
- `search_agent_references.json`
- `design_explorer.json` (iteration snapshots: `design_explorer_iter_k.json`)
- `code_generator.json` (iteration snapshots: `code_generator_iter_k.json`)
- `debug_agent.json` (only if an error occurs)
- `visual_evaluator.json` (iteration snapshots: `visual_evaluator_iter_k.json`)
- `plot.png` / `plot.svg` / `plot.pdf` (configurable via `--format` and `--out-name`)
- `results.json` (index of all artifacts)
- `report.html` (single-page HTML with the figure and key JSONs)

You can also point the pipeline at a CSV file:

```
python run.py --query "Scatter plot of height vs weight by gender" --data data/people.csv --format svg
```

## .env Configuration (preferred)

You can configure all options via a `.env` file. This project prefers `D:\hope_last\.env` if it exists, otherwise falls back to `.env` in the current directory. CLI flags override `.env`.

Supported keys:

- Core pipeline:
  - `QUERY` (string)
  - `DATA` (path)
  - `OUT` (dir), `OUT_NAME` (filename base), `FORMAT` (png|svg|pdf), `DPI`, `WIDTH`, `HEIGHT`, `STYLE`
  - `CHART` (bar|line|scatter|histogram|box|heatmap|area)
  - `X`, `Y`, `COLOR`, `FACET`
  - `NO_REFINE` (true/false)
  - `MAX_ITER` (default 3)
  - `QUALITY_THRESHOLD` (default 0.85)
  - `QUALITY_METRIC` (default `specification_adherence_score`)
  - Extras: `ANNOTATE_BARS`, `ANNOTATE_TOTALS`, `STACKED`, `ERROR_BARS` (true/false)

- LLM API:
  - `LLM_PROVIDER` (`openai-compatible`, `google`, `anthropic`)
  - `BASE_URL` or `OPENAI_BASE_URL` (e.g., `https://api.zhizengzeng.com/v1`)
  - `API_KEY` or `OPENAI_API_KEY` (required)
  - `GEMINI_API_KEY` / `GOOGLE_API_KEY` (for Gemini)
  - `ANTHROPIC_API_KEY` (for Claude)
  - `MODEL_NAME` (e.g., `gpt-4o-mini`, `gemini-2.5-pro`, `claude-4-sonnet`)
  - `LLM_TIMEOUT` (seconds, or `connect,read` tuple)
  - `LLM_MAX_OUTPUT_TOKENS` (optional)

Example:

```
QUERY=Plot monthly sales by region as a grouped bar chart; color by region; annotate totals.
OUT=outputs
OUT_NAME=envplot
FORMAT=png
DPI=160
WIDTH=8
HEIGHT=5
STYLE=seaborn
CHART=bar
X=month
Y=sales
COLOR=region
MAX_ITER=3
QUALITY_THRESHOLD=0.85
QUALITY_METRIC=specification_adherence_score
LLM_PROVIDER=openai-compatible
# Optional:
# FACET=region
# NO_REFINE=false
# ANNOTATE_BARS=true
# ANNOTATE_TOTALS=true
# STACKED=false
# ERROR_BARS=false
# GEMINI_API_KEY=your_google_key
# ANTHROPIC_API_KEY=your_claude_key
```

## Data

Place CSV files in `data/`. If none are present or `--data` is not passed, the Data Processor will synthesize a small dataset based on the query and chart type.

## Project Structure

```
agents/
  __init__.py
  llm_orchestrator.py   # All agent logic via LLM API
  report_builder.py     # HTML report generator
utils/
  __init__.py
  io_utils.py
  config.py
  llm_client.py         # Generic OpenAI-compatible client
run.py                  # API-only pipeline entrypoint
requirements.txt
README.md
outputs/  # generated artifacts
data/     # optional input CSVs

```

## Self-Reflection Loop

The pipeline now runs a quality-driven loop inspired by the CoDA paper. Each iteration generates a new design, code plan, and plot, then evaluates semantic accuracy against the generated figure. Artifacts per iteration are written as `design_explorer_iter_k.json`, `code_generator_iter_k.json`, `generated_plot_iter_k.py`, and `visual_evaluator_iter_k.json`. The run stops early when the evaluator reports a metric (default `specification_adherence_score`) at or above the configured threshold.

Key controls:

- `--max-iter` / `MAX_ITER`
- `--quality-threshold` / `QUALITY_THRESHOLD`
- `--quality-metric` / `QUALITY_METRIC`

The best iteration is merged back into the canonical files (`design_explorer.json`, `code_generator.json`, `generated_plot.py`, `visual_evaluator.json`).

## Benchmark Metrics

A lightweight metrics toolkit lives under `bench/` to reproduce EPR/VSR/OS comparisons. Example usage:

```bash
python -m bench.run_metrics outputs/results.json other_logs/results.json
```

The script expects VizFlow `results.json` artifacts and inspects the linked `visual_evaluator` outputs to aggregate execution pass rate (EPR), visual success rate (VSR), and the average specification adherence score (OS).

```

## Notes

- The Matplotlib plotting uses a non-interactive backend (`Agg`) to save figures without a GUI.
- The Visual Evaluator performs semantic checks against the interpreted query and mapping via the API.
- The Debug Agent produces a structured fix plan if plotting fails and attempts a single automatic retry.
- Advanced CLI:
  - `--chart` to override detected plot type (bar|line|scatter|histogram|box|heatmap|area)
  - `--x/--y/--color` to override column mapping
  - `--facet` to create small multiples by a category (experimental)
  - `--style`, `--width`, `--height`, `--dpi`, `--format`, `--out-name`
  - `--max-iter` / `--quality-threshold` / `--quality-metric` to control the self-reflection loop (default 3 iterations, threshold 0.85 on `specification_adherence_score`)
  - `--llm-provider` to select `openai-compatible`, `google`, or `anthropic`
  - `--no-refine` to disable design refinement
  - Timestamped runs:
    - `--auto-run-dir` to create a timestamped subfolder for each run
    - `--run-dir-root` base folder for runs (e.g., `D:\nvAgent-main\logs\review_runs`)
    - `--run-prefix` name prefix (default `review_batch_`, final like `review_batch_20251006T045835Z`)

You can set the same via `.env`:

- `AUTO_RUN_DIR=true|false`
- `RUN_DIR_ROOT=D:\nvAgent-main\logs\review_runs`
- `RUN_PREFIX=review_batch_`
- `MAX_ITER=3`
- `QUALITY_THRESHOLD=0.85`
- `QUALITY_METRIC=specification_adherence_score`
- `LLM_PROVIDER=openai-compatible`
