# VizFlow: End-to-End Visualization Agent Pipeline

VizFlow is a fully LLM-driven recreation of the CoDA multi-agent visualization system described in `task.txt`. The legacy local heuristics were removed; every reasoning step flows through an HTTP LLM API while still producing runnable Matplotlib code and structured artifacts.

## Overview
- Multi-agent pipeline orchestrated from `run.py` that turns a natural-language visualization request (plus optional CSV) into a Matplotlib figure and JSON traces.
- Pipeline stages mirror the CoDA specification: query analysis, data preparation, mapping, exemplar search, iterative design/code/debug loop, and visual evaluation.
- Maintains dataset registries, upgraded FigureSpec v1.2 mappings, style context, and heatmap shaping helpers so that code generation stays grounded in real columns.
- Supports OpenAI-compatible, Google Gemini, and Anthropic Claude APIs via `utils.llm_client.LLMClient` with optional model fallback detection and error logging.
- Ships with benchmarking utilities (EPR/VSR/OS), curated Matplotlib references, sample datasets, and HTML reporting to inspect each agent’s output.

## Repository Layout
```
agents/
  llm_orchestrator.py    # Prompts and LLM-facing helpers for every agent
  report_builder.py      # HTML dashboard collating JSON artifacts
bench/
  metrics.py             # Benchmark record model and metric aggregation
  run_metrics.py         # CLI helper to compute EPR/VSR/OS from results.json files
data/
  monthly_sales_profit.csv
  sample_sales.csv
scripts/
  figure1d_caption_manual.py  # Manual Matplotlib recreation of Figure 1d
utils/
  config.py              # .env loading and CLI default injection
  io_utils.py            # JSON/Text I/O conveniences
  llm_client.py          # REST client for OpenAI-compatible, Gemini, and Claude
  spec_adapter.py        # Translate FigureSpec → paper schema summaries
  web_search.py          # Static Matplotlib documentation references
run.py                   # Single entry point that orchestrates all agents
requirements.txt
README.md
task.txt                 # Original CoDA-aligned plan (Chinese)
```

## Pipeline Architecture

### Agent Flow
1. **Query Analyzer (`agents.llm_orchestrator.llm_query_analyzer`)**  
   Decomposes the user prompt into an interpreted intent, chart recommendation, constraints, and a prioritized TODO list. The output seeds later agents and captures extras such as annotation wishes or accessibility targets.
2. **Data Processor (`llm_data_processor`)**  
   Loads or synthesizes tabular data, emits a cleaned `pandas.DataFrame`, and returns dataset/field registries plus quality issues. `_ensure_registry_from_df` enriches the registry with inferred column roles and aggregates.
3. **Visualization Mapping (`llm_viz_mapping`)**  
   Generates a FigureSpec-like mapping that is upgraded to v1.2 via `upgrade_to_v1_2`, reconciled against the dataset registry, and validated by `validate_spec_vs_registry`. The spec is also converted into a paper-schema summary through `figure_spec_to_paper_schema` / `paper_schema_summary`. Heatmap-specific helpers (`_prepare_heatmap_dataframe`, `_order_heatmap_columns`, `_format_heatmap_condition`) restructure genomics-style matrices into long-form data when `rect` geoms are detected.
4. **Search Agent (`llm_search_agent` + `utils.web_search.official_references`)**  
   Requests LLM-generated exemplar code and augments it with curated Matplotlib documentation links so every run retains reproducible references.
5. **Aesthetic Stylist (`llm_aesthetic_stylist` / `llm_aesthetic_stylist_refine`)**  
   Produces a structured StyleSpec (palette, transparency, statistical annotations, layout, accessibility) aligned with declared policies and stage feedback. StyleSpec patches are merged across iterations and persisted for auditing.
6. **Iterative Design & Code Generation (`llm_design_explorer` → `llm_code_generator` → execution + `llm_debug_agent`)**  
   For each iteration, the design explorer expands layout plans while consuming the latest StyleSpec; the stylist delivers refreshed directives which the code generator is instructed to honour explicitly. Generated Matplotlib code runs inside a controlled namespace (`_build_exec_locals`). Execution failures trigger the debug agent, which proposes fixes that are re-run automatically.
7. **Visual Evaluation (`llm_staged_visual_evaluator`)**  
   The evaluator now returns four-layer diagnostics (L1 orchestration → L4 polish) plus an overall score, taking StyleSpec, policies, and TODO items into account. Each layer surfaces risks, JSON patch suggestions, and a recommended target stage while preserving the legacy `semantic_accuracy` block for compatibility.
8. **Reporting & Indexing**  
   Once stopping criteria are met, `run.py` writes canonical artifacts (`design_explorer.json`, `generated_plot.py`, `visual_evaluator.json`, etc.), builds `results.json`, and calls `agents.report_builder.build_html_report` to assemble `report.html`. Any API fallbacks are captured in `llm_errors.json`.

### Iterative Self-Reflection Loop
- Controlled by `--max-iter` (default 3) and `--quality-threshold` (default 0.85). The loop exits early when the evaluator metric meets the threshold or the iteration budget is exhausted.
- Each cycle stores intermediate files: `design_explorer_iter_k.json`, `code_generator_plan_iter_k.json`, `generated_plot_iter_k.py`, and `visual_evaluator_iter_k.json`.
- Feedback wiring mirrors CoDA: evaluator findings, debug patches, and outstanding TODO items are forwarded into the next design iteration via the `feedback` payload.
- The best iteration (highest metric) is promoted to the canonical artifacts while still preserving every iteration’s JSON/PNG for auditing.

### Data & Spec Handling
- Dataset and field registries are merged with runtime heuristics so the mapping always references valid columns and their inferred semantic roles.
- `reconcile_mapping_with_registry` and `validate_spec_vs_registry` guard against stale column names and inconsistent axis assignments.
- `_extract_style_context` collates global and per-layer style hints so the design/code agents can respect palette, alpha, and other aesthetic decisions.
- Heatmap-specific preprocessing derives tidy data and categorical ordering, including manual `HEATMAP_CELL_ORDER` / `HEATMAP_DRUG_ORDER` lists aligned with the sample Nature dataset.

### Reference Mining & Reporting
- The search agent output is augmented with static gallery links in `search_agent_references.json` and executable snippets in `search_agent_example.py`.
- `agents.report_builder.build_html_report` collates the main JSON outputs and the generated figure into a single inspection page.
- `llm_errors.json` records per-stage fallbacks whenever `_with_model_fallback` needs to switch models or recover from API errors.

## Configuration & Secrets
- `.env` defaults are injected by `utils.config.apply_env_defaults`. Load order: `ENV_PATH` (if set) → `D:\hope_last\.env` → local `.env`. CLI flags always take precedence.
- Core keys:  
  `QUERY`, `DATA`, `OUT`, `OUT_NAME`, `FORMAT`, `DPI`, `WIDTH`, `HEIGHT`, `CHART`, `X`, `Y`, `COLOR`, `FACET`, `NO_REFINE`, `MAX_ITER`, `QUALITY_THRESHOLD`, `QUALITY_METRIC`.
- Styling booleans (`ANNOTATE_BARS`, `ANNOTATE_TOTALS`, `STACKED`, `ERROR_BARS`) and style names are surfaced to the design explorer via the `_env_extras` payload.
- LLM credentials & behaviour:  
  - `LLM_PROVIDER` = `openai-compatible` (default) | `google` | `anthropic`  
  - `BASE_URL` / `OPENAI_BASE_URL`, `API_KEY` / `OPENAI_API_KEY` (or provider-specific keys like `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`)  
  - `MODEL_NAME`, `LLM_TIMEOUT`, `LLM_MAX_OUTPUT_TOKENS`
- `LLMClient.from_env` automatically configures headers/endpoints for each provider and can list available models to aid debugging.

Example `.env`:

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

## Running the Pipeline

### Quick Start
```
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: . .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

### Example Runs
```
python run.py --query "Plot monthly sales by region as a grouped bar chart; color by region; annotate totals." \
  --format png --dpi 160 --width 8 --height 5 --style seaborn --out-name myplot
```

Point the pipeline at a CSV (Excel sheets are also supported via `--sheet`):

```
python run.py --query "Scatter plot of height vs weight by gender" --data data/people.csv --format svg
```

### Advanced CLI Flags
- `--chart`, `--x`, `--y`, `--color`, `--facet` override automatic mappings.
- `--no-refine` disables the iterative refinement loop.
- `--max-iter`, `--quality-threshold`, `--quality-metric` customise the self-reflection policy.
- `--llm-provider` selects a configured provider profile.
- Run-directory controls: `--auto-run-dir`, `--run-dir-root`, `--run-prefix` create timestamped subfolders (e.g. `logs/review_batch_20251012T054500Z`).

## Outputs
VizFlow writes all artifacts to the chosen `out/` directory:

- Core JSON traces: `query_analyzer.json`, `data_processor.json`, `viz_mapping.json`, `viz_mapping_paper.json`, `design_explorer.json`, `style_spec.json`, `code_generator.json`, `visual_evaluator.json`.
- Iteration snapshots: `design_explorer_iter_k.json`, `style_spec_iter_k.json`, `code_generator_plan_iter_k.json`, `generated_plot_iter_k.py`, `visual_evaluator_iter_k.json`, `debug_agent_iter_k.json` (当发生错误时)。
- Code & media: `generated_plot.py`, `plot.<format>`, per-iteration `generated_plot_iter_k.py`, and the processed CSV (`processed_data.csv`).
- Index & reporting: `results.json` (paths + best iteration metadata), `search_agent_example.py`, `search_agent_references.json`, `report.html`, optional `style_spec_refined.json`, `llm_errors.json`.

## Benchmarking & Utilities
- `python -m bench.run_metrics outputs/results.json` compares VizFlow runs and reports Execution Pass Rate (EPR), Visual Success Rate (VSR), and Overall Score (OS).
- `bench.metrics.BenchmarkRecord` can be imported to aggregate multiple experiment folders programmatically.
- `scripts/figure1d_caption_manual.py` recreates the Nature Figure 1d example using the bundled CSV and demonstrates conventional Matplotlib authoring for comparison.

## Data
- `data/monthly_sales_profit.csv` and `data/sample_sales.csv` provide quick-start datasets.
- If `--data` is omitted, `llm_data_processor` synthesizes a small dataset consistent with the inferred chart type.

## Troubleshooting & Development Notes
- `llm_errors.json` captures any stage that required model fallback or returned API errors.
- `debug_agent.json` (or per-iteration variants) records automatic fix attempts when the generated Matplotlib fails.
- `processed_data.csv` mirrors the Data Processor output so you can inspect exactly what downstream agents consumed.
- The pipeline uses a non-interactive Matplotlib backend (`Agg`) and writes figures directly; no GUI is required.
- Re-run with `--auto-run-dir` to keep historical runs separated without manual folder management.
