# AdaptEvolve-MAS — Experiment Sprint

This adds a reproducible, resumable experiment harness around your existing
pipeline **without changing the pipeline logic or the Gradio UI**. The pipeline
is extracted verbatim from `AE2_upgraded (3).ipynb` into an importable module
(`adaptevolve_core.py`) with a few surgical hooks added (swappable LLM backend,
NoMeta / NoRAG ablation switches, ground-truth correctness tests, and cost knobs).

## Files

| File | Purpose |
|------|---------|
| `providers.py` | Swappable LLM backends: Gemini, Qwen-on-GPU, any OpenAI-compatible API. Adds a namespaced response cache + RPM rate limiter. Drop-in for the old `GeminiLLM`. |
| `tasks.py` | Ground-truth test suites for the 3 tasks (sorting / KMP / LRU), with reference + broken implementations. |
| `build_core.py` | Regenerates `adaptevolve_core.py` from the notebook (verbatim extract + asserted patches). Re-run if you change the notebook. |
| `adaptevolve_core.py` | **Auto-generated.** The importable pipeline (state, agents, LangGraph, session) with the experiment hooks. |
| `experiment_utils.py` | The matrix runner + resumable JSONL/CSV logging. |
| `run_experiments.py` | Headless entry point (CLI). |
| `smoke_test.py` | Run this FIRST to validate your environment + integration. |
| `analyze_results.py` | Aggregates logs → LaTeX tables (+ optional plots). |

## 1. Install dependencies

```bash
pip install google-genai langgraph langchain-core langchain-community \
            sentence-transformers numpy duckduckgo-search
pip install matplotlib            # optional, for plots
# For the Qwen backend only:
pip install torch transformers accelerate bitsandbytes
# For an OpenAI-compatible backend (OpenRouter/Groq/...) only:
pip install openai
```

## 2. Choose a backend (env vars)

The backend is selected at import time from environment variables, so the same
code runs locally or on Colab.

**Gemini (recommended; run locally — no GPU needed):**
```bash
export AE_BACKEND=gemini
export AE_GEMINI_MODEL=gemini-2.0-flash      # stable, higher free limits
export GEMINI_API_KEY=...                     # or AE2=...  (Colab secret "AE2" also works)
```

**Qwen on a Colab T4 (no API limits, slower):**
```python
import os
os.environ["AE_BACKEND"] = "qwen"
os.environ["AE_QWEN_CACHE_DIR"] = "/content/drive/MyDrive/hf_cache"   # cache weights on Drive
# (mount Drive first: from google.colab import drive; drive.mount('/content/drive'))
```

**"Other" (OpenRouter / Groq / paid Gemini, etc.):**
```bash
export AE_BACKEND=openrouter            # or groq
export OPENAI_API_KEY=...
# then pass model_name in make_provider, or tell me the model and I'll wire a default.
```

## 3. Run

```bash
python smoke_test.py            # validate: provider ping + ground-truth + 1 short run
python run_experiments.py       # full 27-run matrix (3 tasks x 3 conditions x 3 seeds)
python analyze_results.py       # -> results/tables/*.tex (+ results/figures/*.png)
```

On Colab, after uploading the `.py` files and mounting Drive, the same commands
work in a cell via `!python run_experiments.py`, or:
```python
import adaptevolve_core as core, experiment_utils as ex
ex.run_matrix(core.evolution_graph, core.create_initial_state, core.llm)
```

### Cost knobs (defaults already trimmed for the free tier)
`max_cycles=4`, `population_size=3`, `num_generations=2` → roughly ~50 LLM calls
per run, ~1,350 for the full matrix. Override via `--max-cycles/--pop/--gens` or
`AE_MAX_CYCLES/AE_POP/AE_GENS`.

### Resumability
`results/runs.jsonl` is appended after each run; re-running skips any
`(task, condition, seed)` already completed. A Colab disconnect or rate-limit
pause never loses finished runs. The response cache (`results/llm_cache/`) is
namespaced per `(task, condition, seed)`, so re-running a specific seed is free
while different seeds stay independent (preserving the mean±std).

## 4. Put the numbers in the paper
`analyze_results.py` writes `results/tables/{ablation,trajectory,dimensions,weights,operators}.tex`.
Send me the `results/` folder (or just `summary.csv` + the `tables/`), and I'll
wire the **real** numbers into `main_upgraded.tex` (and update the abstract /
discussion to match whatever the data actually shows — including honestly
reporting any ablation that turns out not to help).
