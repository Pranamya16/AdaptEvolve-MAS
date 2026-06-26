# AdaptEvolve-MAS — Streamlit App Plan

## Overview

A full-stack Streamlit web application that wraps the AdaptEvolve-MAS pipeline
from `AE2_upgraded.ipynb`, replacing the Gradio interface with a persistent,
multi-user web app with login and session history.

## Source of Truth

The canonical pipeline lives in `AE2_upgraded.ipynb`.
`streamlit_app/adaptevolve_core.py` is extracted from it (cells 1–10, Colab
imports removed).  **Do not hand-edit `adaptevolve_core.py`** — re-run the
extraction if the notebook changes.

## File Structure

```
BE Project/
├── AE2_upgraded.ipynb          ← canonical source
├── adaptevolve_core.py         ← extracted pipeline (auto-generated)
├── STREAMLIT_PLAN.md           ← this file
└── streamlit_app/
    ├── app.py                  ← entry point: auth + routing
    ├── config.yaml             ← streamlit-authenticator credentials
    ├── adaptevolve.db          ← SQLite session history (auto-created)
    ├── database.py             ← DB layer
    ├── evolution_runner.py     ← thread+queue wrapper for streaming
    ├── requirements_app.txt
    ├── components/
    │   ├── progress_panel.py   ← per-cycle step card
    │   ├── score_chart.py      ← Plotly trajectory chart
    │   └── code_viewer.py      ← syntax-highlighted code + download
    └── pages/
        ├── chat.py             ← Tab 1: Chat (LLM Q&A + file context)
        ├── evolve.py           ← Tab 2: Quick Run (evolution UI)
        ├── documents.py        ← Tab 3: Documents (RAG upload)
        ├── monitor.py          ← Tab 4: Monitor (live status)
        ├── agents_info.py      ← Tab 5: Agents (static info)
        └── settings.py         ← Tab 6: Settings (API key + weights)
```

## The 6 Tabs (matching Gradio UI)

| Tab | Page | What it does |
|---|---|---|
| 💬 Chat | chat.py | LLM Q&A with file context + slash commands (/run, /status, /solution, /help, /reset) |
| 🚀 Quick Run | evolve.py | Goal input + sliders → run evolution → real-time step cards → final report + charts |
| 📄 Documents | documents.py | Upload file or paste text into RAG vector store + view stats |
| 📊 Monitor | monitor.py | Score trajectory, current best solution, cycle history |
| 🤖 Agents | agents_info.py | Static description of all 4 agents + Lyra 4-D methodology |
| ⚙️ Settings | settings.py | Update Gemini API key + evaluation criteria weights |

## Authentication

- Library: `streamlit-authenticator` 0.3.x
- Config: `streamlit_app/config.yaml` (bcrypt-hashed passwords)
- Users: pranamya (Pranamya@2024), demo_user (Demo@1234), admin (Admin@9999)
- Session: JWT cookie, 7-day expiry

## Pipeline Integration

`adaptevolve_core.py` exposes these globals used by the pages:
- `session` — the global `AdaptEvolveSession` instance
- `chat_interface` — the `ChatInterface` wrapping `session`
- `rag_system` — the `AgenticRAG` instance
- `evolution_graph` — the compiled LangGraph
- `create_initial_state(goal, max_cycles)` — state factory
- `build_prompt(history, file_context, query)` — chat prompt builder
- `generate_response(prompt)` — Gemini call
- `MODEL_NAME`, `API_KEY`, `client`, `llm` — LLM globals

## Non-Blocking Evolution

`evolution_runner.py` wraps `evolution_graph.stream()` in a daemon thread
with a `queue.Queue`. The Quick Run page drains the queue every 1.5s via
`st.rerun()` and renders each step as a collapsible card.

## Session History

SQLite (`adaptevolve.db`) stores completed evolution sessions:
- `session_id`, `username`, `goal`, `started_at`, `finished_at`
- `final_score`, `n_cycles_run`, `status`, `best_code`
- `score_trajectory` (JSON), `dimension_scores` (JSON), `mechanic_log` (JSON)

## LLM Backend (Local Ollama — default — or Cloud Gemini)

The pipeline runs through a **pluggable LLM backend** so it works fully offline
with no API quota. Selected via env vars (or the Settings tab at runtime):

| Env var | Default | Meaning |
|---|---|---|
| `AE_LLM_BACKEND` | `ollama` | `ollama` (local) or `gemini` (cloud) |
| `AE_LLM_MODEL` | `qwen2.5-coder:7b-instruct` | local model (Ollama tag) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `AE2` / `GEMINI_API_KEY` | — | only needed when backend = `gemini` |

Implementation: `adaptevolve_core.py` exposes `GeminiLLM`, `OllamaLLM`, a shared
`_parse_json_response()` helper, and `build_llm()`. The global `llm` is built by
`build_llm()`, so all agents/RAG/evolution are backend-agnostic. `OllamaLLM`
uses Ollama's `format="json"` mode for reliable structured output (Mechanic).

### One-time local model setup (needs internet once, then fully offline)

1. Install **Ollama**: https://ollama.com/download (Windows installer).
2. Pull the model (≈4.7 GB for 7B):
   ```bash
   ollama pull qwen2.5-coder:7b-instruct      # quality (default)
   # or, for speed on a 4GB GPU (fits entirely in VRAM):
   ollama pull qwen2.5-coder:3b-instruct
   ```
3. Verify: `ollama list` should show the model. Ollama runs as a background
   service; no internet needed after the pull.

**Hardware note (16 GB RAM, 4 GB VRAM):** the 7B (Q4, ~4.5 GB) spills partly
into system RAM → slower but better quality and JSON reliability. The 3B fits
fully in 4 GB VRAM → fast. Switch models anytime in the Settings tab.

**Keep weights inside the project (optional):** set `OLLAMA_MODELS` to a folder
under the project before pulling, e.g. (PowerShell)
`setx OLLAMA_MODELS "d:\Old_Documents\BE Project\ollama_models"`.

## How to Run

```bash
# 1. Make sure Ollama is running and the model is pulled (see above).
# 2. From the BE Project root directory:
streamlit run streamlit_app/app.py
```

Default backend is local Ollama — no key required. To use Gemini instead, set
`AE_LLM_BACKEND=gemini` and provide a key in `.streamlit/secrets.toml`:
```toml
AE2 = "your-gemini-api-key"
```
OR export it: `export AE2=your-gemini-api-key`. You can also switch backend and
model live from the **Settings** tab.

## Dependencies

Install with:
```bash
pip install -r streamlit_app/requirements_app.txt
pip install google-genai langgraph sentence-transformers numpy duckduckgo-search
```
