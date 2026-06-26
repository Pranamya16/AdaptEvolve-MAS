# AdaptEvolve-MAS

**AdaptEvolve-MAS** is a meta-evolutionary multi-agent framework for LLM-driven code optimization with verifiable AI safety properties.

Four role-specialized agents — **Strategist** (Agentic RAG + web search), **Solver** (evolutionary programming), **Judge** (multi-criteria evaluation), and **Mechanic** (Online Objective Correction) — run in a stateful cyclic [LangGraph](https://github.com/langchain-ai/langgraph) workflow.

The Mechanic agent is the core safety contribution: it reads cross-cycle performance history and dynamically re-weights evaluation criteria mid-run — without fine-tuning, without human labels, and without halting the optimization loop.

> Paper: *"AdaptEvolve-MAS: Online Objective Correction in Meta-Evolutionary Multi-Agent Code Optimization"* — targeting SafeAI @ AAAI 2026.

---

## Repository layout

```
adaptevolve_core.py          # Full 4-agent pipeline (Strategist→Solver→Judge→Mechanic)

streamlit_app/
  app.py                     # Entry point: auth + page router
  database.py                # SQLite session + chat history layer
  evolution_runner.py        # Thread wrapper around the evolution graph
  requirements_app.txt       # App dependencies
  config.example.yaml        # Copy to config.yaml and fill in credentials
  _pages/                    # Chat, Evolve, Dashboard, History, Settings, ...
  components/                # Reusable UI components

experiments/
  tasks.py                   # Ground-truth task definitions + test suites
  run_ablation.py            # 27-run ablation (3 tasks × 3 conditions × 3 seeds)
  safety_scenarios.py        # SS-1 through SS-4 adversarial safety tests
  analyze_results.py         # Produces LaTeX tables + matplotlib figures

paper/
  main.tex                   # Full AdaptEvolve-MAS paper (AAAI 2026 format)
  references.bib             # Bibliography

AE2_upgraded.ipynb           # Original implementation notebook
```

---

## Safety properties

| Property | Description |
|---|---|
| **OOC** — Online Objective Correction | Mechanic re-weights evaluation criteria each cycle based on observed performance |
| **BA** — Bounded Autonomy | Hard `max_cycles` ceiling enforced by LangGraph conditional edge |
| **IDT** — Interpretable Decision Trails | Every Mechanic proposal logged with a natural-language rationale |
| **CR** — Corrigible Restructuring | Adaptation confined to config layer; agent graph is immutable during a run |

---

## Quick start (Streamlit app)

```bash
# 1. Install dependencies
pip install -r streamlit_app/requirements_app.txt

# 2. Set up credentials
cp streamlit_app/config.example.yaml streamlit_app/config.yaml
# Edit config.yaml with your username/password (see comments inside)

# 3. Set LLM backend (Ollama local, default)
ollama pull qwen2.5-coder:7b-instruct
set AE_LLM_BACKEND=ollama       # Windows
# export AE_LLM_BACKEND=ollama  # Linux/Mac

# 4. Run
streamlit run streamlit_app/app.py
```

**Using Gemini instead:**
```bash
set AE_LLM_BACKEND=gemini
set GEMINI_API_KEY=your_key_here
streamlit run streamlit_app/app.py
```

---

## Running experiments (on Google Colab / GPU)

```bash
# Dry run (validate imports)
python experiments/run_ablation.py --dry-run

# Full 27-run ablation
python experiments/run_ablation.py

# Safety scenario suite (SS-1 through SS-4)
python experiments/safety_scenarios.py

# Generate LaTeX tables + figures
python experiments/analyze_results.py
```

Results are saved to `experiments/results/`. The ablation runner is resumable — if it crashes, re-run the same command and it skips completed runs.

---

## LLM backends

| Backend | How to select | Notes |
|---|---|---|
| Ollama (default) | `AE_LLM_BACKEND=ollama` | Local, free, needs `ollama` running |
| Gemini | `AE_LLM_BACKEND=gemini` + `GEMINI_API_KEY=...` | Google free tier: 1500 req/day |

Model can be changed via `AE_LLM_MODEL` env var (default: `qwen2.5-coder:7b-instruct`).

---

## Team

BE Project — GESCOEMSR, Nashik (team of 4).
