"""
Agents Info page — static description of the 4 AdaptEvolve agents
and the Lyra 4-D Methodology.  Mirrors the Gradio "Agents" tab.
"""
import streamlit as st


def render() -> None:
    st.title("🤖 Agents")
    st.caption("The four role-specialized agents that power AdaptEvolve-MAS.")

    st.markdown("""
```
┌─────────────┐
│  STRATEGIST  │ ← Research & Plan (Agentic RAG)
│  (Lyra)      │
└──────┬───────┘
       │
       ▼
┌─────────────┐
│   SOLVER     │ ← Evolutionary Code Generation
│  (OpenEvolve)│   mutation · crossover · selection
└──────┬───────┘
       │
       ▼
┌─────────────┐
│    JUDGE     │ ← Multi-faceted Evaluation
│  (Benchmark) │   time · memory · correctness · quality
└──────┬───────┘
       │
       ▼
┌─────────────┐     ┌─────┐
│  MECHANIC    │────►│ END │  (converged or max_cycles)
│  (Meta-Learn)│     └─────┘
└──────┬───────┘
       │ (if continue)
       ▼
  STRATEGIST (refined prompt, next cycle)
```
""")

    with st.expander("🧠 Strategist — Agentic RAG", expanded=True):
        st.markdown("""
**Role:** Research & strategic planning

The Strategist uses Agentic RAG to autonomously plan a multi-step research workflow:
1. Generates targeted search queries from the optimization goal
2. Searches the internal document vector store
3. Searches the web (DuckDuckGo)
4. Synthesizes findings into a grounded solver prompt

On subsequent cycles it refines the prompt using the Judge's feedback and the Mechanic's analysis.

**Lyra 4-D:** DECONSTRUCT the goal → DIAGNOSE gaps → DEVELOP a research plan → DELIVER a precise solver prompt.
""")

    with st.expander("⚙️ Solver — Evolutionary Engine (OpenEvolve)"):
        st.markdown("""
**Role:** Generate and evolve code solutions

The Solver maintains a population of candidate programs evolved via:
- **Point mutation** — targeted improvements to one code element
- **Crossover** — combines the best aspects of two parent solutions
- **Selection** — (μ+λ) truncation selection keeps top performers

Uses the OpenEvolve library if available, otherwise falls back to a built-in Gemini-powered engine.

**Lyra 4-D:** DECONSTRUCT the prompt → DIAGNOSE population diversity → DEVELOP evolved candidates → DELIVER best solution.
""")

    with st.expander("⚖️ Judge — Multi-Criteria Evaluator"):
        st.markdown("""
**Role:** Rigorous evaluation of candidate solutions

Evaluates each solution on four weighted dimensions:

| Dimension | Default Weight | How Measured |
|---|---|---|
| Execution Time | 0.30 | Sandboxed timing |
| Memory Usage | 0.25 | `sys.getsizeof` proxy |
| Correctness | 0.30 | Test-case pass rate |
| Code Quality | 0.15 | Static analysis (pylint-style) |

Produces a **holistic score** (0–1) and actionable feedback for the next cycle.

**Lyra 4-D:** DECONSTRUCT criteria → DIAGNOSE solution weaknesses → DEVELOP metrics → DELIVER weighted score + feedback.
""")

    with st.expander("🛠️ Mechanic — Meta-Learner"):
        st.markdown("""
**Role:** Improve the optimization process itself (second-order loop)

After each cycle the Mechanic:
1. **Analyses** the score trajectory and convergence patterns
2. **Proposes** a new evolutionary operator (with rationale)
3. **Re-weights** the Judge's evaluation criteria based on observed gaps
4. **Conditions** the Strategist's next prompt refinement

Includes automatic **convergence detection**: if holistic score changes < 0.01 for 2+ consecutive cycles, the Mechanic signals termination.

**Lyra 4-D:** DECONSTRUCT cycle history → DIAGNOSE bottlenecks → DEVELOP new operators & criteria → DELIVER meta-improvements.
""")

    st.divider()
    st.markdown("""
### Lyra 4-D Methodology

All four agents share a unified four-phase reasoning pattern:

| Phase | Description |
|---|---|
| **1. DECONSTRUCT** | Extract core requirements, constraints, objectives |
| **2. DIAGNOSE** | Audit current state for gaps, weaknesses, opportunities |
| **3. DEVELOP** | Apply role-specific techniques to generate outputs |
| **4. DELIVER** | Format and structure outputs for downstream consumption |

This provides interpretability and consistency across all agent interactions.
""")
