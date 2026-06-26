import streamlit as st


def render() -> None:
    st.title("About AdaptEvolve-MAS")

    st.markdown("""
## AdaptEvolve-MAS: Adaptive Meta-Evolutionary Multi-Agent Code Optimization

AdaptEvolve-MAS is a research framework developed at **GESCOEMSR, Nashik** that addresses a fundamental
limitation in LLM-based code generation: static optimization pipelines fail silently when their
evaluation objectives misalign with the true task requirements.

---

### The Pack Metaphor

Like a wolf pack that restructures its hunting strategy based on terrain, AdaptEvolve-MAS uses four
role-specialized AI agents that collectively adapt both *what they search for* and *how they search*:

| Agent | Role | Safety Property |
|---|---|---|
| 🧠 **Strategist** | Agentic RAG research + prompt planning | Grounds decisions in verified knowledge |
| ⚙️ **Solver** | Evolutionary code generation (mutation & crossover) | Population diversity prevents local optima |
| ⚖️ **Judge** | Multi-criteria evaluation (correctness, speed, quality) | Transparent, auditable scoring |
| 🛠️ **Mechanic** | Meta-learner — adapts criteria weights and operators | **Online Objective Correction (OOC)** |

---

### AI Safety Properties

AdaptEvolve-MAS is designed with four verifiable safety properties:

1. **Bounded Autonomy (BA)** — The system terminates after at most `max_cycles` steps,
   preventing runaway autonomous operation.

2. **Interpretable Decision Trails (IDT)** — Every Mechanic decision is logged with
   natural-language rationale, providing a full audit trail.

3. **Online Objective Correction (OOC)** — The Mechanic detects proxy-metric stagnation as
   an implicit signal of objective misalignment and reweights evaluation criteria *within a
   single run*, without human intervention.

4. **Corrigible Restructuring (CR)** — The system adapts its configuration (operators, weights,
   prompts) but cannot modify its own architecture, providing principled second-order
   adaptation within a fixed computational boundary.

---

### Tech Stack

- **LangGraph** — stateful cyclic agent workflow
- **Gemini API** — LLM backbone for all four agents
- **SentenceTransformers** — document embeddings for RAG
- **DuckDuckGo** — web search for research grounding
- **Streamlit** — this web interface

---

### Team

**Pranamya Deshpande · Janhavi Joshi · Parnika Nirgude**
Guide: **Prof. Trupti Atre**
Department of Computer Engineering, GESCOEMSR, Nashik
""")
