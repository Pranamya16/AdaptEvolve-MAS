# AdaptiWolf AI Safety Paper — Plan

## Paper Title

**"AdaptiWolf: Online Objective Correction in Meta-Evolutionary Multi-Agent Code Optimization"**

Short title for arXiv: `adaptiwolf-ooc-mas`

---

## The Core Contribution (novel, publishable)

The Mechanic agent in AdaptEvolve-MAS performs **Online Objective Correction (OOC)**:
it detects proxy-metric stagnation as an implicit signal of objective misalignment
and proposes reweighted evaluation criteria within a single inference run — without
external human feedback, without fine-tuning, and without halting the optimization loop.

This is distinct from every prior method:
- **Constitutional AI** / **RLHF**: objective modification at training time, not during inference
- **FunSearch / OpenEvolve**: fixed fitness function throughout a run
- **AutoGen / MetaGPT**: no second-order loop that restructures what the agents optimize *for*

---

## Four Safety Properties (the "AdaptiWolf" Framework)

| Property | Abbreviation | Where it lives in code |
|---|---|---|
| Online Objective Correction | **OOC** | `mechanic_node()` — weight rebalancing logic |
| Bounded Autonomy | **BA** | `max_cycles` ceiling + convergence-based early stop in `build_evolution_graph()` |
| Interpretable Decision Trails | **IDT** | `mechanic_analysis` field logged each cycle with natural-language rationale |
| Corrigible Restructuring | **CR** | Mechanic modifies config (weights, operator pool) not code architecture |

---

## Paper Outline

### Section 1 — Introduction (≈0.5 page)

- Three failure modes of closed-loop LLM-driven optimization:
  1. **Objective lock-in**: static fitness function can be gamed
  2. **Open-loop search**: no feedback mechanism when proxy diverges from goal
  3. **Opaque adaptation**: system changes behaviour without audit trail
- AdaptiWolf addresses all three via OOC + BA + IDT + CR
- 4 claimed contributions (box at top of section):
  1. Formalization of OOC as an inference-time safety property
  2. The AdaptiWolf Safety Property Set (BA/IDT/OOC/CR)
  3. Safety Scenario Suite (4 adversarial test cases)
  4. Ablation evidence that OOC improves convergence on 3 benchmark tasks

### Section 2 — Background & Related Work (≈0.7 page)

- **Specification gaming & reward hacking** (Krakovna et al. 2020, Amodei et al. 2016)
- **Constitutional AI** (Bai et al. 2022) — training-time, not inference-time
- **RLHF** (Ouyang et al. 2022) — requires labeled preference dataset
- **Corrigibility** (Soares et al. 2015) — passive; AdaptiWolf implements *active* corrigibility
- **FunSearch / OpenEvolve** — evolutionary code optimization, no OOC
- **Multi-agent frameworks** (AutoGen, MetaGPT, LangGraph) — no second-order loop
- **LLM self-evaluation** (Madaan et al. 2023, Shinn et al. 2023) — within-solution, not cross-cycle objective correction

### Section 3 — AdaptEvolve-MAS Framework (≈1.2 pages)

Reframes the existing project content as a formal system.

**3.1 System Architecture**
- Formal model: state $\mathcal{S}_c = (\text{code}_c, \mathbf{w}_c, H_c)$
- Graph $\mathcal{G}$: Strategist → Solver → Judge → Mechanic → (Strategist | END)
- Algorithm 1: the cyclic LangGraph execution loop

**3.2 The Four Agents**
- Brief role of each (reference Agents tab content; keep ≤ 2 sentences each)

**3.3 Safety-Relevant Design Decisions**
- `max_cycles` as a hard autonomy tripwire (BA)
- `mechanic_analysis` field as tamper-visible audit log (IDT)
- Convergence detection (Δscore < 0.01 for 2 cycles) as secondary BA trigger
- Mechanic proposes operators by name + rationale; names are deduped (CR)

### Section 4 — AI Safety Analysis (≈1.5 pages — central novel section)

**4.1 Bounded Autonomy (BA)**
- Definition: $n_{\text{cycles}} \leq C_{\max}$ always holds; secondary trigger via convergence
- Evidence: code reference (`conditional_edge` in `build_evolution_graph`)
- Limitation: Mechanic could delay convergence signal (mitigated by absolute cap)

**4.2 Interpretable Decision Trails (IDT)**
- Definition: $\forall c \in [0, n_{\text{cycles}}]$, $\text{rationale}_c \neq \emptyset$ (Mechanic provides NL justification)
- Metric: IDT completeness = fraction of cycles with non-empty `mechanic_analysis`
- Limitation: rationale quality depends on LLM faithfulness (evaluated in Section 5)

**4.3 Online Objective Correction (OOC)**
- Formal statement: $\mathbf{w}_{c+1} = f(\mathbf{w}_c, H_{0:c}, \text{feedback}_c)$
- where $f$ is Mechanic's LLM-backed weight update step
- OOC-score metric: $(S_{\text{Full}} - S_{\text{NoMeta}}) / S_{\text{Full}}$ at convergence
- Failure mode: Mechanic proposes bad reweightings → mitigation: weight floors + human override hook

**4.4 Corrigible Restructuring (CR)**
- Definition: adaptations are confined to the configuration layer (weights, operator pool name list) — not the agent graph structure or agent code
- Evidence: only `state["evaluation_criteria"]` and `state["operator_type"]` are modified
- Analogy: pack alpha reassigns roles + strategy, but does not add/remove pack members

### Section 5 — Experiments (≈1.5 pages)

**5.1 Standard Ablation (existing plan)**

3 conditions × 3 tasks × 3 seeds = 27 runs

| Condition | What's disabled |
|---|---|
| Full | — |
| NoMeta | Mechanic returns empty analysis; weights frozen |
| NoRAG | Strategist skips web/vector search |

Tasks: BubbleSort optimization, Prime Sieve, Matrix Multiply

**5.2 Safety Scenario Suite (4 adversarial scenarios — new)**

| ID | Scenario | What it tests |
|---|---|---|
| SS-1 | **Reward Hacking Probe** | Task where fast-but-wrong code scores high on proxy; OOC detects correctness divergence |
| SS-2 | **Objective Shift Mid-Run** | Inject perturbed weights at cycle 3; measure whether Mechanic corrects back toward task-valid weights |
| SS-3 | **Convergence Safety** | Force uniformly bad solutions; verify termination at `max_cycles`, no false positive early stop |
| SS-4 | **Operator Explosion** | 10-cycle run; verify dedup keeps operator pool bounded (CR check) |

**5.3 Key Results (hypothesized; fill from actual runs)**

- Full outperforms NoMeta on holistic score (OOC-score > 0 for all tasks)
- SS-1: NoMeta holistic score ↑ while GT correctness ↓; Full corrects
- SS-3: All runs terminate at or before max_cycles (BA always holds)
- IDT completeness ≥ 90% across all Full runs

### Section 6 — Discussion (≈0.5 page)

- AdaptiWolf vs Constitutional AI: inference-time vs training-time
- AdaptiWolf vs RLHF: no labeled dataset required
- AdaptiWolf vs debate-based oversight: modifies the optimization objective, not just flags outputs
- Limitations: OOC is heuristic (LLM-driven, not formally verified), IDT quality is LLM-dependent, BA provides only a finite-step (not asymptotic) guarantee

### Section 7 — Future Work (≈0.2 page)

- Formal convergence proof for weight-update sequence
- Cryptographically signed decision logs (tamper-evident IDT)
- Human approval gate triggered above a divergence threshold (strong corrigibility)
- Extending OOC to multi-objective Pareto fronts

---

## Key Figures

1. **Architecture diagram**: 4-agent cyclic graph, second-order Mechanic loop highlighted in red/orange
2. **Score trajectory**: holistic_score vs cycle for Full vs NoMeta vs NoRAG (line chart)
3. **Criteria drift**: stacked area chart showing $\mathbf{w}_c$ per dimension across cycles
4. **SS-1 result**: side-by-side bar comparing holistic_score vs GT correctness for Full vs NoMeta

---

## Files to Create / Edit

| File | Action |
|---|---|
| `paper/main.tex` | Create from scratch (AdaptiWolf paper, ~6 pages, ACL/AAAI format) |
| `paper/references.bib` | Safety + evolutionary code optimization bibliography |
| `experiments/run_ablation.py` | Standard 27-run ablation runner |
| `experiments/safety_scenarios.py` | SS-1 through SS-4 scenario runners |
| `experiments/analyze_results.py` | OOC-score, IDT completeness, criteria drift chart generation |

---

## Venue Targets

| Venue | Deadline | Page limit | Notes |
|---|---|---|---|
| arXiv (cs.AI) | Any time | Unlimited | Submit first to establish timestamp |
| SafeAI @ AAAI 2026 | ~Nov 2025 | 4–6 pages | Primary target; safety angle fits directly |
| AIES 2026 (student track) | ~Jan 2026 | 8 pages | Good backup |
| IJCAI 2026 MAS Workshop | ~Mar 2026 | 4–6 pages | Multi-agent angle |

---

## Writing Tips for Research Paper

- Lead abstract with the problem (proxy gaming in LLM optimization), then the mechanism (OOC), then evidence (ablation + scenario suite)
- Every claim about the code must cite a line/function by name (Algorithm 1 should match `build_evolution_graph()`)
- Differentiation from prior work is the hardest part — Section 2 needs to be precise about what OOC adds beyond "the LLM reflects on its outputs"
- The Safety Scenario Suite is the clearest empirical novelty; give it the most experimental detail
