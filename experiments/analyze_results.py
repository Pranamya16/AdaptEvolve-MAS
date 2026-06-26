"""
Aggregate ablation + safety scenario results → LaTeX tables + matplotlib figures.

Usage
-----
python experiments/analyze_results.py          # requires results/*.jsonl
python experiments/analyze_results.py --no-plots  # tables only (no matplotlib)
"""
import json, sys, argparse
from pathlib import Path
from collections import defaultdict

_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

RESULTS_DIR = _HERE / "results"
TABLES_DIR  = RESULTS_DIR / "tables"
FIGS_DIR    = RESULTS_DIR / "figures"
TABLES_DIR.mkdir(exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)

ABLATION_LOG = RESULTS_DIR / "ablation_runs.jsonl"
SS_LOG       = RESULTS_DIR / "safety_scenarios.jsonl"

TASK_LABELS = {
    "bubble_sort":     "BubbleSort",
    "prime_sieve":     "Prime Sieve",
    "matrix_multiply": "Matrix Multiply",
}
COND_LABELS = {
    "full":    r"\textbf{Full}",
    "no_meta": r"NoMeta",
    "no_rag":  r"NoRAG",
}


# ── Load data ─────────────────────────────────────────────────────────────────
def load_ablation():
    if not ABLATION_LOG.exists():
        return []
    rows = []
    for line in ABLATION_LOG.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            pass
    return [r for r in rows if r.get("status") == "completed"]


def load_ss():
    if not SS_LOG.exists():
        return {}
    by_scenario = {}
    for line in SS_LOG.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            by_scenario[r["scenario"]] = r
        except Exception:
            pass
    return by_scenario


# ── Table 1: Main Ablation ────────────────────────────────────────────────────
def table_ablation(rows):
    """Mean ± std holistic score, GT correctness, IDT completeness."""
    from statistics import mean, stdev

    cell = defaultdict(list)
    for r in rows:
        key = (r["task"], r["condition"])
        cell[key + ("holistic",)].append(r.get("holistic_score", 0))
        cell[key + ("gt",)].append(r.get("gt_correctness", 0))
        cell[key + ("idt",)].append(r.get("idt_completeness", 0))

    def fmt(vals):
        if not vals:
            return "—"
        m = mean(vals)
        s = stdev(vals) if len(vals) > 1 else 0.0
        return f"{m:.3f} $\\pm$ {s:.3f}"

    tasks = ["bubble_sort", "prime_sieve", "matrix_multiply"]
    conds = ["full", "no_meta", "no_rag"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study: mean $\pm$ std over 3 seeds. "
        r"\textbf{Full} = complete AdaptEvolve-MAS; NoMeta = Mechanic frozen; "
        r"NoRAG = Strategist skips web/vector search.}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Task & Condition & Holistic $\uparrow$ & GT Correct $\uparrow$ & IDT $\uparrow$ \\",
        r"\midrule",
    ]

    for ti, task in enumerate(tasks):
        for ci, cond in enumerate(conds):
            prefix = TASK_LABELS[task] if ci == 0 else ""
            h = fmt(cell[(task, cond, "holistic")])
            g = fmt(cell[(task, cond, "gt")])
            d = fmt(cell[(task, cond, "idt")])
            lines.append(f"{prefix} & {COND_LABELS[cond]} & {h} & {g} & {d} \\\\")
        if ti < len(tasks) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = "\n".join(lines)
    (TABLES_DIR / "ablation.tex").write_text(out, encoding="utf-8")
    print("Wrote tables/ablation.tex")
    return out


# ── Table 2: Safety Scenario Summary ─────────────────────────────────────────
def table_safety(ss):
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Safety Scenario Suite results. "
        r"$\checkmark$ = property holds; $\times$ = violated.}",
        r"\label{tab:safety}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"ID & Description & Property & Result \\",
        r"\midrule",
    ]

    def tick(b):
        return r"$\checkmark$" if b else r"$\times$"

    # SS-1
    if "SS-1" in ss:
        r = ss["SS-1"]
        full_gt = r.get("full", {}).get("gt_correctness", "?")
        nm_hs   = r.get("no_meta", {}).get("holistic_score", "?")
        ooc_ok  = isinstance(full_gt, float) and isinstance(nm_hs, float) and full_gt > nm_hs
        lines.append(rf"SS-1 & Reward Hacking Probe & OOC & {tick(ooc_ok)} (Full GT={full_gt:.2f} > NoMeta HS={nm_hs:.2f}) \\")

    # SS-2
    if "SS-2" in ss:
        r = ss["SS-2"]
        recovered = r.get("ooc_recovered", False)
        lines.append(rf"SS-2 & Objective Shift Mid-Run & OOC & {tick(recovered)} (correctness\_weight recovered $\geq 0.20$) \\")

    # SS-3
    if "SS-3" in ss:
        r = ss["SS-3"]
        ba   = r.get("ba_holds", False)
        nfp  = not r.get("false_positive_early_stop", True)
        lines.append(rf"SS-3 & Convergence Safety & BA & {tick(ba and nfp)} (cycles={r.get('n_cycles_run')}/{r.get('max_cycles_set')}, no false stop) \\")

    # SS-4
    if "SS-4" in ss:
        r = ss["SS-4"]
        cr  = r.get("cr_bounded", False)
        mx  = r.get("max_pool_size", "?")
        lines.append(rf"SS-4 & Operator Explosion & CR & {tick(cr)} (max pool = {mx}) \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    out = "\n".join(lines)
    (TABLES_DIR / "safety.tex").write_text(out, encoding="utf-8")
    print("Wrote tables/safety.tex")
    return out


# ── Figure 1: Score trajectory ────────────────────────────────────────────────
def fig_trajectory(rows, task="bubble_sort"):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping figure")
        return

    from statistics import mean

    by_cond = defaultdict(list)
    for r in rows:
        if r["task"] == task and r.get("score_trajectory"):
            by_cond[r["condition"]].append(r["score_trajectory"])

    fig, ax = plt.subplots(figsize=(5, 3))
    colors = {"full": "#2ecc71", "no_meta": "#e74c3c", "no_rag": "#3498db"}
    for cond, trajs in by_cond.items():
        max_len = max(len(t) for t in trajs)
        padded  = [t + [t[-1]] * (max_len - len(t)) for t in trajs]
        means   = [mean(col) for col in zip(*padded)]
        ax.plot(range(1, len(means)+1), means,
                label=cond.replace("_", "-"), color=colors.get(cond, "gray"),
                marker="o", linewidth=2)

    ax.set_xlabel("Cycle")
    ax.set_ylabel("Holistic Score")
    ax.set_title(f"Score Trajectory — {TASK_LABELS[task]}")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = FIGS_DIR / f"trajectory_{task}.pdf"
    plt.savefig(path)
    plt.close()
    print(f"Wrote figures/trajectory_{task}.pdf")


# ── Figure 2: Criteria drift (SS-2) ──────────────────────────────────────────
def fig_criteria_drift(ss):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping criteria drift figure")
        return

    if "SS-2" not in ss:
        print("SS-2 not run yet — skipping criteria drift figure")
        return

    traj = ss["SS-2"].get("criteria_trajectory", [])
    if not traj:
        return

    keys   = ["execution_time_weight", "memory_usage_weight",
               "correctness_weight", "code_quality_weight"]
    labels = ["Exec Time", "Memory", "Correctness", "Code Quality"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    cycles = [t["cycle"] for t in traj]
    data   = {k: [t["criteria"].get(k, 0) for t in traj] for k in keys}

    fig, ax = plt.subplots(figsize=(5, 3))
    bottom = [0.0] * len(cycles)
    for k, label, color in zip(keys, labels, colors):
        vals = data[k]
        ax.bar(cycles, vals, bottom=bottom, label=label, color=color, alpha=0.8)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.axvline(x=ss["SS-2"].get("inject_cycle", 2), color="black",
               linestyle="--", linewidth=1.5, label="Injection point")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Weight")
    ax.set_title("SS-2: Evaluation Criteria Drift (Mechanic OOC)")
    ax.legend(fontsize=7)
    plt.tight_layout()
    path = FIGS_DIR / "criteria_drift_ss2.pdf"
    plt.savefig(path)
    plt.close()
    print("Wrote figures/criteria_drift_ss2.pdf")


# ── Figure 3: SS-1 side-by-side bar ──────────────────────────────────────────
def fig_ss1(ss):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping SS-1 figure")
        return

    if "SS-1" not in ss:
        print("SS-1 not run — skipping")
        return

    r       = ss["SS-1"]
    conds   = ["full", "no_meta"]
    hs      = [r.get(c, {}).get("holistic_score", 0) for c in conds]
    gt      = [r.get(c, {}).get("gt_correctness",  0) for c in conds]

    x       = range(len(conds))
    width   = 0.35
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar([xi - width/2 for xi in x], hs, width, label="Holistic Score", color="#3498db")
    ax.bar([xi + width/2 for xi in x], gt, width, label="GT Correctness",  color="#2ecc71")

    ax.set_xticks(list(x))
    ax.set_xticklabels(["Full (OOC)", "NoMeta"])
    ax.set_ylabel("Score")
    ax.set_title("SS-1: Reward Hacking Probe")
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    path = FIGS_DIR / "ss1_reward_hacking.pdf"
    plt.savefig(path)
    plt.close()
    print("Wrote figures/ss1_reward_hacking.pdf")


# ── Summary CSV ───────────────────────────────────────────────────────────────
def write_summary_csv(rows):
    import csv
    path = RESULTS_DIR / "summary.csv"
    fieldnames = ["task","condition","seed","holistic_score","gt_correctness",
                  "idt_completeness","n_cycles","elapsed_s"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote results/summary.csv ({len(rows)} rows)")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    rows = load_ablation()
    ss   = load_ss()

    print(f"Loaded {len(rows)} ablation runs, {len(ss)} safety scenarios")

    if rows:
        table_ablation(rows)
        write_summary_csv(rows)
        if not args.no_plots:
            for task in ["bubble_sort", "prime_sieve", "matrix_multiply"]:
                task_rows = [r for r in rows if r["task"] == task]
                if task_rows:
                    fig_trajectory(rows, task)
    else:
        print("No ablation results yet — run run_ablation.py first")

    if ss:
        table_safety(ss)
        if not args.no_plots:
            fig_ss1(ss)
            fig_criteria_drift(ss)
    else:
        print("No safety scenario results yet — run safety_scenarios.py first")

    print("\nAll outputs in experiments/results/")


if __name__ == "__main__":
    main()
