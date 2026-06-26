"""
Standard ablation runner — 3 tasks × 3 conditions × 3 seeds = 27 runs.

Conditions
----------
full     : normal AdaptEvolve-MAS (Strategist + Solver + Judge + Mechanic)
no_meta  : Mechanic skips criteria/operator updates (weights frozen)
no_rag   : Strategist skips web + vector search (no AgenticRAG)

Usage
-----
python experiments/run_ablation.py                      # all 27 runs
python experiments/run_ablation.py --task bubble_sort   # one task
python experiments/run_ablation.py --condition no_meta  # one condition
python experiments/run_ablation.py --dry-run            # validate imports
"""
import os, sys, json, time, argparse, copy, traceback
from datetime import datetime, timezone
from pathlib import Path

# ── Ensure project root is on path ───────────────────────────────────────────
_HERE = Path(__file__).parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_HERE))

RESULTS_DIR = _HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)
LOG_FILE = RESULTS_DIR / "ablation_runs.jsonl"

from tasks import TASKS, check_correctness, time_solution

# ── Ablation matrix ───────────────────────────────────────────────────────────
TASK_KEYS   = ["bubble_sort", "prime_sieve", "matrix_multiply"]
CONDITIONS  = ["full", "no_meta", "no_rag"]
SEEDS       = [0, 1, 2]

# Evolution hyperparameters (trimmed for speed; override with env vars)
MAX_CYCLES  = int(os.environ.get("AE_MAX_CYCLES", "3"))
POP_SIZE    = int(os.environ.get("AE_POP",        "3"))
NUM_GENS    = int(os.environ.get("AE_GENS",       "2"))


# ── Load completed runs from log (for resumability) ──────────────────────────
def _load_completed() -> set:
    done = set()
    if LOG_FILE.exists():
        for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done.add((r["task"], r["condition"], r["seed"]))
            except Exception:
                pass
    return done


# ── Patch helpers (monkey-patch without modifying adaptevolve_core) ───────────
def _patch_no_meta(ae):
    """NoMeta: Mechanic runs but does NOT update criteria or operators."""
    original_mechanic = ae.mechanic_node

    def noop_mechanic(state):
        result = original_mechanic(state)
        # Restore unchanged criteria and clear proposed changes
        frozen = dict(state.get("evaluation_criteria", {}))
        result["evaluation_criteria"] = frozen
        result["proposed_criteria"]   = {}
        result["proposed_operators"]  = []
        result["mechanic_analysis"]   = "[NoMeta] weight updates disabled"
        return result

    ae.mechanic_node = noop_mechanic
    ae.evolution_graph = ae.build_evolution_graph()


def _patch_no_rag(ae):
    """NoRAG: Strategist skips web and vector-store research."""
    original_strategist = ae.strategist_node

    def noop_rag_strategist(state):
        # Temporarily disable the web_search tool on the rag_system
        original_search = ae.rag_system.web_search.search
        ae.rag_system.web_search.search = lambda *a, **kw: []

        original_vs_search = ae.rag_system.vector_store.search
        ae.rag_system.vector_store.search = lambda *a, **kw: []

        try:
            result = original_strategist(state)
        finally:
            ae.rag_system.web_search.search     = original_search
            ae.rag_system.vector_store.search   = original_vs_search

        return result

    ae.strategist_node = noop_rag_strategist
    ae.evolution_graph = ae.build_evolution_graph()


def _reset_patches(ae, original_mechanic, original_strategist):
    """Restore original functions and rebuild graph between runs."""
    ae.mechanic_node   = original_mechanic
    ae.strategist_node = original_strategist
    ae.evolution_graph = ae.build_evolution_graph()


# ── Single run ────────────────────────────────────────────────────────────────
def run_one(task_key: str, condition: str, seed: int, ae) -> dict:
    import random, hashlib
    task = TASKS[task_key]

    # Deterministic seed via hash (LLM itself is non-deterministic, but at least
    # random.seed controls any sampling we add later)
    rng_seed = int(hashlib.md5(f"{task_key}{condition}{seed}".encode()).hexdigest(), 16) % (2**32)
    random.seed(rng_seed)

    state = ae.create_initial_state(task["goal"], max_cycles=MAX_CYCLES)

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()

    try:
        final_state = None
        for chunk in ae.evolution_graph.stream(state, {"recursion_limit": 120}):
            for node_name, node_output in chunk.items():
                if isinstance(node_output, dict):
                    state = {**state, **node_output}
            final_state = state
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "task": task_key,
            "condition": condition,
            "seed": seed,
            "status": "error",
            "error": traceback.format_exc(limit=5),
            "elapsed_s": round(elapsed, 2),
            "started_at": started,
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }

    elapsed = time.perf_counter() - t0

    best_code = (final_state or {}).get("best_solution", {}).get("code", "")
    holistic_score = (final_state or {}).get("holistic_score", 0.0) or 0.0
    n_cycles = (final_state or {}).get("current_cycle", 0)
    score_trajectory = [
        ch.get("holistic_score", 0)
        for ch in (final_state or {}).get("cycle_history", [])
    ]
    dimension_scores = (final_state or {}).get("evaluation_results", {}).get("score", {}).get("dimension_scores", {})

    # Ground-truth correctness (independent of holistic score)
    gt_correctness = check_correctness(best_code, task_key) if best_code else 0.0
    avg_time_ms    = time_solution(best_code, task_key) if best_code else -1.0

    # IDT: fraction of cycles with non-empty mechanic_analysis
    mechanic_analyses = [
        ch for ch in (final_state or {}).get("cycle_history", [])
        if isinstance(ch, dict) and ch.get("mechanic_analysis", "").strip()
    ]
    idt_completeness = len(mechanic_analyses) / max(n_cycles, 1)

    # Final evaluation criteria (shows whether Mechanic actually updated them)
    final_criteria = (final_state or {}).get("evaluation_criteria", {})

    return {
        "task":              task_key,
        "condition":         condition,
        "seed":              seed,
        "status":            "completed",
        "holistic_score":    round(holistic_score, 4),
        "gt_correctness":    round(gt_correctness, 4),
        "n_cycles":          n_cycles,
        "max_cycles":        MAX_CYCLES,
        "score_trajectory":  score_trajectory,
        "dimension_scores":  dimension_scores,
        "final_criteria":    final_criteria,
        "idt_completeness":  round(idt_completeness, 4),
        "avg_time_ms":       round(avg_time_ms, 2),
        "elapsed_s":         round(elapsed, 2),
        "started_at":        started,
        "finished_at":       datetime.now(timezone.utc).isoformat(),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",      choices=TASK_KEYS,  default=None)
    parser.add_argument("--condition", choices=CONDITIONS, default=None)
    parser.add_argument("--seed",      type=int,           default=None)
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    print("Loading adaptevolve_core …")
    import adaptevolve_core as ae
    original_mechanic   = ae.mechanic_node
    original_strategist = ae.strategist_node
    print(f"Pipeline loaded. Backend: {ae.LLM_BACKEND} / {getattr(ae, 'OLLAMA_MODEL_NAME', ae.MODEL_NAME)}")

    if args.dry_run:
        print("[DRY RUN] Imports OK. Exiting.")
        return

    task_keys  = [args.task]      if args.task      else TASK_KEYS
    conditions = [args.condition] if args.condition  else CONDITIONS
    seeds      = [args.seed]      if args.seed is not None else SEEDS

    completed = _load_completed()
    total     = len(task_keys) * len(conditions) * len(seeds)
    run_idx   = 0

    for task_key in task_keys:
        for condition in conditions:
            for seed in seeds:
                run_idx += 1
                key = (task_key, condition, seed)
                if key in completed:
                    print(f"[{run_idx}/{total}] SKIP  {task_key}/{condition}/seed{seed} (already done)")
                    continue

                print(f"\n[{run_idx}/{total}] RUN   {task_key}/{condition}/seed{seed}")

                # Apply condition patches
                _reset_patches(ae, original_mechanic, original_strategist)
                if condition == "no_meta":
                    _patch_no_meta(ae)
                elif condition == "no_rag":
                    _patch_no_rag(ae)

                result = run_one(task_key, condition, seed, ae)

                # Always restore after each run
                _reset_patches(ae, original_mechanic, original_strategist)

                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")

                status = result["status"]
                if status == "completed":
                    print(f"       score={result['holistic_score']:.3f}  "
                          f"gt={result['gt_correctness']:.2f}  "
                          f"cycles={result['n_cycles']}  "
                          f"idt={result['idt_completeness']:.2f}  "
                          f"time={result['elapsed_s']:.0f}s")
                else:
                    print(f"       ERROR: {result.get('error','?')[:120]}")

    print(f"\nDone. Results in {LOG_FILE}")


if __name__ == "__main__":
    main()
