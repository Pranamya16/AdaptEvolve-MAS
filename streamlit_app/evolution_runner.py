"""
evolution_runner.py -- Wraps adaptevolve_core's evolution pipeline in a
daemon thread + queue so the Streamlit UI can poll for updates
without blocking the main thread.
"""
import sys
import os
import queue
import threading
import traceback
from datetime import datetime, timezone
from typing import Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class EvolutionRunner:
    """
    Runs `evolution_graph.stream(state)` in a daemon thread.
    The UI calls `drain()` on each Streamlit rerun to collect updates.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self.is_running: bool = False

    def start(
        self,
        goal: str,
        max_cycles: int,
        population_size: int,
        num_generations: int,
        documents: list,           # [(text_str, source_name), ...]
    ) -> None:
        self.is_running = True
        self._thread = threading.Thread(
            target=self._run,
            args=(goal, max_cycles, population_size, num_generations, documents),
            daemon=True,
        )
        self._thread.start()

    def _run(self, goal, max_cycles, population_size, num_generations, documents):
        try:
            import adaptevolve_core as ae

            # Ingest uploaded documents into the shared RAG system
            for text, name in documents:
                ae.rag_system.ingest_document(text, source=name)

            state = ae.create_initial_state(goal, max_cycles=max_cycles)
            # Patch population/generation knobs into state
            state["population_size"] = population_size  # type: ignore[assignment]
            state["num_generations"] = num_generations  # type: ignore[assignment]

            for step in ae.evolution_graph.stream(state):
                for node, delta in step.items():
                    state.update(delta)
                    self._queue.put({
                        "type": "step",
                        "node": node,
                        "cycle": state.get("current_cycle", 0),
                        "score": float(state.get("holistic_score", 0.0) or 0.0),
                        "active_agent": state.get("active_agent", ""),
                        "best_code": state.get("best_solution", {}).get("code", ""),
                        "messages": list(delta.get("messages", [])),
                        "mechanic_analysis": state.get("mechanic_analysis", ""),
                        "proposed_operators": list(state.get("proposed_operators", [])),
                        "cycle_history": list(state.get("cycle_history", [])),
                        "evaluation_results": dict(state.get("evaluation_results", {})),
                        "ts": datetime.now(timezone.utc).isoformat(),
                    })

            # Build final summary using session's report method
            session = ae.AdaptEvolveSession()
            session.current_state = state
            report = session._generate_final_report(state)

            self._queue.put({
                "type": "done",
                "report": report,
                "holistic_score": float(state.get("holistic_score", 0.0) or 0.0),
                "current_cycle": state.get("current_cycle", 0),
                "best_code": state.get("best_solution", {}).get("code", ""),
                "score_trajectory": [
                    float(h.get("holistic_score", 0) or 0)
                    for h in state.get("cycle_history", [])
                ],
                "dimension_scores": state.get("evaluation_results", {}).get(
                    "score", {}
                ).get("dimension_scores", {}),
                "mechanic_log": [
                    h.get("mechanic_analysis", "")
                    for h in state.get("cycle_history", [])
                    if h.get("mechanic_analysis")
                ],
            })
        except Exception:
            self._queue.put({
                "type": "error",
                "error": traceback.format_exc(),
                "ts": datetime.now(timezone.utc).isoformat(),
            })
        finally:
            self.is_running = False

    def drain(self) -> list:
        """Pull all pending updates from the queue without blocking."""
        updates = []
        while True:
            try:
                updates.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return updates
