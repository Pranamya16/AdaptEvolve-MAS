"""
Monitor page — live status, current best solution, and cycle history.
Mirrors the Gradio "Monitor" tab.
"""
import sys, os
import streamlit as st

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
for p in (_APP_DIR, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


def render() -> None:
    st.title("📊 Monitor")
    st.caption("Real-time status of the current evolution session.")

    import adaptevolve_core as ae
    from components.score_chart import render_trajectory_chart
    from components.code_viewer import render_code

    if st.button("🔄 Refresh", type="primary"):
        pass  # Streamlit reruns on any interaction

    session = ae.session
    state = session.current_state

    if state is None:
        st.info("No active session. Start an evolution from the **Quick Run** tab.")
        return

    # ── Status metrics ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Cycle", f"{state.get('current_cycle', 0)} / {state.get('max_cycles', 5)}")
    col2.metric("Score", f"{state.get('holistic_score', 0.0):.4f}")
    col3.metric("Status", state.get("status", "—"))
    col4.metric("Active Agent", state.get("active_agent", "—"))

    st.caption(f"Population size: {len(state.get('population', []))}  |  "
               f"Continue: {state.get('should_continue', False)}")

    # ── Score trajectory ──────────────────────────────────────────────────────
    history = state.get("cycle_history", [])
    scores = [float(h.get("holistic_score", 0) or 0) for h in history]
    if scores:
        st.divider()
        render_trajectory_chart(scores)

    # ── Current best solution ─────────────────────────────────────────────────
    st.divider()
    st.markdown("#### Current Best Solution")
    best = state.get("best_solution") or {}
    render_code(best.get("code", ""), label="Best Solution", score=best.get("score"))

    # ── Cycle history table ───────────────────────────────────────────────────
    if history:
        st.divider()
        st.markdown("#### Cycle History")
        for entry in history:
            cycle_n = entry.get("cycle", "?")
            score = float(entry.get("holistic_score", 0) or 0)
            pop = entry.get("population_size", 0)
            ts = entry.get("timestamp", "")[:16]
            with st.expander(f"Cycle {cycle_n} | Score: {score:.4f} | Pop: {pop} | {ts}"):
                summary = entry.get("evaluation_summary", {})
                if summary:
                    for dim, val in summary.items():
                        st.write(f"- **{dim}:** {val:.3f}")
                mech = entry.get("mechanic_analysis", "")
                if mech:
                    st.markdown("**Mechanic analysis:**")
                    st.info(mech[:600])
