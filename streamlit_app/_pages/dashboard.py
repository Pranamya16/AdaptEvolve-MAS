import streamlit as st
from database import get_user_sessions


def render(username: str, display_name: str) -> None:
    st.title(f"Welcome back, {display_name} 👋")
    st.markdown("### AdaptEvolve — Adaptive Meta-Evolutionary Code Optimization")
    st.divider()

    sessions = get_user_sessions(username, limit=100)
    completed = [s for s in sessions if s["status"] == "completed"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sessions", len(sessions))
    col2.metric("Completed", len(completed))
    best_score = max((s["final_score"] or 0) for s in completed) if completed else 0.0
    col3.metric("Best Score", f"{best_score:.4f}")

    st.divider()
    st.markdown("### Quick Start")

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🚀 Start New Evolution", use_container_width=True, type="primary"):
            st.session_state["current_page"] = "evolve"
            st.rerun()
    with col_b:
        if st.button("📋 View History", use_container_width=True):
            st.session_state["current_page"] = "history"
            st.rerun()

    if completed:
        st.divider()
        st.markdown("### Recent Sessions")
        for s in sessions[:5]:
            score_str = f"{s['final_score']:.4f}" if s["final_score"] is not None else "—"
            badge = "✅" if s["status"] == "completed" else "⏳" if s["status"] == "running" else "❌"
            st.markdown(
                f"{badge} **{s['goal'][:60]}{'…' if len(s['goal'])>60 else ''}** "
                f"· Score: `{score_str}` · Cycles: `{s['n_cycles_run'] or '?'}` "
                f"· {s['started_at'][:10]}"
            )
