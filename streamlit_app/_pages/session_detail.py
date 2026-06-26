"""Session detail — shows a past evolution run from history."""
import os, sys, json
import streamlit as st

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
for p in (_APP_DIR, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from database import get_session_detail


def render(session_id: str) -> None:
    if st.button("← Back to chat"):
        st.session_state["view"] = "chat"
        st.rerun()

    if not session_id:
        st.warning("No session selected.")
        return

    s = get_session_detail(session_id)
    if not s:
        st.error("Session not found.")
        return

    st.title("📋 Evolution Session")
    st.caption(f"ID: `{session_id}`")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Status", s.get("status", "—"))
    col2.metric("Final Score", f"{s['final_score']:.3f}" if s.get("final_score") else "—")
    col3.metric("Cycles Run", s.get("n_cycles_run", "—"))
    col4.metric("Max Cycles", s.get("max_cycles", "—"))

    st.markdown(f"**Goal:** {s.get('goal', '—')}")
    st.caption(f"Started: {s.get('started_at','')} · Finished: {s.get('finished_at','—')}")
    st.divider()

    if s.get("best_code"):
        st.markdown("#### Best Solution")
        score = s.get("final_score") or 0
        st.caption(f"Score: {score:.3f}")
        st.code(s["best_code"], language="python")
        st.download_button(
            "⬇️ Download solution.py",
            data=s["best_code"],
            file_name="solution.py",
            mime="text/plain",
        )

    traj = s.get("score_trajectory")
    if traj and isinstance(traj, list) and len(traj) > 1:
        st.markdown("#### Score Trajectory")
        try:
            import plotly.graph_objects as go
            fig = go.Figure(go.Scatter(
                x=list(range(1, len(traj)+1)), y=traj,
                mode="lines+markers", line=dict(color="#4CAF50", width=2),
            ))
            fig.update_layout(
                xaxis_title="Cycle", yaxis_title="Score",
                margin=dict(l=0, r=0, t=10, b=0), height=250,
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.write(traj)

    if s.get("mechanic_log"):
        with st.expander("🛠️ Mechanic Log"):
            st.json(s["mechanic_log"])
