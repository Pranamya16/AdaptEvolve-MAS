import json
import sys
import os
import streamlit as st
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_HERE)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from database import get_user_sessions, get_session_detail
from components.score_chart import render_trajectory_chart, render_dimension_chart
from components.code_viewer import render_code


def render(username: str) -> None:
    st.title("📋 Session History")

    sessions = get_user_sessions(username, limit=100)
    if not sessions:
        st.info("No sessions yet. Run your first evolution!")
        return

    # Build a display dataframe
    df = pd.DataFrame(sessions)[
        ["session_id", "goal", "started_at", "status", "final_score", "n_cycles_run", "max_cycles"]
    ]
    df["goal_short"] = df["goal"].str[:60]
    df["final_score"] = df["final_score"].map(lambda x: f"{x:.4f}" if x is not None else "—")
    df["started_at"] = df["started_at"].str[:16].str.replace("T", " ")

    display_df = df[["goal_short", "started_at", "status", "final_score", "n_cycles_run"]].rename(
        columns={
            "goal_short": "Goal",
            "started_at": "Started",
            "status": "Status",
            "final_score": "Score",
            "n_cycles_run": "Cycles",
        }
    )

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### View Session Details")
    session_labels = {
        s["session_id"]: f"{s['started_at'][:16].replace('T',' ')} — {s['goal'][:50]}"
        for s in sessions
    }
    selected_id = st.selectbox(
        "Select a session",
        options=list(session_labels.keys()),
        format_func=lambda sid: session_labels[sid],
    )

    if selected_id:
        detail = get_session_detail(selected_id)
        if detail:
            st.markdown(f"**Goal:** {detail['goal']}")
            st.markdown(
                f"**Status:** {detail['status']} · **Score:** "
                f"{detail['final_score']:.4f if detail['final_score'] else '—'} · "
                f"**Cycles:** {detail['n_cycles_run'] or '?'}"
            )

            score_traj = detail.get("score_trajectory") or []
            dim_scores = detail.get("dimension_scores") or {}
            if isinstance(score_traj, str):
                try:
                    score_traj = json.loads(score_traj)
                except Exception:
                    score_traj = []
            if isinstance(dim_scores, str):
                try:
                    dim_scores = json.loads(dim_scores)
                except Exception:
                    dim_scores = {}

            col_l, col_r = st.columns(2)
            with col_l:
                render_trajectory_chart(score_traj, title="Score Trajectory")
            with col_r:
                render_dimension_chart(dim_scores)

            render_code(detail.get("best_code", ""), label="Best Solution", score=detail.get("final_score"))
