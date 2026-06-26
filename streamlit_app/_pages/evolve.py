import time
import sys
import os
import streamlit as st
from datetime import datetime, timezone

# Allow imports from parent directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.dirname(_HERE)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from database import insert_session, update_session, new_session_id
from evolution_runner import EvolutionRunner
from components.progress_panel import render_step_card
from components.score_chart import render_trajectory_chart, render_dimension_chart
from components.code_viewer import render_code


def _parse_uploaded_file(uploaded_file) -> tuple[str, str]:
    """Extract text from a Streamlit UploadedFile (pdf, txt, md)."""
    name = uploaded_file.name
    raw = uploaded_file.read()
    if name.lower().endswith(".pdf"):
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=raw, filetype="pdf")
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
        except ImportError:
            text = raw.decode("utf-8", errors="replace")
    else:
        text = raw.decode("utf-8", errors="replace")
    return text, name


def render(username: str) -> None:
    st.title("🚀 Quick Run")
    st.markdown("Configure and launch the AdaptEvolve-MAS optimization pipeline.")
    st.divider()

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("evolution_form"):
        goal = st.text_area(
            "Optimization Goal",
            placeholder="e.g. Implement a memory-efficient sort algorithm for lists with many duplicates.",
            height=120,
        )
        col1, col2, col3 = st.columns(3)
        max_cycles = col1.slider("Max Cycles", 1, 10, 3)
        pop_size = col2.slider("Population Size", 2, 10, 3)
        num_gens = col3.slider("Generations / Cycle", 1, 5, 2)

        uploaded_files = st.file_uploader(
            "Upload reference documents (optional)",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("🚀 Start Evolution", type="primary")

    if submitted and goal.strip():
        # Parse documents
        documents = [_parse_uploaded_file(f) for f in (uploaded_files or [])]

        # Create a new session
        sid = new_session_id()
        insert_session(sid, username, goal.strip(), max_cycles, pop_size, num_gens)

        # Create and start the runner
        runner = EvolutionRunner()
        runner.start(goal.strip(), max_cycles, pop_size, num_gens, documents)

        # Store in session state
        st.session_state["runner"] = runner
        st.session_state["session_id"] = sid
        st.session_state["evolving"] = True
        st.session_state["step_updates"] = []
        st.session_state["final_result"] = None
        st.rerun()

    elif submitted:
        st.warning("Please enter an optimization goal.")

    # ── Live progress display ─────────────────────────────────────────────────
    if st.session_state.get("evolving") or st.session_state.get("final_result"):
        st.divider()
        runner: EvolutionRunner = st.session_state.get("runner")
        sid: str = st.session_state.get("session_id", "")

        # Drain new updates
        if runner and runner.is_running:
            new_updates = runner.drain()
        elif runner:
            new_updates = runner.drain()
        else:
            new_updates = []

        for upd in new_updates:
            if upd["type"] == "step":
                st.session_state["step_updates"].append(upd)
            elif upd["type"] == "done":
                st.session_state["evolving"] = False
                st.session_state["final_result"] = upd["final_state"]
                fs = upd["final_state"]
                update_session(
                    sid,
                    status="completed",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    final_score=fs.get("holistic_score"),
                    n_cycles_run=fs.get("current_cycle"),
                    best_code=fs.get("best_code"),
                    score_trajectory=fs.get("score_trajectory"),
                    dimension_scores=fs.get("dimension_scores"),
                    mechanic_log=fs.get("mechanic_log"),
                )
            elif upd["type"] == "error":
                st.session_state["evolving"] = False
                st.error("Evolution error:\n```\n" + upd.get("error", "") + "\n```")
                update_session(sid, status="error", finished_at=datetime.now(timezone.utc).isoformat())

        # Status indicator
        if st.session_state.get("evolving"):
            n = len(st.session_state["step_updates"])
            st.info(f"⏳ Evolution running… {n} step(s) complete. Page refreshes automatically.")

        # Render accumulated step cards
        step_updates = st.session_state.get("step_updates", [])
        if step_updates:
            st.markdown(f"#### Progress — {len(step_updates)} step(s)")
            for upd in reversed(step_updates[-20:]):  # show last 20, newest first
                render_step_card(upd)

        # Final results
        final = st.session_state.get("final_result")
        if final:
            st.success("✅ Evolution complete!")
            st.markdown(final.get("report", ""))
            st.divider()

            score_traj = final.get("score_trajectory", [])
            dim_scores = final.get("dimension_scores", {})
            col_l, col_r = st.columns(2)
            with col_l:
                render_trajectory_chart(score_traj)
            with col_r:
                render_dimension_chart(dim_scores)

            render_code(
                final.get("best_code", ""),
                label="Best Solution",
                score=final.get("holistic_score"),
            )

        # Auto-rerun while evolution is still going
        if st.session_state.get("evolving"):
            time.sleep(1.5)
            st.rerun()
