"""
Chat page — primary interface, ChatGPT-style.

Regular messages → streamed LLM response.
/run <goal>      → launches the AdaptEvolve-MAS evolution pipeline inline.
/help            → show available commands.
File upload      → text extracted and used as context for subsequent messages.
"""
import os, sys, time, json
import streamlit as st
from datetime import datetime, timezone

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
for p in (_APP_DIR, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import adaptevolve_core as ae
from database import (
    insert_session, update_session, new_session_id,
    upsert_chat_session, save_message_rating,
)
from evolution_runner import EvolutionRunner


# ── File parsing ──────────────────────────────────────────────────────────────

def _parse_file(uploaded_file) -> str:
    name = uploaded_file.name
    raw = uploaded_file.read()
    if name.lower().endswith(".pdf"):
        try:
            import fitz
            doc = fitz.open(stream=raw, filetype="pdf")
            text = "\n\n".join(p.get_text() for p in doc)
            doc.close()
            return text
        except Exception:
            pass
    if name.lower().endswith(".ipynb"):
        try:
            nb = json.loads(raw.decode("utf-8", errors="replace"))
            cells = ["\n".join(c.get("source", [])) for c in nb.get("cells", [])]
            return "\n\n".join(cells)
        except Exception:
            pass
    return raw.decode("utf-8", errors="replace")


# ── Chat session persistence ──────────────────────────────────────────────────

def _save_chat(username: str) -> None:
    """Persist current in-memory messages to the chat_sessions table."""
    messages = st.session_state.get("messages", [])
    if not messages:
        return
    chat_id = st.session_state.get("current_chat_id", new_session_id())
    first_user = next(
        (m["content"] for m in messages if m["role"] == "user"), "Chat"
    )
    title = first_user[:60] + ("..." if len(first_user) > 60 else "")
    upsert_chat_session(chat_id, username, title, messages)


# ── Evolution helpers ─────────────────────────────────────────────────────────

def _start_evolution(goal: str, username: str) -> None:
    sid = new_session_id()
    insert_session(sid, username, goal, max_cycles=3, population_size=3, num_generations=2)
    runner = EvolutionRunner()
    runner.start(goal, max_cycles=3, pop_size=3, num_gens=2, documents=[])
    st.session_state["runner"] = runner
    st.session_state["session_id"] = sid
    st.session_state["evolving"] = True
    st.session_state["step_updates"] = []
    st.session_state["final_result"] = None


def _drain_evolution() -> None:
    runner: EvolutionRunner = st.session_state.get("runner")
    if not runner:
        return
    for upd in runner.drain():
        if upd["type"] == "step":
            st.session_state["step_updates"].append(upd)
        elif upd["type"] == "done":
            st.session_state["evolving"] = False
            fs = upd["final_state"]
            sid = st.session_state.get("session_id", "")
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
            score = fs.get("holistic_score", 0) or 0
            code = fs.get("best_code", "")
            report = fs.get("report", "")
            result_md = (
                f"**Evolution complete!** Final score: **{score:.3f}**\n\n"
                + (f"{report}\n\n" if report else "")
                + (f"**Best solution:**\n```python\n{code}\n```" if code else "")
            )
            st.session_state["messages"].append({"role": "assistant", "content": result_md})
        elif upd["type"] == "error":
            st.session_state["evolving"] = False
            st.session_state["messages"].append({
                "role": "assistant",
                "content": f"Evolution error: {upd.get('error', 'unknown')}",
            })


# ── Command dispatch (slash commands only) ────────────────────────────────────

HELP_TEXT = """**AdaptEvolve-MAS — available commands:**

| Command | What it does |
|---|---|
| `/run <goal>` | Launch the 4-agent code evolution pipeline |
| `/status` | Show current evolution session status |
| `/solution` | Show the best code solution found so far |
| `/reset` | Reset the current session |
| `/help` | Show this message |

**Just chat normally** for any general question.

**To evolve code**, describe your goal clearly:
> `/run implement a memory-efficient quicksort for large arrays with many duplicates`"""


def _try_command(prompt: str, username: str):
    """Handle slash commands. Returns reply string, or None if not a command."""
    cmd = prompt.strip()

    if cmd.lower() == "/help":
        return HELP_TEXT

    if cmd.lower().startswith("/run "):
        goal = cmd[5:].strip()
        if not goal:
            return "Please provide a goal after `/run`. Example:\n`/run implement a fast sorting algorithm`"
        _start_evolution(goal, username)
        return (
            f"**Starting evolution for:**\n\n> {goal}\n\n"
            "Running **Strategist -> Solver -> Judge -> Mechanic**.\n\n"
            "_Strategist will search the web for relevant algorithms and techniques, "
            "then evolve and score solutions across multiple cycles. "
            "The Mechanic adapts evaluation criteria after each cycle (meta-learning)._"
        )

    if cmd.lower() == "/status":
        state = ae.session.current_state
        if not state:
            return "No active evolution session. Use `/run <goal>` to start one."
        cycle = state.get("current_cycle", 0)
        score = state.get("holistic_score") or 0
        agent = state.get("current_agent", "-")
        return f"**Cycle:** {cycle} | **Score:** {score:.3f} | **Agent:** `{agent}`"

    if cmd.lower() == "/solution":
        state = ae.session.current_state
        if not state or not state.get("best_code"):
            return "No solution yet. Use `/run <goal>` to start an evolution run."
        score = state.get("holistic_score") or 0
        return f"**Best solution (score {score:.3f}):**\n```python\n{state['best_code']}\n```"

    if cmd.lower() == "/reset":
        ae.session.current_state = None
        ae.session.chat_history = []
        ae.session.is_running = False
        return "Session reset. Ready for a new goal."

    return None  # not a command — caller should stream LLM response


# ── Render ────────────────────────────────────────────────────────────────────

def render(username: str) -> None:
    if st.session_state.get("evolving"):
        _drain_evolution()

    messages = st.session_state.get("messages", [])

    if not messages and not st.session_state.get("evolving"):
        st.markdown(
            "<div style='text-align:center;padding:3rem 0 1.5rem;'>"
            "<h2 style='font-weight:600;'>AdaptEvolve-MAS</h2>"
            "<p style='color:gray;font-size:1rem;'>"
            "Chat naturally, or type "
            "<code>/run &lt;your coding goal&gt;</code> to launch the 4-agent optimizer."
            "</p></div>",
            unsafe_allow_html=True,
        )

    chat_id = st.session_state.get("current_chat_id", "")
    ratings = st.session_state.get("message_ratings", {})

    for idx, msg in enumerate(messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                current = ratings.get(idx)
                up_label   = "**👍**" if current == 1  else "👍"
                down_label = "**👎**" if current == -1 else "👎"
                c1, c2, _ = st.columns([1, 1, 16])
                if c1.button(up_label,   key=f"up_{idx}",   help="Good response"):
                    st.session_state.setdefault("message_ratings", {})[idx] = 1
                    if chat_id:
                        save_message_rating(chat_id, idx, 1)
                    st.rerun()
                if c2.button(down_label, key=f"down_{idx}", help="Bad response"):
                    st.session_state.setdefault("message_ratings", {})[idx] = -1
                    if chat_id:
                        save_message_rating(chat_id, idx, -1)
                    st.rerun()

    if st.session_state.get("evolving"):
        steps = st.session_state.get("step_updates", [])
        latest = steps[-1] if steps else None
        with st.chat_message("assistant"):
            if latest:
                node = latest.get("node", "")
                cycle = latest.get("cycle", "?")
                score = latest.get("score")
                score_txt = f"  -  score {score:.3f}" if score else ""
                st.markdown(f"**Cycle {cycle}** - `{node}`{score_txt}")
            else:
                st.markdown("Evolution pipeline starting...")
        time.sleep(1.5)
        st.rerun()

    ctx = st.session_state.get("file_context", "")
    if ctx:
        col_badge, col_clr = st.columns([8, 1])
        col_badge.caption(f"File context loaded - {len(ctx):,} chars")
        if col_clr.button("x", key="clear_ctx", help="Clear file context"):
            st.session_state["file_context"] = ""
            st.rerun()

    disabled = bool(st.session_state.get("evolving"))
    placeholder = (
        "Evolution running - please wait..."
        if disabled
        else "Message AdaptEvolve... (or /run <goal> to optimize code)"
    )

    prompt = None
    new_files = []

    if not disabled:
        try:
            response = st.chat_input(
                placeholder,
                accept_file="multiple",
                file_type=["pdf", "txt", "md", "py", "ipynb"],
            )
            if response:
                prompt = response.text
                new_files = list(response.files or [])
        except TypeError:
            uploaded = st.file_uploader(
                "Attach file",
                type=["pdf", "txt", "md", "py", "ipynb"],
                label_visibility="collapsed",
                key="fallback_upload",
            )
            if uploaded:
                new_files = [uploaded]
            prompt = st.chat_input(placeholder)
    else:
        st.chat_input(placeholder, disabled=True)

    for f in new_files:
        parsed = _parse_file(f)
        existing = st.session_state.get("file_context", "")
        st.session_state["file_context"] = (
            existing + f"\n\n--- {f.name} ---\n{parsed}" if existing else parsed
        )
        st.session_state["messages"].append({
            "role": "user",
            "content": f"**{f.name}** attached ({len(parsed):,} chars)",
        })
        _save_chat(username)

    if prompt and not disabled:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        reply = _try_command(prompt, username)
        if reply is not None:
            # Slash command — instant response, no streaming
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.session_state["messages"].append({"role": "assistant", "content": reply})
        else:
            # Natural language — stream tokens as they arrive
            file_context = st.session_state.get("file_context", "")
            history = st.session_state.get("messages", [])
            full_prompt = ae.build_prompt(history, file_context, prompt)
            with st.chat_message("assistant"):
                reply = st.write_stream(ae.generate_response_stream(full_prompt))
            st.session_state["messages"].append({"role": "assistant", "content": reply})

        _save_chat(username)
        st.rerun()
