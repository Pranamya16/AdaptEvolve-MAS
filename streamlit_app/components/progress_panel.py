import streamlit as st

_AGENT_ICONS = {
    "strategist": "🧠",
    "solver": "⚙️",
    "judge": "⚖️",
    "mechanic": "🛠️",
}

_AGENT_LABELS = {
    "strategist": "Strategist — Research & Plan",
    "solver": "Solver — Evolutionary Code Generation",
    "judge": "Judge — Multi-Criteria Evaluation",
    "mechanic": "Mechanic — Meta-Learning",
}


def render_step_card(update: dict) -> None:
    """Render a single pipeline step as a collapsible card."""
    node = update.get("node", "unknown")
    cycle = update.get("cycle", 0)
    score = update.get("score", 0.0)
    icon = _AGENT_ICONS.get(node, "📍")
    label = _AGENT_LABELS.get(node, node.capitalize())

    title = f"{icon} Cycle {cycle} · {label} | Score: {score:.4f}"
    with st.expander(title, expanded=False):
        messages = update.get("messages", [])
        if messages:
            for msg in messages:
                role = msg.get("role", "system")
                content = msg.get("content", "")
                if role == "assistant":
                    st.markdown(f"> {content}")
                else:
                    st.caption(content)

        mechanic = update.get("mechanic_analysis", "")
        if mechanic and node == "mechanic":
            st.markdown("**Mechanic Analysis:**")
            st.info(mechanic[:800] + ("…" if len(mechanic) > 800 else ""))

        ops = update.get("proposed_operators", [])
        if ops:
            st.markdown(f"**Proposed operators:** {', '.join(o.get('name','?') for o in ops)}")

        code = update.get("best_code", "")
        if code and node == "judge":
            st.markdown("**Best solution so far:**")
            st.code(code[:600] + ("…" if len(code) > 600 else ""), language="python")
