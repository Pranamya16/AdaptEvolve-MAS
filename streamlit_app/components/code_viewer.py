import streamlit as st


def render_code(code: str, label: str = "Best Solution", score: float | None = None) -> None:
    """Display a code block with an optional score badge and copy hint."""
    if not code:
        st.caption("No solution generated yet.")
        return

    header = f"### {label}"
    if score is not None:
        header += f"  ·  Score: `{score:.4f}`"
    st.markdown(header)
    st.code(code, language="python")
    st.download_button(
        label="Download solution.py",
        data=code,
        file_name="solution.py",
        mime="text/x-python",
    )
