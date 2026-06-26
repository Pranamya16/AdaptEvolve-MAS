import plotly.graph_objects as go
import streamlit as st


def render_trajectory_chart(score_trajectory: list[float], title: str = "Holistic Score Over Cycles") -> None:
    """Plot holistic score per cycle as a line chart."""
    if not score_trajectory:
        st.caption("No trajectory data yet.")
        return

    cycles = list(range(1, len(score_trajectory) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cycles,
        y=score_trajectory,
        mode="lines+markers",
        line=dict(color="#7C3AED", width=2.5),
        marker=dict(size=8, color="#7C3AED"),
        name="Holistic Score",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Cycle",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_xaxes(gridcolor="#e5e7eb")
    fig.update_yaxes(gridcolor="#e5e7eb")
    st.plotly_chart(fig, use_container_width=True)


def render_dimension_chart(dimension_scores: dict) -> None:
    """Render a bar chart of per-dimension evaluation scores."""
    if not dimension_scores:
        return
    dims = list(dimension_scores.keys())
    vals = [float(dimension_scores[d]) for d in dims]
    fig = go.Figure(go.Bar(
        x=dims,
        y=vals,
        marker_color="#7C3AED",
    ))
    fig.update_layout(
        title="Score Breakdown by Dimension",
        yaxis=dict(range=[0, 1.05]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=40, r=20, t=50, b=80),
    )
    st.plotly_chart(fig, use_container_width=True)
