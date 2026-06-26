"""
Settings page — choose the LLM backend (local Ollama or cloud Gemini),
set the model, manage the Gemini API key, and tune evaluation criteria weights.
Mirrors the Gradio "Settings" tab.
"""
import os, sys
import streamlit as st

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_APP_DIR)
for p in (_APP_DIR, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


def render() -> None:
    st.title("⚙️ Settings")

    import adaptevolve_core as ae

    # ── System Status ────────────────────────────────────────────────────────
    st.markdown("#### System Status")
    backend = getattr(ae, "LLM_BACKEND", "?")
    model   = (
        getattr(ae, "OLLAMA_MODEL_NAME", "?")
        if backend == "ollama"
        else getattr(ae, "MODEL_NAME", "?")
    )
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("LLM Backend", backend.upper())
    col_b.metric("Model", model.split(":")[0])
    col_c.metric("Web Search", "Active")

    st.markdown(
        "| Module | Status | Details |\n"
        "|---|---|---|\n"
        "| **Strategist** (AgenticRAG) | Active | DuckDuckGo search + vector store on every `/run` |\n"
        "| **Meta-learning** (Mechanic) | Active | LLM re-weights evaluation criteria each cycle (OOC) |\n"
        "| **Evolutionary Engine** | Active | Mutation + crossover operators per generation |\n"
        "| **Convergence Guard** | Active | Stops if score plateau < 0.01 over 3 cycles or max cycles reached |"
    )
    st.divider()

    # ── LLM backend ─────────────────────────────────────────────────────────
    st.markdown("#### LLM Backend")
    st.caption(
        "Run fully offline with a local model (Ollama) — no API quota — or use "
        "cloud Gemini for maximum quality."
    )

    backends = ["ollama", "gemini"]
    current_backend = getattr(ae, "LLM_BACKEND", "ollama")
    backend = st.radio(
        "Backend",
        backends,
        index=backends.index(current_backend) if current_backend in backends else 0,
        format_func=lambda b: "🖥️ Ollama (local, offline)" if b == "ollama" else "☁️ Gemini (cloud)",
        horizontal=True,
    )

    if backend == "ollama":
        default_model = getattr(ae, "OLLAMA_MODEL_NAME", "qwen2.5-coder:7b-instruct")
        model = st.text_input(
            "Ollama model",
            value=default_model,
            help="e.g. qwen2.5-coder:7b-instruct (quality) or qwen2.5-coder:3b-instruct (fast). "
                 "Pull it once with:  ollama pull <model>",
        )
        host = st.text_input("Ollama host", value=getattr(ae, "OLLAMA_HOST", "http://localhost:11434"))
    else:
        model = st.text_input("Gemini model", value=getattr(ae, "MODEL_NAME", "gemini-3-flash-preview"))
        host = None

    col_apply, col_test = st.columns(2)

    if col_apply.button("Apply Backend", use_container_width=True, type="primary"):
        try:
            if backend == "ollama":
                ae.OLLAMA_MODEL_NAME = model.strip()
                if host:
                    ae.OLLAMA_HOST = host.strip()
                    os.environ["OLLAMA_HOST"] = host.strip()
                ae.llm = ae.OllamaLLM(model_name=model.strip(), host=ae.OLLAMA_HOST)
                os.environ["AE_LLM_MODEL"] = model.strip()
            else:
                ae.MODEL_NAME = model.strip()
                ae.llm = ae.GeminiLLM(model_name=model.strip())
            ae.LLM_BACKEND = backend
            os.environ["AE_LLM_BACKEND"] = backend
            st.success(f"Active backend set to **{backend}** · model `{model.strip()}`.")
        except Exception as e:
            st.error(f"Failed to switch backend: {e}")

    if col_test.button("Test Connection", use_container_width=True):
        try:
            test_llm = (
                ae.OllamaLLM(model_name=model.strip(), host=host.strip() if host else None)
                if backend == "ollama"
                else ae.GeminiLLM(model_name=model.strip())
            )
            reply = test_llm.generate("Reply with the single word: OK", retries=1)
            st.success(f"✅ {backend} responded: {reply.strip()[:80]}")
        except Exception as e:
            hint = ""
            if backend == "ollama":
                hint = ("\n\nIs Ollama running and the model pulled?  Try:\n"
                        f"```\nollama pull {model.strip()}\nollama list\n```")
            st.error(f"Connection failed: {e}{hint}")

    st.caption(f"Active backend: **{getattr(ae, 'LLM_BACKEND', '?')}**")
    st.divider()

    col_left, col_right = st.columns(2)

    # ── Gemini API key (only relevant for the Gemini backend) ────────────────
    with col_left:
        st.markdown("#### Gemini API Key")
        new_key = st.text_input("API Key", type="password",
                                placeholder="Paste your Gemini key here to update it")
        if st.button("Update API Key", use_container_width=True):
            if new_key.strip():
                try:
                    from google import genai
                    test_client = genai.Client(api_key=new_key.strip())
                    test_client.models.generate_content(
                        model=ae.MODEL_NAME, contents="Say OK"
                    )
                    ae.API_KEY = new_key.strip()
                    ae.client = test_client
                    if isinstance(ae.llm, ae.GeminiLLM):
                        ae.llm.client = test_client
                    os.environ["AE2"] = new_key.strip()
                    st.success("API key updated and verified.")
                except Exception as e:
                    st.error(f"Key verification failed: {e}")
            else:
                st.warning("Please enter a key.")

        current_key = ae.API_KEY or os.environ.get("AE2", "")
        if current_key:
            st.caption(f"Current key: `{current_key[:8]}…`")
        else:
            st.caption("No key set. Only needed for the Gemini backend.")

    # ── Evaluation weights ────────────────────────────────────────────────────
    with col_right:
        st.markdown("#### Evaluation Criteria Weights")
        state = ae.session.current_state
        current = state.get("evaluation_criteria", {}) if state else {}

        exec_w = st.slider("Execution Time Weight",   0.0, 1.0,
                           float(current.get("execution_time_weight", 0.30)), 0.05)
        mem_w  = st.slider("Memory Usage Weight",     0.0, 1.0,
                           float(current.get("memory_usage_weight",   0.25)), 0.05)
        corr_w = st.slider("Correctness Weight",      0.0, 1.0,
                           float(current.get("correctness_weight",    0.30)), 0.05)
        qual_w = st.slider("Code Quality Weight",     0.0, 1.0,
                           float(current.get("code_quality_weight",   0.15)), 0.05)

        total = exec_w + mem_w + corr_w + qual_w
        st.caption(f"Sum: {total:.2f} (should be ≈ 1.0)")

        if st.button("Apply Weights", use_container_width=True, type="primary"):
            new_criteria = {
                "execution_time_weight": exec_w,
                "memory_usage_weight":   mem_w,
                "correctness_weight":    corr_w,
                "code_quality_weight":   qual_w,
            }
            if ae.session.current_state:
                ae.session.current_state["evaluation_criteria"] = new_criteria
                st.success("Weights updated for the current session.")
            else:
                st.info("No active session — weights will apply when the next session starts.")
            st.session_state["default_criteria"] = new_criteria
