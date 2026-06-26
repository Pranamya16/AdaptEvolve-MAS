"""
AdaptEvolve-MAS — Streamlit entry point
Run from project root:  streamlit run streamlit_app/app.py
"""
import os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
for p in (_HERE, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import streamlit as st
    for _var in ("AE2", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        _val = st.secrets.get(_var, "")
        if _val:
            os.environ.setdefault(_var, _val)
            break
except Exception:
    pass

import yaml
import streamlit as st
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth

from database import (
    init_db, get_user_sessions, get_user_chat_sessions,
    get_chat_session_messages, get_message_ratings, new_session_id,
)

init_db()

st.set_page_config(
    page_title="AdaptEvolve-MAS",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit's default multipage nav and top bar decoration
st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none !important; }
.stDeployButton { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Auth ──────────────────────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(_HERE, "config.yaml")
with open(_CONFIG_PATH) as f:
    config = yaml.load(f, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config["credentials"],
    config["cookie"]["name"],
    config["cookie"]["key"],
    config["cookie"]["expiry_days"],
)

try:
    auth_result = authenticator.login(location="main")
    if isinstance(auth_result, tuple):
        name, auth_status, username = auth_result
    else:
        name = st.session_state.get("name")
        auth_status = st.session_state.get("authentication_status")
        username = st.session_state.get("username")
except Exception as exc:
    st.error(f"Authentication error: {exc}")
    st.stop()

if auth_status is False:
    st.error("Incorrect username or password.")
    st.stop()
elif auth_status is None:
    st.warning("Please log in to continue.")
    st.stop()

# ── Load pipeline once ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AdaptEvolve-MAS pipeline…")
def _load_pipeline():
    import adaptevolve_core  # noqa: triggers module-level init
    return True

_load_pipeline()

# ── Session state defaults ────────────────────────────────────────────────────
if "view" not in st.session_state:
    st.session_state["view"] = "chat"
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "current_chat_id" not in st.session_state:
    st.session_state["current_chat_id"] = new_session_id()
if "file_context" not in st.session_state:
    st.session_state["file_context"] = ""

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 AdaptEvolve-MAS")
    st.caption(f"Logged in as **{name}**")
    st.divider()

    if st.button("✏️  New Chat", use_container_width=True, type="primary"):
        st.session_state["messages"] = []
        st.session_state["current_chat_id"] = new_session_id()
        st.session_state["file_context"] = ""
        st.session_state["view"] = "chat"
        for key in ("runner", "evolving", "step_updates", "final_result", "session_id", "message_ratings"):
            st.session_state.pop(key, None)
        st.rerun()

    # ── Chat history ──────────────────────────────────────────────────────────
    try:
        chat_sessions = get_user_chat_sessions(username, limit=20)
    except Exception:
        chat_sessions = []

    try:
        evo_sessions = get_user_sessions(username, limit=10)
    except Exception:
        evo_sessions = []

    if chat_sessions or evo_sessions:
        st.markdown("**Chats**")

    for s in chat_sessions:
        label = s.get("title") or "Chat"
        if len(label) > 36:
            label = label[:36] + "..."
        if st.button(
            label,
            key=f"chat_{s['chat_id']}",
            use_container_width=True,
            help=s.get("title", ""),
        ):
            msgs = get_chat_session_messages(s["chat_id"])
            raw_ratings = get_message_ratings(s["chat_id"])
            st.session_state["messages"] = msgs
            st.session_state["message_ratings"] = {int(k): v for k, v in raw_ratings.items()}
            st.session_state["current_chat_id"] = s["chat_id"]
            st.session_state["view"] = "chat"
            st.rerun()

    if evo_sessions:
        st.markdown("**Evolutions**")
        for s in evo_sessions:
            raw_goal = s.get("goal") or "Run"
            label = raw_goal[:32] + "..." if len(raw_goal) > 32 else raw_goal
            score = s.get("final_score")
            score_txt = f"  {score:.2f}" if score else ""
            if st.button(
                f"{label}{score_txt}",
                key=f"evo_{s['session_id']}",
                use_container_width=True,
                help=raw_goal,
            ):
                st.session_state["view"] = "session_detail"
                st.session_state["selected_session_id"] = s["session_id"]
                st.rerun()

    st.divider()

    if st.button("⚙️ Settings", use_container_width=True):
        st.session_state["view"] = "settings"
        st.rerun()

    if st.button("Logout", use_container_width=True, key="logout_btn"):
        # 'unrendered' skips button rendering and directly clears auth state + cookie
        authenticator.logout(location="unrendered")
        for _k in ("authentication_status", "name", "username",
                   "messages", "current_chat_id", "file_context",
                   "message_ratings", "view", "runner", "evolving",
                   "step_updates", "final_result", "session_id"):
            st.session_state.pop(_k, None)
        st.rerun()

# ── Route ─────────────────────────────────────────────────────────────────────
view = st.session_state.get("view", "chat")

if view == "chat":
    from _pages.chat import render
    render(username)

elif view == "settings":
    from _pages.settings import render
    render()

elif view == "session_detail":
    from _pages.session_detail import render
    render(st.session_state.get("selected_session_id", ""))
