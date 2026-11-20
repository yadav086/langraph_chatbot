import uuid
import streamlit as st
from chatbot import (
    delete_thread,
    export_chat_markdown,
    get_chat_history,
    list_saved_threads,
    stream_message,
    validate_api_key,
)

# 1. Page Configuration
st.set_page_config(
    page_title="Gemini AI Workspace",
    page_icon=":material/smart_toy:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# 2. Pure-Black OLED CSS
st.markdown(
    """
    <style>
    @import url('https://googleapis.com');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    html, body, .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"],
    [data-testid="stChatInputContainer"],
    [data-testid="stSidebar"],
    [data-testid="stSidebarContent"] {
        background-color: #000000 !important;
        background-image: none !important;
        color: #F4F4F5 !important;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid #1A1A1A !important;
    }
    [data-testid="stSidebarUserContent"] {
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
        padding-top: 1.5rem !important;
    }

    input, textarea, select,
    [data-baseweb="input"],
    [data-baseweb="base-input"],
    [data-baseweb="textarea"],
    [data-baseweb="select"] > div {
        background-color: #0A0A0A !important;
        color: #FFFFFF !important;
        border: 1px solid #242424 !important;
        border-radius: 10px !important;
    }

    [data-testid="stChatInput"] {
        background-color: #0D0D0D !important;
        border: 1px solid #262626 !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.85) !important;
    }
    [data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        border: none !important;
    }

    .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
        background-color: #0D0D0D !important;
        color: #E4E4E7 !important;
        border: 1px solid #262626 !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
        background-color: #18181B !important;
        border-color: #3F3F46 !important;
        color: #FFFFFF !important;
    }

    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 0.75rem 0.25rem !important;
        gap: 0.85rem !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #0D0D0D !important;
        border: 1px solid #1E1E1E !important;
        border-radius: 14px !important;
        padding: 0.85rem 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }

    pre, code {
        background-color: #0A0A0A !important;
        border: 1px solid #1F1F1F !important;
        border-radius: 10px !important;
    }

    .block-container {
        max-width: 780px;
        padding-top: 2.25rem;
        padding-bottom: 6rem;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        margin-top: 0.35rem;
    }
    .status-connected {
        background-color: rgba(16, 185, 129, 0.12);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.28);
    }
    .status-disconnected {
        background-color: rgba(245, 158, 11, 0.12);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.28);
    }

    hr {
        border-color: #1A1A1A !important;
        margin: 1.1rem 0 !important;
    }

    #MainMenu, footer, [data-testid="stDecoration"], .stDeployButton, [data-testid="manage-app-button"] {
        display: none !important;
        visibility: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. Session State Initialization
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "connected" not in st.session_state:
    st.session_state.connected = False
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "chat_default"
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None
if "model_name" not in st.session_state:
    st.session_state.model_name = "gemini-3.5-flash-lite"

# 4. Sidebar Controls
with st.sidebar:
    st.markdown("#### :material/key: API Configuration")

    with st.form("api_key_form", border=False):
        api_input = st.text_input(
            "Google API Key",
            type="password",
            placeholder="AIzaSy...",
            value=st.session_state.api_key,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "Connect",
            use_container_width=True,
            icon=":material/bolt:",
        )

    if submitted:
        if not api_input.strip():
            st.error("Please enter your Google API key.")
        else:
            with st.spinner("Verifying credentials..."):
                is_valid, msg = validate_api_key(
                    api_input.strip(), st.session_state.model_name
                )
                if is_valid:
                    st.session_state.api_key = api_input.strip()
                    st.session_state.connected = True
                else:
                    st.session_state.connected = False
                    st.error(msg)

    if st.session_state.connected:
        st.markdown(
            '<div class="status-badge status-connected">● Connected</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-badge status-disconnected">● Not Connected</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Conversation Management
    st.markdown("#### :material/forum: Conversations")
    if st.button(
        "New Conversation",
        use_container_width=True,
        icon=":material/add_comment:",
    ):
        st.session_state.thread_id = f"chat_{uuid.uuid4().hex[:8]}"
        st.rerun()

    if st.session_state.connected:
        saved_threads = list_saved_threads(st.session_state.api_key)
        if saved_threads:
            thread_ids = [t["thread_id"] for t in saved_threads]
            labels = {t["thread_id"]: t["title"] for t in saved_threads}

            if st.session_state.thread_id not in thread_ids:
                thread_ids.insert(0, st.session_state.thread_id)
                labels[st.session_state.thread_id] = "New Chat"

            col_select, col_del = st.columns([5, 1.2], vertical_alignment="center")
            with col_select:
                selected = st.selectbox(
                    "Switch Thread",
                    options=thread_ids,
                    index=thread_ids.index(st.session_state.thread_id),
                    format_func=lambda tid: labels.get(tid, tid),
                    label_visibility="collapsed",
                )
                if selected != st.session_state.thread_id:
                    st.session_state.thread_id = selected
                    st.rerun()

            with col_del:
                if st.button(
                    "",
                    icon=":material/delete:",
                    help="Delete active conversation",
                    use_container_width=True,
                ):
                    delete_thread(st.session_state.thread_id)
                    st.session_state.thread_id = f"chat_{uuid.uuid4().hex[:8]}"
                    st.rerun()

    st.divider()

    # Model & Persona Settings
    st.markdown("#### :material/tune: Model Settings")
    model_input = st.text_input(
        "Model Name",
        value=st.session_state.model_name,
        placeholder="e.g., gemini-3.5-flash-lite",
        help="Enter any valid Google Gemini model ID.",
    )
    model_name = model_input.strip() or "gemini-3.5-flash-lite"
    st.session_state.model_name = model_name

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
    )
    system_prompt = st.text_area(
        "System Instructions",
        placeholder="e.g., You are a senior Python engineer. Be concise.",
        height=85,
    )

# 5. Main Workspace Header
header_col, export_col = st.columns([5, 1.2], vertical_alignment="center")
with header_col:
    st.markdown("## :material/smart_toy: Gemini AI Assistant")
    st.caption(f"Model `{model_name}`  •  Thread `{st.session_state.thread_id}`")

if not st.session_state.connected:
    st.info(
        "Enter your **Google API Key** in the left sidebar and click **Connect** to begin.",
        icon=":material/lock:",
    )
    st.stop()

history = get_chat_history(st.session_state.api_key, st.session_state.thread_id)

with export_col:
    if history:
        md_content = export_chat_markdown(
            st.session_state.api_key, st.session_state.thread_id
        )
        st.download_button(
            label="Export",
            data=md_content,
            file_name=f"{st.session_state.thread_id}.md",
            mime="text/markdown",
            icon=":material/download:",
            use_container_width=True,
        )

st.divider()

# 6. Starter Prompts for Empty Threads
if not history:
    st.markdown("#### How can I help you today?")
    c1, c2, c3 = st.columns(3)
    with c1:
