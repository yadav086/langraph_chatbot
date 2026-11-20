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

# 2. Custom Styling
st.markdown(
    """
    <style>
    .block-container {
        max-width: 820px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.25rem;
    }
    .status-connected {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.35);
    }
    .status-disconnected {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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

# 4. Sidebar Controls
with st.sidebar:
    st.markdown("### :material/key: API Configuration")

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
            type="primary",
            icon=":material/bolt:",
        )

    if submitted:
        if not api_input.strip():
            st.error("Please enter your Google API key.")
        else:
            with st.spinner("Verifying credentials..."):
                is_valid, msg = validate_api_key(api_input.strip())
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
    st.markdown("### :material/forum: Conversations")
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

            col_select, col_del = st.columns([5, 1])
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
    st.markdown("### :material/tune: Model Settings")
    model_name = st.selectbox(
        "Model",
        options=["gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
    )
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
        height=90,
    )

# 5. Main Workspace Header
header_col, export_col = st.columns([5, 1])
with header_col:
    st.markdown("## :material/smart_toy: Gemini AI Assistant")
    st.caption(f"LangGraph + SQLite Memory  •  Thread `{st.session_state.thread_id}`")

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

# 6. Starter Prompts for Empty Threads
if not history:
    st.markdown("#### How can I help you today?")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Explain LangGraph memory", use_container_width=True):
            st.session_state.pending_prompt = (
                "Explain how LangGraph checkpointers persist conversation memory."
            )
            st.rerun()
    with c2:
        if st.button("Write a Python SQLite script", use_container_width=True):
            st.session_state.pending_prompt = (
                "Write a clean Python script to query a SQLite database safely."
            )
            st.rerun()
    with c3:
        if st.button("Summarize key AI trends", use_container_width=True):
            st.session_state.pending_prompt = (
                "Give me 5 concise bullet points on agentic AI workflows."
            )
            st.rerun()

# 7. Render Message History with Human & AI Icons
for msg in history:
    avatar = ":material/person:" if msg["role"] == "user" else ":material/smart_toy:"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# 8. Handle User Input & Live Token Streaming
prompt = st.chat_input("Ask anything...") or st.session_state.pending_prompt
if prompt:
    st.session_state.pending_prompt = None

    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        st.write_stream(
            stream_message(
                api_key=st.session_state.api_key,
                thread_id=st.session_state.thread_id,
                user_text=prompt,
                model_name=model_name,
                temperature=temperature,
                system_prompt=system_prompt,
            )
        )
    st.rerun()
