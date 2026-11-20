
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


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Gemini AI Workspace",
    page_icon=":material/smart_toy:",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 5rem;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 5px 12px;
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

        .workspace-caption {
            color: #888888;
            font-size: 0.85rem;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. SESSION STATE INITIALIZATION
# ============================================================

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "connected" not in st.session_state:
    st.session_state.connected = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "chat_default"

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ============================================================
# 4. SIDEBAR CONFIGURATION
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # API CONFIGURATION
    # --------------------------------------------------------

    st.markdown("### :material/key: API Configuration")

    with st.form("api_key_form", border=False):

        api_input = st.text_input(
            "Google API Key",
            type="password",
            placeholder="Enter your Google API key",
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

                try:
                    is_valid, msg = validate_api_key(
                        api_input.strip()
                    )

                    if is_valid:

                        st.session_state.api_key = api_input.strip()
                        st.session_state.connected = True

                        st.success("API key verified successfully.")

                    else:

                        st.session_state.connected = False
                        st.session_state.api_key = ""

                        st.error(msg)

                except Exception as e:

                    st.session_state.connected = False
                    st.error(f"API validation failed: {str(e)}")

    # Connection Status

    if st.session_state.connected:

        st.markdown(
            """
            <div class="status-badge status-connected">
                ● Connected
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="status-badge status-disconnected">
                ● Not Connected
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()


    # --------------------------------------------------------
    # CONVERSATION MANAGEMENT
    # --------------------------------------------------------

    st.markdown("### :material/forum: Conversations")

    if st.button(
        "New Conversation",
        use_container_width=True,
        icon=":material/add_comment:",
    ):

        st.session_state.thread_id = (
            f"chat_{uuid.uuid4().hex[:8]}"
        )

        st.session_state.pending_prompt = None

        st.rerun()

    if st.session_state.connected:

        saved_threads = list_saved_threads(
            st.session_state.api_key
        )

        if saved_threads:

            thread_ids = [
                t["thread_id"] for t in saved_threads
            ]

            labels = {
                t["thread_id"]: t["title"]
                for t in saved_threads
            }

            if st.session_state.thread_id not in thread_ids:

                thread_ids.insert(
                    0,
                    st.session_state.thread_id
                )

                labels[st.session_state.thread_id] = "New Chat"

            col_select, col_del = st.columns([5, 1])

            with col_select:

                selected = st.selectbox(
                    "Switch Thread",
                    options=thread_ids,
                    index=thread_ids.index(
                        st.session_state.thread_id
                    ),
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

                    st.session_state.thread_id = (
                        f"chat_{uuid.uuid4().hex[:8]}"
                    )

                    st.session_state.pending_prompt = None

                    st.rerun()

    st.divider()


    # --------------------------------------------------------
    # MODEL SETTINGS
    # --------------------------------------------------------

    st.markdown("### :material/tune: Model Settings")

    # Manual Model Name Input - No Dropdown

    model_name = st.text_input(
        "Gemini Model Name",
        value="gemini-2.5-flash",
        placeholder="Enter Gemini model name",
        help=(
            "Enter a Gemini model identifier supported "
            "by your Google API key."
        ),
        key="model_name",
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Controls the creativity of the AI response.",
    )

    system_prompt = st.text_area(
        "System Instructions",
        placeholder=(
            "e.g., You are a senior Python engineer. "
            "Be concise and provide practical examples."
        ),
        height=110,
        help="Define the behaviour and personality of Gemini AI.",
    )

    st.divider()

    st.caption("Powered by Google Gemini + LangGraph")


# ============================================================
# 5. MAIN WORKSPACE HEADER
# ============================================================

header_col, export_col = st.columns([5, 1])

with header_col:

    st.markdown(
        "## :material/smart_toy: Gemini AI Assistant"
    )

    st.markdown(
        f"""
        <div class="workspace-caption">
            LangGraph + SQLite Memory
            &nbsp; • &nbsp;
            Thread: {st.session_state.thread_id}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 6. API CONNECTION CHECK
# ============================================================

if not st.session_state.connected:

    st.info(
        "Enter your **Google API Key** in the left sidebar "
        "and click **Connect** to begin.",
        icon=":material/lock:",
    )

    st.stop()


# ============================================================
# 7. LOAD CHAT HISTORY
# ============================================================

history = get_chat_history(
    st.session_state.api_key,
    st.session_state.thread_id,
)


# ============================================================
# 8. EXPORT CHAT HISTORY
# ============================================================

with export_col:

    if history:

        md_content = export_chat_markdown(
            st.session_state.api_key,
            st.session_state.thread_id,
        )

        st.download_button(
            label="Export",
            data=md_content,
            file_name=f"{st.session_state.thread_id}.md",
            mime="text/markdown",
            icon=":material/download:",
            use_container_width=True,
        )


# ============================================================
# 9. STARTER PROMPTS
# ============================================================

if not history:

    st.markdown("#### How can I help you today?")

    c1, c2, c3 = st.columns(3)

    with c1:

        if st.button(
            "Explain LangGraph Memory",
            use_container_width=True,
        ):

            st.session_state.pending_prompt = (
                "Explain how LangGraph checkpointers "
                "persist conversation memory."
            )

            st.rerun()

    with c2:

        if st.button(
            "Python SQLite Script",
            use_container_width=True,
        ):

            st.session_state.pending_prompt = (
                "Write a clean Python script to query "
                "a SQLite database safely."
            )

            st.rerun()

    with c3:

        if st.button(
            "AI Trends",
            use_container_width=True,
        ):

            st.session_state.pending_prompt = (
                "Give me 5 concise bullet points "
                "on agentic AI workflows."
            )

            st.rerun()


# ============================================================
# 10. RENDER CONVERSATION HISTORY
# ============================================================

for msg in history:

    avatar = (
        ":material/person:"
        if msg["role"] == "user"
        else ":material/smart_toy:"
    )

    with st.chat_message(
        msg["role"],
        avatar=avatar,
    ):

        st.markdown(msg["content"])


# ============================================================
# 11. HANDLE USER INPUT & TOKEN STREAMING
# ============================================================

prompt = (
    st.chat_input("Ask anything...")
    or st.session_state.pending_prompt
)

if prompt:

    # Validate model name

    if not model_name.strip():

        st.error(
            "Please enter a Gemini model name "
            "in the sidebar before sending a message."
        )

        st.stop()

    st.session_state.pending_prompt = None

    # Display User Message

    with st.chat_message(
        "user",
        avatar=":material/person:",
    ):

        st.markdown(prompt)

    # Generate AI Response

    with st.chat_message(
        "assistant",
        avatar=":material/smart_toy:",
    ):

        try:

            st.write_stream(
                stream_message(
                    api_key=st.session_state.api_key,
                    thread_id=st.session_state.thread_id,
                    user_text=prompt,
                    model_name=model_name.strip(),
                    temperature=temperature,
                    system_prompt=system_prompt,
                )
            )

        except Exception as e:

            st.error(
                f"Unable to generate response: {str(e)}"
            )

    st.rerun()
