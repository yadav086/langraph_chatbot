import sqlite3
import uuid
from typing import Annotated, Generator

import streamlit as st
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Workspace",
    page_icon=":material/smart_toy:",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. BACKEND: STATE, SQLITE & MULTI-PROVIDER LLM
# ============================================================

class ChatState(BaseModel):
    message: Annotated[list[BaseMessage], add_messages]


@st.cache_resource
def get_sqlite_resources():
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    return conn, checkpointer


conn, checkpointer = get_sqlite_resources()


def _create_llm(api_key: str, model_name: str, temperature: float = 0.0, **kwargs):
    """Initializes any supported LangChain chat model dynamically."""
    provider = None
    clean_model = model_name.strip()

    if ":" not in clean_model:
        lower_name = clean_model.lower()
        if lower_name.startswith("gemini"):
            provider = "google_genai"
        elif lower_name.startswith("grok"):
            provider = "xai"
        elif lower_name.startswith(("gpt-", "o1", "o3")):
            provider = "openai"
        elif lower_name.startswith("claude"):
            provider = "anthropic"

    return init_chat_model(
        model=clean_model,
        model_provider=provider,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )


def get_chatbot(
    api_key: str,
    model_name: str,
    temperature: float = 0.0,
    system_prompt: str = "",
):
    """Builds and compiles the LangGraph chatbot with SQLite persistence."""
    llm = _create_llm(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
    )

    def chat_node(state: ChatState):
        messages = list(state.message)
        if system_prompt.strip():
            messages = [SystemMessage(content=system_prompt.strip())] + messages
        response = llm.invoke(messages)
        return {"message": [response]}

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)

    return graph.compile(checkpointer=checkpointer)


def _get_checkpoint_graph():
    """Compiles a lightweight graph solely for reading SQLite state without needing an LLM."""
    graph = StateGraph(ChatState)
    graph.add_node("noop", lambda state: state)
    graph.add_edge(START, "noop")
    graph.add_edge("noop", END)
    return graph.compile(checkpointer=checkpointer)


def validate_api_key(api_key: str, model_name: str) -> tuple[bool, str]:
    """Tests the API key against the user-specified model."""
    if not model_name.strip():
        return False, "Please enter a model name first."
    try:
        llm = _create_llm(
            api_key=api_key,
            model_name=model_name,
            max_tokens=5,
        )
        llm.invoke("hi")
        return True, f"Connected ({model_name})"
    except Exception as e:
        return False, f"Connection failed: {e}"


def list_saved_threads() -> list[dict]:
    """Lists all saved thread_ids from SQLite along with a preview title."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        return []

    threads = []
    reader = _get_checkpoint_graph()
    for (tid,) in rows:
        state = reader.get_state({"configurable": {"thread_id": tid}})
        title = tid
        if state and state.values and "message" in state.values:
            for msg in state.values["message"]:
                if isinstance(msg, HumanMessage) and isinstance(msg.content, str):
                    clean = msg.content.strip().replace("\n", " ")
                    title = clean[:32] + ("..." if len(clean) > 32 else "")
                    break
        threads.append({"thread_id": tid, "title": title})
    return list(reversed(threads))


def delete_thread(thread_id: str) -> None:
    """Deletes all checkpoint records for a given thread_id from SQLite."""
    cursor = conn.cursor()
    for table in ("checkpoints", "checkpoint_blobs", "checkpoint_writes"):
        try:
            cursor.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
        except sqlite3.OperationalError:
            pass
    conn.commit()


def get_chat_history(thread_id: str) -> list[dict]:
    """Loads existing messages from SQLite for the given thread_id."""
    reader = _get_checkpoint_graph()
    config = {"configurable": {"thread_id": thread_id}}
    state = reader.get_state(config)

    history = []
    if state and state.values and "message" in state.values:
        for msg in state.values["message"]:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.text:
                history.append({"role": "assistant", "content": msg.text})
    return history


def export_chat_markdown(thread_id: str) -> str:
    """Formats the conversation history as a downloadable Markdown string."""
    history = get_chat_history(thread_id)
    lines = [f"# Chat Export ({thread_id})\n"]
    for msg in history:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"### {speaker}\n{msg['content']}\n")
    return "\n".join(lines)


def stream_message(
    api_key: str,
    thread_id: str,
    user_text: str,
    model_name: str,
    temperature: float = 0.0,
    system_prompt: str = "",
) -> Generator[str, None, None]:
    """Streams response tokens word-by-word while saving state to SQLite."""
    chatbot = get_chatbot(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        system_prompt=system_prompt,
    )
    config = {"configurable": {"thread_id": thread_id}}

    for chunk, _metadata in chatbot.stream(
        {"message": [HumanMessage(content=user_text)]},
        config=config,
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessageChunk) and chunk.text:
            yield chunk.text


# ============================================================
# 3. CUSTOM STYLING
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
# 4. SESSION STATE INITIALIZATION
# ============================================================

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

if "model_name" not in st.session_state:
    st.session_state.model_name = ""

if "connected" not in st.session_state:
    st.session_state.connected = False

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "chat_default"

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# ============================================================
# 5. SIDEBAR CONFIGURATION
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # API & MODEL CONFIGURATION
    # --------------------------------------------------------

    st.markdown("### :material/key: Provider Configuration")

    with st.form("api_key_form", border=False):

        model_input = st.text_input(
            "Model Name",
            placeholder="e.g., gemini-2.5-flash, grok-2-latest, gpt-4o",
            value=st.session_state.model_name,
            help=(
                "Enter any supported model name (e.g., gemini-2.5-flash, "
                "grok-2-latest, gpt-4o, claude-3-5-sonnet-latest)."
            ),
        )

        api_input = st.text_input(
            "API Key",
            type="password",
            placeholder="Enter your API key",
            value=st.session_state.api_key,
        )

        submitted = st.form_submit_button(
            "Connect",
            use_container_width=True,
            type="primary",
            icon=":material/bolt:",
        )

    if submitted:

        if not model_input.strip() or not api_input.strip():

            st.error("Please enter both a Model Name and an API Key.")

        else:

            with st.spinner("Verifying credentials..."):

                try:
                    is_valid, msg = validate_api_key(
                        api_input.strip(),
                        model_input.strip(),
                    )

                    if is_valid:

                        st.session_state.api_key = api_input.strip()
                        st.session_state.model_name = model_input.strip()
                        st.session_state.connected = True

                        st.success(msg)

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
            f"""
            <div class="status-badge status-connected">
                ● Connected ({st.session_state.model_name})
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

        st.session_state.thread_id = f"chat_{uuid.uuid4().hex[:8]}"
        st.session_state.pending_prompt = None
        st.rerun()

    saved_threads = list_saved_threads()

    if saved_threads or st.session_state.connected:

        thread_ids = [t["thread_id"] for t in saved_threads]
        labels = {t["thread_id"]: t["title"] for t in saved_threads}

        if st.session_state.thread_id not in thread_ids:
            thread_ids.insert(0, st.session_state.thread_id)
            labels[st.session_state.thread_id] = "New Chat"

        col_select, col_del = st.columns([4, 1])

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
                st.session_state.pending_prompt = None
                st.rerun()

    st.divider()

    # --------------------------------------------------------
    # MODEL SETTINGS
    # --------------------------------------------------------

    st.markdown("### :material/tune: Model Settings")

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
        help="Define the behaviour and personality of AI.",
    )

    st.divider()

    st.caption("Powered by LangGraph + Multi-Provider LLMs")


# ============================================================
# 6. MAIN WORKSPACE HEADER
# ============================================================

header_col, export_col = st.columns([4, 1])

with header_col:

    st.markdown("## :material/smart_toy: AI Assistant")

    active_model_label = (
        st.session_state.model_name
        if st.session_state.model_name
        else "No Model Selected"
    )

    st.markdown(
        f"""
        <div class="workspace-caption">
            Model: <strong>{active_model_label}</strong>
            &nbsp; • &nbsp;
            Thread: {st.session_state.thread_id}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 7. API CONNECTION CHECK
# ============================================================

if not st.session_state.connected:

    st.info(
        "Enter your **Model Name** and **API Key** in the left sidebar "
        "and click **Connect** to begin.",
        icon=":material/lock:",
    )

    st.stop()


# ============================================================
# 8. LOAD CHAT HISTORY
# ============================================================

history = get_chat_history(st.session_state.thread_id)


# ============================================================
# 9. EXPORT CHAT HISTORY
# ============================================================

with export_col:

    if history:

        md_content = export_chat_markdown(st.session_state.thread_id)

        st.download_button(
            label="Export",
            data=md_content,
            file_name=f"{st.session_state.thread_id}.md",
            mime="text/markdown",
            icon=":material/download:",
            use_container_width=True,
        )


# ============================================================
# 10. STARTER PROMPTS
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
# 11. RENDER CONVERSATION HISTORY
# ============================================================

for msg in history:

    avatar = (
        ":material/person:"
        if msg["role"] == "user"
        else ":material/smart_toy:"
    )

    with st.chat_message(msg["role"], avatar=avatar):

        st.markdown(msg["content"])


# ============================================================
# 12. HANDLE USER INPUT & TOKEN STREAMING
# ============================================================

prompt = (
    st.chat_input("Ask anything...")
    or st.session_state.pending_prompt
)

if prompt:

    if not st.session_state.model_name.strip():

        st.error(
            "Please enter a Model Name and connect "
            "in the sidebar before sending a message."
        )
        st.stop()

    st.session_state.pending_prompt = None

    # Display User Message

    with st.chat_message("user", avatar=":material/person:"):

        st.markdown(prompt)

    # Generate AI Response

    with st.chat_message("assistant", avatar=":material/smart_toy:"):

        try:

            st.write_stream(
                stream_message(
                    api_key=st.session_state.api_key,
                    thread_id=st.session_state.thread_id,
                    user_text=prompt,
                    model_name=st.session_state.model_name.strip(),
                    temperature=temperature,
                    system_prompt=system_prompt,
                )
            )

        except Exception as e:

            st.error(f"Unable to generate response: {str(e)}")

    st.rerun()
