import sqlite3
from typing import Annotated, Generator
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel


class ChatState(BaseModel):
    message: Annotated[list[BaseMessage], add_messages]


# Shared SQLite connection (check_same_thread=False is required for Streamlit)
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
checkpointer = SqliteSaver(conn)


def get_chatbot(
    api_key: str,
    #model_name: str = "gemini-3.5-flash-lite",
    model_name: str = "",
    temperature: float = 0.0,
    system_prompt: str = "",
):
    """Builds and compiles the LangGraph chatbot with SQLite persistence."""
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
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


def validate_api_key(api_key: str) -> tuple[bool, str]:
    """Tests the Google API key with a minimal request."""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            google_api_key=api_key,
            max_tokens=5,
        )
        llm.invoke("hi")
        return True, "Connected"
    except Exception as e:
        return False, f"Connection failed: {e}"


def list_saved_threads(api_key: str) -> list[dict]:
    """Lists all saved thread_ids from SQLite along with a preview title."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT thread_id FROM checkpoints")
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        return []

    threads = []
    chatbot = get_chatbot(api_key)
    for (tid,) in rows:
        state = chatbot.get_state({"configurable": {"thread_id": tid}})
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


def get_chat_history(api_key: str, thread_id: str) -> list[dict]:
    """Loads existing messages from SQLite for the given thread_id."""
    chatbot = get_chatbot(api_key)
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)

    history = []
    if state and state.values and "message" in state.values:
        for msg in state.values["message"]:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage) and msg.text:
                history.append({"role": "assistant", "content": msg.text})
    return history


def export_chat_markdown(api_key: str, thread_id: str) -> str:
    """Formats the conversation history as a downloadable Markdown string."""
    history = get_chat_history(api_key, thread_id)
    lines = [f"# Chat Export ({thread_id})\n"]
    for msg in history:
        speaker = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"### {speaker}\n{msg['content']}\n")
    return "\n".join(lines)


def stream_message(
    api_key: str,
    thread_id: str,
    user_text: str,
    model_name: str = "gemini-3.5-flash-lite",
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
