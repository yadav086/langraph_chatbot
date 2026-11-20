# Gemini AI Workspace (LangGraph + Streamlit + SQLite)

A production-ready, stateful conversational AI web application built with **LangGraph**, **Google Gemini**, **SQLite Checkpointing**, and **Streamlit**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quickstart Installation](#quickstart-installation)
- [Usage Guide](#usage-guide)
- [Configuration & Customization](#configuration--customization)
- [Database Persistence](#database-persistence)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project separates backend graph orchestration (`chatbot.py`) from the interactive frontend UI (`chatbot_app.py`). Every conversation is persisted to a local SQLite database (`checkpoints.sqlite`) via LangGraph's `SqliteSaver`, enabling multi-session memory, thread switching, and seamless resumption across app restarts.

---

## Key Features

- **Real-Time Token Streaming:** Streams responses word-by-word using LangGraph `stream_mode="messages"` and Streamlit's `st.write_stream()`.
- **Persistent Multi-Thread Memory:** Stores full conversation state in `checkpoints.sqlite` so history survives browser refreshes and server restarts.
- **API Key Validation & Status Badge:** Verifies the user's Google Gemini API key before unlocking the workspace and displays a live `Connected` / `Not Connected` pill indicator.
- **Conversation Thread Manager:** Create new threads, switch between saved past conversations titled by their first user prompt, or delete old threads from SQLite.
- **Dynamic Model & Persona Controls:** Adjust the Gemini model (`gemini-3.5-flash-lite`, `gemini-2.5-flash`, `gemini-2.5-pro`), temperature slider (`0.0` to `1.0`), and custom system instructions in real time.
- **One-Click Markdown Export:** Download any conversation thread as a clean `.md` transcript.
- **Modern Material UI:** Centered chat layout with scalable Google Material Design avatars (`:material/person:` and `:material/smart_toy:`) and empty-state prompt starters.

---

## Architecture

```text
+-------------------------------------------------------------------+
|                        Streamlit UI (app.py)                      |
|  - Sidebar: API Key Auth, Thread Switcher, Model & Persona Config |
|  - Workspace: Starter Prompts, History Viewer, Live Stream Output |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                     LangGraph Backend (backend.py)                |
|  - StateGraph(ChatState) -> [START] -> [chat_node] -> [END]       |
|  - ChatGoogleGenerativeAI (Streaming & Invocation)                |
+---------------------------------+---------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                   SQLite Checkpointer (SqliteSaver)               |
|  - Local File: checkpoints.sqlite                                 |
|  - Tables: checkpoints, checkpoint_blobs, checkpoint_writes       |
+-------------------------------------------------------------------+
```

---

## Project Structure

```text
gemini-langgraph-assistant/
├── chatbot_app.py                # Streamlit frontend application
├── chatbot.py            # LangGraph state graph, streaming & SQLite logic
├── requirements.txt      # Python package dependencies
├── README.md             # Project documentation
└── checkpoints.sqlite    # Auto-generated SQLite database (created on first run)
```

---

## Prerequisites

- **Python 3.10+** installed on your system.
- A valid **Google Gemini API Key** from [Google AI Studio](https://aistudio.google.com/).

---

## Quickstart Installation

### 1. Create and Activate a Virtual Environment

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Launch the Application

```bash
streamlit run app.py
```

The application will open automatically in your browser at `http://localhost:8501`.

---

## Usage Guide

1. **Authenticate:** Paste your Google Gemini API key into the **API Configuration** box in the left sidebar and click **Connect**.
2. **Verify Status:** Confirm the status badge turns green (`● Connected`).
3. **Configure Model (Optional):** Choose your preferred Gemini model, adjust temperature, or enter custom **System Instructions** in the sidebar.
4. **Start Chatting:** Click a starter card or type a prompt into the chat input box at the bottom.
5. **Manage Threads:**
   - Click **New Conversation** to start an isolated thread.
   - Use the **Conversations** dropdown to revisit past chats stored in SQLite.
   - Click the **Trash icon** next to the dropdown to permanently delete the active thread.
   - Click **Export** in the top-right corner to download the thread as a `.md` file.

---

## Configuration & Customization

| Parameter | Location | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_name` | `chatbot_app.py` / `backend.py` | `gemini-3.5-flash-lite` | Gemini model used for generation |
| `temperature` | `chatbot_app.py` | `0.2` | Controls randomness (`0.0` = deterministic, `1.0` = creative) |
| `system_prompt` | `chatbot_app.py` | `""` | Prepended as a `SystemMessage` on every turn |
| `DB_PATH` | `chatbot.py` | `checkpoints.sqlite` | Local SQLite file storing LangGraph checkpoints |

---

## Database Persistence

- LangGraph's `SqliteSaver` automatically initializes `checkpoints.sqlite` on startup.
- Because Streamlit reruns scripts across multiple worker threads, `sqlite3.connect("checkpoints.sqlite", check_same_thread=False)` is enabled in `backend.py`.
- To reset all stored chat history completely, stop the Streamlit server and delete `checkpoints.sqlite`.

---

## Troubleshooting

- **`API_KEY_INVALID` or Connection Failed:** Ensure your API key has no leading/trailing spaces and that the Gemini API is enabled in your Google AI Studio / Cloud project.
- **`sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that same thread`:** Verify `check_same_thread=False` is present in `sqlite3.connect(...)` inside `backend.py`.
- **Module Import Errors:** Make sure your virtual environment is activated before running `streamlit run app.py`.
