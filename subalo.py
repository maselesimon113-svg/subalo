"""
Simple Streamlit Chatbot using Ollama (qwen2.5:0.5b model)

Prerequisites:
1. Install Ollama: https://ollama.com/download
2. Pull the model:  ollama pull qwen2.5:0.5b
3. Make sure Ollama is running (it usually runs automatically as a background
   service after installation, or start it manually with `ollama serve`)
4. Install Python dependencies:
       pip install streamlit ollama
5. Run the app:
       streamlit run chatbot_app.py
"""

import streamlit as st
import ollama

# ------------------------------------------------------------------
# App configuration
# ------------------------------------------------------------------
MODEL_NAME = "qwen2.5:0.5b"

st.set_page_config(
    page_title="SUBALO AI",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 SUBALO AI")
st.caption(f"Powered by Ollama running `{MODEL_NAME}` locally")

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")

    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful, friendly assistant. Answer clearly and concisely.",
        height=100,
    )

    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)

    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# ------------------------------------------------------------------
# Session state: chat history
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ...}

# ------------------------------------------------------------------
# Display existing chat history
# ------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------------------------------------------------------
# Helper: stream a response from Ollama
# ------------------------------------------------------------------
def stream_ollama_response(messages, temperature):
    """
    Streams a chat completion from a local Ollama model.
    `messages` should be a list of {"role": "user"/"assistant"/"system", "content": ...}
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    stream = ollama.chat(
        model=MODEL_NAME,
        messages=full_messages,
        stream=True,
        options={"temperature": temperature},
    )

    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content

# ------------------------------------------------------------------
# Chat input
# ------------------------------------------------------------------
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Show and store the user's message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate and stream the assistant's reply
    with st.chat_message("assistant"):
        try:
            response_text = st.write_stream(
                stream_ollama_response(st.session_state.messages, temperature)
            )
        except Exception as e:
            response_text = (
                "⚠️ Could not reach the Ollama server. "
                "Make sure Ollama is installed and running, and that you've "
                f"pulled the model with `ollama pull {MODEL_NAME}`.\n\n"
                f"Error details: {e}"
            )
            st.error(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})