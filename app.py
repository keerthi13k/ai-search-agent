import streamlit as st
from agent import create_agent, ask_agent
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="AI Research Agent", page_icon="🤖", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h2>🤖 AI Research Agent</h2>
    <p>Powered by LangChain • Groq LLaMA 3.3 • Real-time Web Search</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_agent():
    return create_agent()

agent_data = load_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("🛠️ Available Tools")
    st.markdown("🔍 **Web Search** — current news & events\n\n📚 **Wikipedia** — background knowledge\n\n🧮 **Calculator** — math problems")
    st.header("💡 Try These")
    examples = [
        "What happened in AI news today?",
        "What is quantum entanglement?",
        "What is 15% of 8500?",
        "Who is the CEO of OpenAI?",
        "What is 2 to the power of 32?",
    ]
    for q in examples:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.metric("Questions Asked", len(st.session_state.messages) // 2)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = None
if "pending_question" in st.session_state:
    prompt = st.session_state.pending_question
    del st.session_state.pending_question

user_input = st.chat_input("Ask me anything...")
if user_input:
    prompt = user_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chat_history = []
    msgs = st.session_state.messages[:-1]
    for i in range(0, len(msgs) - 1, 2):
        if i + 1 < len(msgs):
            chat_history.append(HumanMessage(content=msgs[i]["content"]))
            chat_history.append(AIMessage(content=msgs[i+1]["content"]))

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking and searching..."):
            response = ask_agent(agent_data, prompt, chat_history)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})