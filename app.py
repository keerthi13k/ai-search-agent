import streamlit as st
from agent import create_agent, ask_agent

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for cleaner look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .tool-badge {
        display: inline-block;
        background: #f0f2f6;
        border-radius: 15px;
        padding: 3px 10px;
        margin: 2px;
        font-size: 12px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h2>🤖 AI Research Agent</h2>
    <p>Powered by LangChain ReAct • Groq LLaMA 3 • Real-time Web Search</p>
</div>
""", unsafe_allow_html=True)

# Initialize agent (cached)
@st.cache_resource
def load_agent():
    return create_agent()

agent_executor = load_agent()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.header("🧠 How It Works")
    st.markdown("""
    This agent uses the **ReAct** framework:
    
    1. **Thinks** about your question
    2. **Picks the right tool**
    3. **Acts** using that tool
    4. **Observes** the result
    5. **Answers** intelligently
    """)
    
    st.header("🛠️ Available Tools")
    st.markdown("""
    <span class="tool-badge">🔍 Web Search</span>
    <span class="tool-badge">📚 Wikipedia</span>
    <span class="tool-badge">🧮 Calculator</span>
    """, unsafe_allow_html=True)

    st.header("💡 Try These")
    example_questions = [
        "What happened in AI news today?",
        "What is quantum entanglement?",
        "What is 15% of 8500?",
        "Who is the CEO of OpenAI?",
        "Explain the French Revolution",
        "What is 2 to the power of 32?",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.header("📊 Session Stats")
    st.metric("Questions Asked", len(st.session_state.messages) // 2)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle example question clicks from sidebar
if "pending_question" in st.session_state:
    prompt = st.session_state.pending_question
    del st.session_state.pending_question
else:
    prompt = None

# Chat input
user_input = st.chat_input("Ask me anything...")
if user_input:
    prompt = user_input

# Process the question
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build chat history from session for memory
    from langchain.schema import HumanMessage, AIMessage
    chat_history = []
    msgs = st.session_state.messages[:-1]  # exclude current message
    for i in range(0, len(msgs) - 1, 2):
        if i + 1 < len(msgs):
            chat_history.append(HumanMessage(content=msgs[i]["content"]))
            chat_history.append(AIMessage(content=msgs[i+1]["content"]))

    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking and searching..."):
            response = ask_agent(agent_executor, prompt, chat_history)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})