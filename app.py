import streamlit as st
import langchain
import langchain_groq
import langchain_community
import langchain_core

st.write("langchain version:", langchain.__version__)
st.write("langchain_groq version:", langchain_groq.__version__)
st.write("langchain_community version:", langchain_community.__version__)
st.write("langchain_core version:", langchain_core.__version__)

# Test the exact import
try:
    from langchain.agents import AgentExecutor, create_react_agent
    st.success("✅ langchain.agents works!")
except Exception as e:
    st.error(f"❌ langchain.agents failed: {e}")

try:
    from langchain_core.tools import Tool
    st.success("✅ langchain_core.tools works!")
except Exception as e:
    st.error(f"❌ langchain_core.tools failed: {e}")
