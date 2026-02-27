# 🤖 AI Research Agent

An autonomous AI agent that searches the web in real-time and answers questions using the **ReAct (Reasoning + Acting)** framework.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green)
![Groq](https://img.shields.io/badge/Groq-LLaMA3.3-orange)

## 🚀 What It Does
- Searches the web for real-time information
- Looks up Wikipedia for background knowledge
- Solves math problems with a calculator
- Remembers your conversation context

## 🧠 How It Works
Uses the **ReAct** framework — the agent thinks, picks a tool, uses it, reads the result, and answers:
```
User Question → Agent Thinks → Picks Tool → Gets Result → Answers
```

## ⚙️ Tech Stack
| Component | Technology |
|-----------|------------|
| Agent Framework | LangChain ReAct |
| LLM | Groq + LLaMA 3.3 70B |
| Web Search | Tavily API |
| Knowledge | Wikipedia |
| Math | Python Calculator |
| UI | Streamlit |

## 🏃 Quick Start
```bash
git clone https://github.com/keerthi13k/ai-search-agent
cd ai-search-agent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key" > .env
echo "TAVILY_API_KEY=your_key" >> .env
streamlit run app.py
```

## 🔑 Free API Keys
- Groq: https://console.groq.com
- Tavily: https://tavily.com
