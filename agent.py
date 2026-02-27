from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import math

load_dotenv()

def create_agent():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY")
    )

    _search = TavilySearch(max_results=5, api_key=os.environ.get("TAVILY_API_KEY"))
    _wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))

    def calculator(expression: str) -> str:
        try:
            allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            allowed["__builtins__"] = {}
            result = eval(expression, allowed)
            return str(result)
        except Exception as e:
            return f"Calculator error: {str(e)}"

    tools = [
        Tool(name="WebSearch",
             func=_search.run,
             description="Search internet for current news and events. Input: search query."),
        Tool(name="Wikipedia",
             func=_wiki.run,
             description="Look up background knowledge and explanations. Input: topic name."),
        Tool(name="Calculator",
             func=calculator,
             description="Math calculations. Input: Python expression like 2**10 or sqrt(144)."),
    ]

    prompt = PromptTemplate.from_template("""You are a helpful AI assistant.

TOOLS AVAILABLE:
{tools}

Tool names: {tool_names}

STRICT FORMAT - follow this exactly:
Thought: Do I need to use a tool? Yes
Action: WebSearch
Action Input: search query here
Observation: result here
Thought: Do I need to use a tool? No
Final Answer: answer here

If no tool needed:
Thought: Do I need to use a tool? No
Final Answer: answer here

Chat History:
{chat_history}

Question: {input}
{agent_scratchpad}""")

    agent = create_react_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )

    return executor


def ask_agent(agent_executor, question, chat_history):
    try:
        history_str = ""
        for msg in chat_history:
            if isinstance(msg, HumanMessage):
                history_str += f"Human: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                history_str += f"Assistant: {msg.content}\n"

        response = agent_executor.invoke({
            "input": question,
            "chat_history": history_str
        })
        return response["output"]
    except Exception as e:
        return f"Error: {str(e)}"
