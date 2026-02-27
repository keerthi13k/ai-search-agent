from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from dotenv import load_dotenv
import os
import math

load_dotenv()

# Initialize search tools
_search = TavilySearch(max_results=5, api_key=os.environ.get("TAVILY_API_KEY"))
_wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))

@tool
def web_search(query: str) -> str:
    """Search the internet for current news, recent events, prices, or up-to-date information."""
    return _search.run(query)

@tool
def wikipedia(query: str) -> str:
    """Look up background knowledge, definitions, history, or science facts."""
    return _wiki.run(query)

@tool
def calculator(expression: str) -> str:
    """Perform math calculations. Input must be a Python math expression like '2**10' or 'sqrt(144)'."""
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["__builtins__"] = {}
        result = eval(expression, allowed)
        return str(result)
    except Exception as e:
        return f"Calculator error: {str(e)}"

def create_agent():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.environ.get("GROQ_API_KEY")
    )
    tools = [web_search, wikipedia, calculator]
    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools, tools

def ask_agent(agent_data, question, chat_history):
    try:
        llm_with_tools, tools = agent_data
        tool_map = {t.name: t for t in tools}

        messages = [SystemMessage(content="""You are a helpful AI research assistant.
Use web_search for current news and events.
Use wikipedia for background knowledge and explanations.
Use calculator for math problems.
Always use the appropriate tool to give accurate answers.""")]

        for msg in chat_history:
            messages.append(msg)

        messages.append(HumanMessage(content=question))

        for _ in range(5):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return response.content

            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                if tool_name in tool_map:
                    result = tool_map[tool_name].invoke(tool_args)
                else:
                    result = f"Tool {tool_name} not found"

                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))

        return messages[-1].content

    except Exception as e:
        return f"Error: {str(e)}"