from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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

    search = TavilySearch(
        max_results=5,
        api_key=os.environ.get("TAVILY_API_KEY")
    )

    wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=2))

    def calculator(expression: str) -> str:
        try:
            allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
            allowed["__builtins__"] = {}
            result = eval(expression, allowed)
            return str(result)
        except Exception as e:
            return f"Calculator error: {str(e)}"

    tools = [
        Tool(name="WebSearch", func=search.run,
             description="Search internet for current news, events, prices. Input: search query string."),
        Tool(name="Wikipedia", func=wiki.run,
             description="Get background knowledge, definitions, history, science facts. Input: topic or concept."),
        Tool(name="Calculator", func=calculator,
             description="Math calculations. Input: Python math expression like '2**10' or 'sqrt(144)'."),
    ]

    llm_with_tools = llm.bind_tools(tools)
    return llm_with_tools, tools


def ask_agent(agent_data, question, chat_history):
    try:
        llm_with_tools, tools = agent_data
        tool_map = {t.name: t for t in tools}

        # Build messages
        messages = [SystemMessage(content="""You are a helpful AI research assistant. 
You have access to Web Search, Wikipedia, and Calculator tools.
Always use tools when you need current information or calculations.
Give comprehensive, helpful answers.""")]

        # Add chat history
        for msg in chat_history:
            messages.append(msg)

        # Add current question
        messages.append(HumanMessage(content=question))

        # Agentic loop
        for _ in range(5):
            response = llm_with_tools.invoke(messages)
            messages.append(response)

            # If no tool calls, we have final answer
            if not response.tool_calls:
                return response.content

            # Execute tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
               # Get whatever argument Groq sends, regardless of key name
                args = tool_call["args"]
                if isinstance(args, dict):
                    tool_input = next(iter(args.values()), "")
                else:
                    tool_input = str(args)

                if tool_name in tool_map:
                    result = tool_map[tool_name].func(tool_input)
                else:
                    result = f"Tool {tool_name} not found"

                from langchain_core.messages import ToolMessage
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))

        return messages[-1].content

    except Exception as e:
        return f"Error: {str(e)}"