from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_core.agents import create_react_agent
from langchain.agents import AgentExecutor
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
        Tool(
            name="Web Search",
            func=search.run,
            description="Use this to search the internet for current news, recent events, prices, or anything requiring up-to-date information. Input should be a search query string."
        ),
        Tool(
            name="Wikipedia",
            func=wiki.run,
            description="Use this for background knowledge, definitions, history, science, or any factual topic. Input should be a topic or concept."
        ),
        Tool(
            name="Calculator",
            func=calculator,
            description="Use this for math calculations. Input should be a Python math expression like '2 ** 10' or 'sqrt(144)'."
        ),
    ]

    react_prompt = PromptTemplate.from_template("""You are a helpful AI assistant with access to tools.

You have access to the following tools:
{tools}

Use this format EXACTLY:
Thought: Do I need to use a tool? Yes or No
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: Do I need to use a tool? No
Final Answer: your final answer here

Previous conversation:
{chat_history}

Question: {input}
{agent_scratchpad}""")

    agent = create_react_agent(llm, tools, react_prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )

    return agent_executor


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