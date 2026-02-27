from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_tavily import TavilySearch
from langchain.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain import hub
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
import os
import math

load_dotenv()

def create_agent():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    search = TavilySearch(max_results=5, api_key=os.getenv("TAVILY_API_KEY"))
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
            description="""Use this to search the internet for current news, 
            recent events, prices, or anything requiring up-to-date information.
            Input should be a search query string."""
        ),
        Tool(
            name="Wikipedia",
            func=wiki.run,
            description="""Use this for background knowledge, definitions, history, 
            science, or any factual topic that doesn't need to be current. 
            Input should be a topic or concept."""
        ),
        Tool(
            name="Calculator",
            func=calculator,
            description="""Use this for any math calculations. 
            Input should be a valid Python math expression like '2 ** 10' or 'sqrt(144)'."""
        ),
    ]

    prompt = hub.pull("hwchase17/react-chat")  # react-chat supports memory natively

    agent = create_react_agent(llm, tools, prompt)

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
        # Pass chat history directly each time
        response = agent_executor.invoke({
            "input": question,
            "chat_history": chat_history
        })
        return response["output"]
    except Exception as e:
        return f"Error: {str(e)}"