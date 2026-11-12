from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

from tavily import TavilyClient
tavily = TavilyClient()

@tool
def search(query: str) -> dict:
    """
    Tool that searches the internet
    Args:
        query (str): The query to search for
    Returns:
        The search result
    """
    print(f"Searching for: {query}")
    return tavily.search(query=query)

llm = ChatAnthropic(model="claude-haiku-4-5")
# llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, max_tokens=500)
tools = [search]
agent = create_agent(model=llm, tools=tools)

result = agent.invoke({
    "messages": [HumanMessage(content="Search for 3 job postings for an AI engineer in LangChain in the Bary Area on LinkedIn and list all of their details")]
})
print(result)

