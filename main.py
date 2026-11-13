from dotenv import load_dotenv
load_dotenv()

from typing import List
from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

# from langchain_tavily import TavilySearch
from tavily import TavilyClient
tavily = TavilyClient()


# ------ tools -----
# search_tool = TavilySearch()
# tools = [search_tool]

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
    return tavily.search(query=query, max_results=5)
tools = [search]

# ------ output structure -----

class Source(BaseModel):
    """
    Schema for a source used by the agent
    """
    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """
    Schema for hte agent response with answer and sources
    """
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, 
                                  description="The list of sources used to generate the answer")
    


# ------ agent -----

llm = ChatAnthropic(model="claude-haiku-4-5")
# llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, max_tokens=500)

agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

result = agent.invoke({
    "messages": [HumanMessage(content="Search for 3 job postings for an AI engineer in LangChain in the Bary Area on LinkedIn and list all of their details")]
})
print(result)

