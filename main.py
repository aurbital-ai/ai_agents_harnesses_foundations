from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic



def main() -> None:
    print("Hello from ai-agents-and-harnesses-foundations!")

@tool
def search(query: str) -> str:
    """
    Tool that searches the internet
    Args:
        query (str): The query to search for
    Returns:
        The search result
    """
    print(f"Searching for: {query}")
    return "Tokyo weather is sunny"

llm = ChatAnthropic(model="claude-haiku-4-5")
# llm = ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0, max_tokens=500)
tools = [search]
agent = create_agent(model=llm, tools=tools)

result = agent.invoke({
    "messages": [HumanMessage(content="What is the weather in Tokyo?")]
})
print(result)

if __name__ == "__main__":
    main()