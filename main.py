from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 5
MODEL = "qwen3.5:9b"
MODEL = "gemma4:12b"

# ----- tools -----

@tool
def get_product_price(product: str) -> float:
    """
    Look up the price of a product in the catalog.
    """
    print(f" >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99,
              "headphones": 149.95,
              "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold.
    """
    print(f"Executing apply_discount with price={price} and discount_tier={discount_tier}")
    discounts={"bronze": 5,
               "silver": 12,
               "gold": 23}
    discount = discounts.get(discount_tier, 0)
    return round(price*(1-discount/100), 2)


tools = [get_product_price, apply_discount]
tools_dict = {tool.name: tool for tool in tools}


# ----- agent  -----

llm = init_chat_model(MODEL, model_provider="ollama", reasoning=True) # use this one
# llm = init_chat_model("ollama:qwen3.5:9b", reasoning=True)
# llm = init_chat_model("ollama:gemma4:12b", reasoning=True)
# llm = init_chat_model("ollama:mistral:latest") # does not support reasoning
# llm = init_chat_model("ollama:ministral-3:14b") # does not support reasoning
# llm = init_chat_model("anthropic:claude-haiku-4-5") # haiku 4.5 does not support reasoning via LangChain init_chat_model; use ChatAnthropic 

llm_with_tools = llm.bind_tools(tools)


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):

    print(f"Model: {MODEL}")
    messages = [
            SystemMessage(content="""You are a helpful shopping assistant. 
            You have access to a product catalog tool and a discount tool.
            
            Rules:
            - Never guess or assume any product price.
            - Always call the get_product_price tool to retrieve the real price.
            - Only call apply_discount after receiving the price from get_product_price. Pass the exact price returned by the tool, never a made-up number.
            - Never calculate discounts yourself. Always use the apply_discount tool.
            - If the user does not specify a discount tier, ask them which their they want. Do not assume one.
            """),
            HumanMessage(content=question)
            ]
    
    # --- agent loop ---
    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)
        print(f"\n\n----- Thinking -----\n{ai_message.additional_kwargs.get("reasoning_content", "(no reasoning returned)")}\n--------------------\n")
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            break
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")
            print(f"[Tool selected] {tool_name} with ags: {tool_args}")
            tool_fn = tools_dict.get(tool_name)
            if not tool_fn:
                raise ValueError(f"Tool '{tool_name}' not found")
            observation = tool_fn.invoke(tool_args) # this currently causes a print statement
            print(f"[Tool Result] {observation}")
            messages.append(ai_message)
            messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
    print("agent_loop exited")
    return (ai_message.text)


if __name__ == "__main__":
    print("LangChain agent with tool calling\n")
    result = run_agent("What is the price of the laptop after applying the gold discount?")
    print(result)
    # print("done")