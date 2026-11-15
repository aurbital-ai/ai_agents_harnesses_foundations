from dotenv import load_dotenv
load_dotenv()

import re
import inspect

import ollama
from langsmith import traceable

MAX_ITERATIONS = 3
# MODEL = "gemma4:12b"
MODEL = "ministral-3:14b"
# MODEL = "qwen3.5:9b"
# MODEL = "mistral:latest"


# ----- tools -----

@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """
    Look up the price of a product in the catalog.
    """
    print(f" >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99,
              "headphones": 149.95,
              "keyboard": 89.50}
    return prices.get(product, 0)

         
@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.
    """
    print(f"Executing apply_discount with price={price} and discount_tier={discount_tier}")
    discounts={"bronze": 5,
               "silver": 12,
               "gold": 23}
    discount = discounts.get(discount_tier, 0)
    return round(price*(1-discount/100), 2)

# define tools dictionary
tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount
}

def get_tool_descriptions(tools_dict):
    descriptions=[]
    for tool_name, tool_function in tools_dict.items():
        # __wrapped__ bypasses decorator wrappers (eg @traceable adds *, config=None)
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
        # print(descriptions)
    return "\n".join(descriptions)


tool_descriptions = get_tool_descriptions(tools)
tool_names = ",".join(tools.keys())


# ----- ReAct prompt -----
# See original template https://smith.langchain.com/hub/hwchase17/react?organizationId=10a67c4b-8a89-4988-aed8-66dd86c8d3d4

react_prompt = f"""
    STRICT RULES - you must follow these exactly:\n
    1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.\n
    2. Only call apply_discount AFTER you have received a price from get_product_price.
    Pass the exact price returned by get_product_price - do NOT modify this number or make up a different number.\n
    3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.\n
    4. If the user does not specify a discount tier, ask them which tier to use - do NOT assume one.\n\n
    
    Answer the following questions as best you can. You have access to the following tools: {tool_descriptions}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action, as comma separated values
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {{question}}
    Thought:
"""



# ----- agent  -----

@traceable(name="Ollama Chat", run_type="llm") # handles tool binding
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=MODEL, messages=messages, options=options)


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    print("\n\n")
    print("*"*60)
    print(f"MODEL: {MODEL}")
    print("="*60)
    print(f"Question: {question}")
    print("="*60)
    prompt = react_prompt.format(question=question)
    scratchpad = ""
    
    
    # --- agent loop ---

    for iteration in range(1, MAX_ITERATIONS+1):
        print("\n")
        print("-"*60)
        print(f"--- Iteration {iteration} ---")
        full_prompt = prompt + scratchpad
        print(f"full_prompt:\n{full_prompt}")

        # stop token prevents LLLM from generating its own Observation, allows us to inject the real result here
        response = ollama_chat_traced(
            model=MODEL,
            messages = [{"role": "user", "content": full_prompt}],
            options = {"stop": ["\nObservation"], "temperature":0}
        )
        print("-"*30)
        print(f"LLM thinking for Iteration {iteration}: {response.message.thinking}")
        print("-"*30)
        print(f"LLM output for Iteration {iteration}: {response.message.content}")
        print("-"*30)
    output = response.message.content 
    print(f"LLM Output: \n{output}")


if __name__ == "__main__":
    result = run_agent("What is the price of the laptop after applying the gold discount?")
    # print(result)
    print("done")