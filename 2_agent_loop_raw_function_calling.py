from dotenv import load_dotenv
load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
# MODEL = "qwen3.5:9b"
MODEL = "gemma4:12b"
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
    Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold.
    """
    print(f"Executing apply_discount with price={price} and discount_tier={discount_tier}")
    discounts={"bronze": 5,
               "silver": 12,
               "gold": 23}
    discount = discounts.get(discount_tier, 0)
    return round(price*(1-discount/100), 2)


# ollama-specific tool JSON schema
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up a price of a product in the catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product, for example: laptop, headphones, keyboard"
                    }
                },
                "required": ["product"]
            }
        }

    },
    {
            "type": "function",
            "function": {
                "name": "apply_discount",
                "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "price": {
                            "type": "number",
                            "description": "The original price"
                        },
                        "discount_tier": {
                            "type": "string",
                            "description": "The discount tier: 'bronze', 'silver', or 'gold'"
                        }
                    },
                    "required": ["price", "discount"]
                }
            }
    
        }
]


# LangChain implementation comparison: each tool def above has @tool decorator, then can use the code below
# tools = [get_product_price, apply_discount]
# tools_dict = {tool.name: tool for tool in tools}


# ----- agent  -----

@traceable(name="Ollama Chat", run_type="llm") # handles tool binding
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):

    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount
    }
    
    messages = [
        {"role": "system",
         "content": (
             "You are a helpful shopping assistant."
             "You have access to a product catalog tool and a discount tool.\n\n"
             "STRICT RULES - you must ofllow these exactly:\n"
             "1. NEVER guess or assume any product price. You MUST call get_product_price first to the get the real price.\n"
             "2. Only call apply_discount AFTER you have received a price from get_product_price."
             "Pass the exact price returned by get_product_price - do NOT modify this number or make up a different number.\n"
             "3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.\n"
             "4. If the user does not specify a discount tier, ask them which tier to use - do NOT assume one."
         )},
        {"role": "user",
         "content": question}
            ]
    
    # --- agent loop ---

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n--- Iteration {iteration} ---")
        response = ollama_chat_traced(messages) # note: this is a fresh call every time
        print(f"\n\n----- Thinking -----\n{response.message.thinking}\n--------------------\n")
        ai_message = response.message
        tool_calls = ai_message.tool_calls
        if not tool_calls:
            return ai_message.content
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            # note: no tool call id in the example in the book
            print(f"[Tool selected] {tool_name} with ags: {tool_args}")
            tool_to_use = tools_dict.get(tool_name)
            if not tool_to_use:
                raise ValueError(f"Tool '{tool_name}' not found")
            observation = tool_to_use(**tool_args)
            print(f"[Tool Result] {observation}")
            messages.append(ai_message)
            messages.append({
                "role": "tool",
                "content": str(observation)
            })
    return None


if __name__ == "__main__":
    print("LangChain agent with tool calling\n")
    result = run_agent("What is the price of the laptop after applying the gold discount? There is only one laptop model.")
    print(result)
    print("done")