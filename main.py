from typing import Annotated, Optional
from typing_extensions import TypedDict
import os
from langgraph.graph.message import AnyMessage, add_messages
from langchain_cerebras import ChatCerebras
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import tools_condition
import re
import numpy as np
import openai
from langchain_core.tools import tool
from datetime import datetime
from groq import Groq
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from system_prompt import system_prompt
import sqlite3
from deepgram import DeepgramClient, SpeakOptions

LANGSMITH_TRACING=True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY= "PLACE YOUR LANGSMITH API KEY HERE"
LANGSMITH_PROJECT="pr-overcooked-appointment-42"


os.environ["CEREBRAS_API_KEY"] = "PLACE YOUR CEREBRAS API KEY HERE"
os.environ['GROQ_API_KEY'] = "PLACE YOUR GROQ API KEY HERE"
os.environ["OPENAI_API_KEY"] = "PLACE YOUR OPENAI API KEY HERE"
os.environ["DEEPGRAM_API_KEY"] = "PLACE YOUR DEEPGRAM API KEY HERE"
#----------------------Utility Functions for app.py here-------------------------------#
client = Groq()
deepgram = DeepgramClient()
def use_groq(model_id,system_prompt,user_prompt):
    completion = client.chat.completions.create(
        model= model_id,
        messages=[{
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        }],
        temperature=0.3,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content

def transcript():
    filename = os.path.dirname(__file__) + "/audio.m4a"

    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
        file=(filename, file.read()),
        model="whisper-large-v3",
        
        )
        print(transcription.text)

def stt(text):
    TEXT = {
        "text": text
    }
    FILENAME = "audio.mp3"
    options = SpeakOptions(
        model="aura-asteria-en",
    )
    response = deepgram.speak.v("1").save(FILENAME, TEXT, options)
    #print(response.to_json(indent=4))
    

#---------------------Utility functions for app.py end here----------------------------#

#--------------------- Vector Retrievers here -----------------------------------------#
with open("revised_faq.md", "r",encoding="utf-8") as f:
    faq_text = f.read()

docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]

class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, oai_client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = oai_client

    @classmethod
    def from_docs(cls, docs, oai_client):
        embeddings = oai_client.embeddings.create(
            model="text-embedding-3-small", input=[doc["page_content"] for doc in docs]
        )
        vectors = [emb.embedding for emb in embeddings.data]
        return cls(docs, vectors, oai_client)

    def query(self, query: str, k: int = 5) -> list[dict]:
        embed = self._client.embeddings.create(
            model="text-embedding-3-small", input=[query]
        )
        # "@" is just a matrix multiplication in python
        scores = np.array(embed.data[0].embedding) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]


retriever = VectorStoreRetriever.from_docs(docs, openai.Client())

#--------------------- Tools start here -----------------------------------------------------#
@tool
def replacement_order(order_id, product) -> str:
    """Request a replacement for a specific product using order ID.
    Args:
      - order_id: str
      - product: str
    Returns:
        str : a string containing the order ID and product name
    """
    if not order_id:
        raise ValueError("Order ID not provided")
    return f"Replacement requested for Order ID: {order_id} and {product}."

@tool 
def refund_issue(user_id,order_id,value, product) -> str:
    """Issue a refund for a specific order ID and product.
    Args: 
      - order_id: str
      - value: in
      - user_id: str
      - product: str
    Returns:
        str : a string containing the order ID and product name and value"""
    if not order_id:
        raise ValueError("Order ID not provided")
    return f"Refund issued for Order ID: {order_id}, product: {product}, and value: {value}."

@tool
def faq_lookup(query: str) -> str:
    """Company internal information system for customer support agents 
    to quickly look up answers to common customer questions."""
    docs = retriever.query(query, k=2)
    return "\n\n".join([doc["page_content"] for doc in docs])

# Path to your SQLite file
db = "grocery_simulation.db"

@tool
def fetch_user_orders_by_user_id(config: RunnableConfig) -> list[dict]:
    """
    Fetch all orders placed by a user using their user id, but show product NAMES
    instead of IDs.

    Expects:
      - user_id: int
    """
    cfg = config.get("configurable", {})
    user_id = cfg.get("user_id")
    #order_date = cfg.get("order_date")

    if not user_id:
        raise ValueError("User Id not provided")

    """# Validate date format
    try:
        datetime.strptime(order_date, "%Y-%m-%d")
    except ValueError:
        raise ValueError("Date format must be YYYY-MM-DD.")"""

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # 1. Fetch raw orders
    cursor.execute("""
        SELECT 
            o.OrderID,
            o.UserID,
            o.OrderItems,
            o.OrderTotal,
            o.OrderStatus,
            o.OrderPlacedTime,
            o.OrderDeliveredTime
        FROM Orders o
        WHERE o.UserID = ?
        ORDER BY o.OrderPlacedTime ASC
    """, (user_id,))
    orders = [dict(zip([c[0] for c in cursor.description], row))
              for row in cursor.fetchall()]

    # 2. Build a ProductID → ProductName map
    cursor.execute("SELECT ProductID, ProductName FROM Products")
    product_map = {pid: name for pid, name in cursor.fetchall()}

    cursor.close()
    conn.close()

    # 3. Replace IDs with names
    for order in orders:
        id_list = order["OrderItems"].split(",")
        name_list = [product_map.get(int(pid), f"<Unknown:{pid}>") for pid in id_list]
        # You can either overwrite OrderItems or add a new key:
        order["ProductNames"] = name_list
        # If you want a single comma‑joined string:
        # order["ProductNames"] = ", ".join(name_list)
        # And optionally remove the old field:
        # del order["OrderItems"]

    return orders

@tool
def analyze_damaged_product(product_image_list, product) -> str:
    """
    Analyze a product image and determine if the product is bad, rotten, torn, or damaged. 

    Args:
      - product_image_list: list      # list of public image url uploaded by the user
      - product: str                  # the product e.g. 'apple', 'milk packet', 'bread', etc.

    Returns:
      str : a string containing the product name and its visual condition
    """

    if not product_image_list or not product:
        raise ValueError("Both 'product_image_list' and 'product' are required.")

    # Build the chat messages
    system_prompt = {
        "role": "system",
        "content": """You are a call-center executive tasked with visually inspecting customer-submitted product photos. 
                   For ex. fruits, decide if they're rotten/bad; for milk packets, if the packet is torn/leaking; 
                   similarly for any other grocery item, focus on obvious visual defects. Answer very briefly .
                   - The product should be clearly visible
                   - The photo should be clear and not blurry
                   - The image should be user captured"""
    }

    user_prompt = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Product Category: {product}\n"
                    
            },
            {
                "type": "image_url",
                "image_url": {"url": product_image_list[0]}
            }
        ]
    }

    # Call Groq chat completion
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[system_prompt, user_prompt],
        temperature=0.1,
        max_completion_tokens=512,
        top_p=1,
        stream=False,
    )

    # The API returns a streaming generator if stream=True; here we turned it off
    verdict = completion.choices[0].message.content
    return verdict

@tool
def analyze_wrong_product(product_image_list, product) -> str:
    """
    Analyze a product image and determine if it is legit and if the product is 'bad.'

    Args:
      - product_image_list: list      # list of public image url uploaded by the user
      - product: str                  # the claimed product

    Returns:
      str : a string confirming that the product is what is claimed.
    """

    if not product_image_list or not product:
        raise ValueError("Both 'product_image_list' and 'product' are required.")

    # Build the chat messages
    system_prompt = {
        "role": "system",
        "content": """You are a call-center executive tasked with visually inspecting customer-submitted product photos. 
                      Analyse the given image and let me know if the image is claimed product or not.
                      - The product should be country delight product
                      - The photo should be clear and not blurry
                      - The image should be user captured
                      Answer breifly in 1 line"""
    }

    user_prompt = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"Claimed Product: {product}\n"
                        
            },
            {
                "type": "image_url",
                "image_url": {"url": product_image_list[0]}
            }
        ]
    }

    # Call Groq chat completion
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[system_prompt, user_prompt],
        temperature=0.1,
        max_completion_tokens=512,
        top_p=1,
        stream=False,
    )

    # The API returns a streaming generator if stream=True; here we turned it off
    verdict = completion.choices[0].message.content
    return verdict


#--------------------- Tools end here -----------------------------------------------------#

#----------------------Some Helper Functions-----------------------------------------------#
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)
#----------------------Helper functions end here--------------------------------------------#

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            user_id = configuration.get("user_id", None)
            state = {**state, "user_info": user_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}


llm = ChatCerebras(model="llama-3.3-70b", temperature=0)
#llm = ChatOpenAI(model = "gpt-4o")
primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""{system_prompt}
            Current user:\n<User>\n Lakshay Dagar \n</User>\nCurrent time: {datetime.now()}.
            \n User ID: 1
            """
        ),
        ("placeholder", "{messages}"),
    ]
)

part_1_tools = [
    faq_lookup,
    fetch_user_orders_by_user_id,
    analyze_damaged_product,
    analyze_wrong_product, 
    refund_issue,
    replacement_order
]
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)


builder = StateGraph(State)


# Define nodes: these do the work
builder.add_node("assistant", Assistant(part_1_assistant_runnable))
builder.add_node("tools", create_tool_node_with_fallback(part_1_tools))
# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition,
)
builder.add_edge("tools", "assistant")

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
#memory = MemorySaver()
#part_1_graph = builder.compile(checkpointer=memory)
import shutil
import uuid

part_1_graph = builder.compile()

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        # The user_id is is used 
        # fetch information about the user
        "user_id": "2",
        # Checkpoints are accessed by thread_id
        "thread_id": thread_id,
    }
}

if __name__ == "__main__":
    def stream_graph_updates(user_input: str):
        for event in part_1_graph.stream({"messages": [{"role": "user", "content": user_input}]}):
            for value in event.values():
                print("Assistant:", value)


    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about Country Delight?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break
"""question = "What was my last order ? user_id: 1 "
_printed = set()
events = part_1_graph.stream(
    {"messages": ("user", question)}, config, stream_mode="values"
)
for event in events:
    _print_event(event, _printed)"""