### Build a basic chatbot
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START , END
from langgraph.graph.message import add_messages
class State(TypedDict):
    messages : Annotated[list, add_messages]

graph_builder=StateGraph(State)   
import os
from dotenv import load_dotenv

# Load .env from the basicChatbot directory
load_dotenv("c:\\Users\\Sam\\Desktop\\Agentic AI\\LangGraph Agentic\\basicChatbot\\.env")
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model

# Available Groq models: llama-3.1-8b-instant, gemma-2-9b-it, etc.
# Using smaller, faster, and currently available models

# Method 1: Using init_chat_model
# llm = init_chat_model("groq:llama-3.1-8b-instant")

# Method 2: Direct ChatGroq initialization  
# llm = ChatGroq(model="llama-3.1-8b-instant")

# Method 3: With explicit parameters
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7
)
llm

##Node functionality
def Chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder=StateGraph(State)  
##adding Node to the graph
graph_builder.add_node("llmChatbot", Chatbot)
##adding edge to the graph
graph_builder.add_edge(START, "llmChatbot")
graph_builder.add_edge("llmChatbot", END)

##complile graph 
graph = graph_builder.compile()

##visualise the graph
from IPython.display import display, Image

# Get PNG binary data and display as image
png_data = graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)

response = graph.invoke({"messages": ["What is the capital of France?"]})
response
response["messages"]


for event in graph.stream({"messages": ["Hi, How are you?"]}):
    for value in event.values():
        print(value["messages"][-1].content)


###tool in agent
from langchain_tavily import TavilySearch

tool = TavilySearch(max_results=3)

result = tool.invoke({
    "query": "What is LangChain?"
})

print(result)

###custom function in agent
def multiply (a: int, b: int) -> int:
    '''
    Multiplies a and b 
    args:
    a(int): The first number to multiply.
    b(int): The second number to multiply.

    result:
    int: output int

    '''

tools=[tool, multiply]
llm_with_tools=llm.bind_tools(tools)