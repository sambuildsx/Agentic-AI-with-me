from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode

import os
from dotenv import load_dotenv

load_dotenv()

mem = MemorySaver()

#creating class
class State (TypedDict):
    messages :  Annotated [list, add_messages]

#-----------config model-------------
llm=ChatGroq(model="llama-3.3-70b-versatile")

websearch_tool = TavilySearch(max_result=2)

@tool
def multiply(a:int , b:int)->int:
    """Divide a and b

    Args:
        a : first integer
        b : second integer

    Returns:
        int : output integer
    
    """
    return a/b

tools=[websearch_tool,multiply]

llm_with_tools= llm.bind_tools(tools)

def llm_memo (state:State):
    return {'messages':[llm.invoke(state['messages'])]}

graph_builder= StateGraph(State)

#------------adding node----------
graph_builder.add_node("llm_memo", llm_memo)
graph_builder.add_node("tools", ToolNode(tools))

#------------adding edges-------
graph_builder.add_edge(START, "llm_memo")
graph_builder.add_edge("llm_memo", END)

#------------compiler-----------
agent =  graph_builder.compile(checkpointer=mem)

config_mem = {'configurable':{'thread_id':'1'}}

res_ans= agent.invoke({'messages':"Hi, my name is Samriddhi"}, config = config_mem)

print(res_ans['messages'][-1].content)

res_qsn = agent.invoke({'messages':'do you remember my name?'}, config = config_mem)

print("\n \n")

print(res_qsn)

print("\n \n")

print(res_qsn['messages'][-1].content)