# day1_langgraph.py
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1. Define the State ---
class AgentState(TypedDict):
    task: str
    plan: str
    result: str

# --- 2. Connect to Groq or Ollama ---
if os.getenv("GROQ_API_KEY"):
    llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY"),
        model="openai/gpt-oss-20b",
        temperature=0.3
    )
    print("☁️  Using Groq")
else:
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="llama3.2:latest",
        temperature=0.3
    )
    print("🔗 Using Ollama")

# --- 3. Define Nodes ---
def planner_node(state: AgentState):
    prompt = f"Create a short 3-step plan to complete this task: {state['task']}"
    response = llm.invoke(prompt)
    return {"plan": response.content}

def executor_node(state: AgentState):
    prompt = f"Follow this plan and produce the final result:\n{state['plan']}"
    response = llm.invoke(prompt)
    return {"result": response.content}

# --- 4. Build the Graph ---
builder = StateGraph(AgentState)
builder.add_node("planner", planner_node)
builder.add_node("executor", executor_node)
builder.set_entry_point("planner")
builder.add_edge("planner", "executor")
builder.add_edge("executor", END)

graph = builder.compile()

# --- 5. Run It ---
if __name__ == "__main__":
    initial_state = {"task": "Write a 3-line welcome message for a BPO agent onboarding email"}
    print("🚀 Running LangGraph Agent...\n")
    final_state = graph.invoke(initial_state)
    print("📋 PLAN:")
    print(final_state["plan"])
    print("\n✅ RESULT:")
    print(final_state["result"])