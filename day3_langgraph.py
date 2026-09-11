# day3_langgraph.py
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # 👈 NEW
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1. STATE ---
class AgentState(TypedDict):
    task: str
    draft: str
    critique: str
    revision_count: int

# --- 2. LLM ---
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-20b",
    temperature=0.3
)
print("☁️  Connected to Groq")

# --- 3. NODES ---
def writer_node(state: AgentState):
    if state.get("critique"):
        prompt = f"Task: {state['task']}\n\nPrevious draft:\n{state['draft']}\n\nCritique:\n{state['critique']}\n\nRevise the draft based on the critique. Return ONLY the revised draft."
    else:
        prompt = f"Write a first draft for this task: {state['task']}. Return ONLY the draft."
    
    response = llm.invoke(prompt)
    print(f"✍️  Writer produced draft (revision #{state.get('revision_count', 0) + 1})")
    return {"draft": response.content}

def critic_node(state: AgentState):
    prompt = f"""You are a strict critic. Review this draft against the task.

Task: {state['task']}
Draft: {state['draft']}

If excellent, respond with exactly: APPROVE
If needs improvement, respond with exactly: REVISE followed by feedback.

Your response:"""
    
    response = llm.invoke(prompt)
    critique = response.content.strip()
    decision = critique.split()[0].upper() if critique else "REVISE"
    print(f"🔍 Critic says: {decision}")
    
    return {
        "critique": critique,
        "revision_count": state.get("revision_count", 0) + 1
    }

# --- 4. ROUTER ---
def should_continue(state: AgentState) -> Literal["writer", "end"]:
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)
    
    if revision_count >= 3:
        print("⚠️  Max revisions reached. Forcing END.")
        return "end"
    if critique.upper().startswith("APPROVE"):
        print("✅ Approved! Ending loop.")
        return "end"
    print("🔁 Needs work. Looping back to writer.")
    return "writer"

# --- 5. BUILD GRAPH ---
builder = StateGraph(AgentState)
builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)
builder.set_entry_point("writer")
builder.add_edge("writer", "critic")
builder.add_conditional_edges(
    "critic",
    should_continue,
    {"writer": "writer", "end": END}
)

# --- 6. 🧠 ATTACH MEMORY CHECKPOINTER ---
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)  # 👈 Key line

# --- 7. RUN MULTIPLE SESSIONS WITH DIFFERENT THREADS ---
if __name__ == "__main__":
    # Each "thread_id" is like a separate conversation/customer
    config_thread_1 = {"configurable": {"thread_id": "customer_rahul"}}
    config_thread_2 = {"configurable": {"thread_id": "customer_priya"}}
    
    print("=" * 60)
    print("🧵 SESSION 1: Rahul's conversation")
    print("=" * 60)
    
    result_1 = graph.invoke(
        {
            "task": "Write a 2-sentence apology email for a delayed order",
            "draft": "",
            "critique": "",
            "revision_count": 0
        },
        config=config_thread_1
    )
    print(f"\n✅ Rahul's final draft:\n{result_1['draft']}\n")
    
    print("=" * 60)
    print("🧵 SESSION 2: Priya's conversation (different thread)")
    print("=" * 60)
    
    result_2 = graph.invoke(
        {
            "task": "Write a 2-sentence welcome email for a new customer",
            "draft": "",
            "critique": "",
            "revision_count": 0
        },
        config=config_thread_2
    )
    print(f"\n✅ Priya's final draft:\n{result_2['draft']}\n")
    
    # --- 8. 🧠 DEMONSTRATE MEMORY PERSISTENCE ---
    print("=" * 60)
    print("🔍 CHECKING MEMORY: Retrieving Rahul's state from checkpoint")
    print("=" * 60)
    
    # Retrieve Rahul's state using the same thread_id
    rahul_state = graph.get_state(config_thread_1)
    print(f"Task: {rahul_state.values['task']}")
    print(f"Revisions: {rahul_state.values['revision_count']}")
    print("\n🎉 Memory persists across sessions! Each thread has its own history.")