# day2_langgraph.py
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1. STATE (Now with more fields to track the loop) ---
class AgentState(TypedDict):
    task: str
    draft: str
    critique: str
    revision_count: int

# --- 2. CONNECT TO GROQ ---
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-20b",
    temperature=0.3
)
print("☁️  Connected to Groq")

# --- 3. NODES ---
def writer_node(state: AgentState):
    """Writes or revises the draft based on feedback."""
    if state.get("critique"):
        # This is a revision — use the critic's feedback
        prompt = f"""
Task: {state['task']}

Previous draft:
{state['draft']}

Critique:
{state['critique']}

Please revise the draft based on the critique above. Return ONLY the revised draft.
        """
    else:
        # First draft
        prompt = f"Write a first draft for this task: {state['task']}. Return ONLY the draft."
    
    response = llm.invoke(prompt)
    print(f"✍️  Writer produced draft (revision #{state.get('revision_count', 0) + 1})")
    return {"draft": response.content}


def critic_node(state: AgentState):
    """Judges the draft quality."""
    prompt = f"""
You are a strict critic. Review this draft against the task.

Task: {state['task']}
Draft: {state['draft']}

If the draft is excellent and fulfills the task, respond with exactly: APPROVE
If it needs improvement, respond with exactly: REVISE followed by specific feedback on the next line.

Your response:
    """
    response = llm.invoke(prompt)
    critique = response.content.strip()
    
    # Extract the decision (first word)
    decision = critique.split()[0].upper() if critique else "REVISE"
    print(f"🔍 Critic says: {decision}")
    
    return {
        "critique": critique,
        "revision_count": state.get("revision_count", 0) + 1
    }


# --- 4. ROUTER (The Conditional Edge) ---
def should_continue(state: AgentState) -> Literal["writer", "end"]:
    """
    Decides: loop back to writer, or end the graph?
    """
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)
    
    # Safety: Max 3 revisions to avoid infinite loops
    if revision_count >= 3:
        print("⚠️  Max revisions reached. Forcing END.")
        return "end"
    
    if critique.upper().startswith("APPROVE"):
        print("✅ Approved! Ending loop.")
        return "end"
    else:
        print("🔁 Needs work. Looping back to writer.")
        return "writer"


# --- 5. BUILD THE GRAPH ---
builder = StateGraph(AgentState)

builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)

builder.set_entry_point("writer")
builder.add_edge("writer", "critic")
builder.add_conditional_edges(
    "critic",           # From this node...
    should_continue,    # ...run this function to decide
    {
        "writer": "writer",  # If it returns "writer", go back to writer node
        "end": END           # If it returns "end", terminate
    }
)

graph = builder.compile()

# --- 6. RUN IT ---
if __name__ == "__main__":
    initial_state = {
        "task": "Write a 2-sentence apology email to a customer whose order was delayed by 3 days. Be empathetic but professional.",
        "draft": "",
        "critique": "",
        "revision_count": 0
    }
    
    print("🚀 Running Self-Correcting Agent...\n")
    print("=" * 60)
    
    final_state = graph.invoke(initial_state)
    
    print("=" * 60)
    print("\n📋 FINAL DRAFT:")
    print(final_state["draft"])
    print(f"\n🔄 Total revisions: {final_state['revision_count']}")