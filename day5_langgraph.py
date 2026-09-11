# day5_langgraph.py
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1. STATE ---
class ApprovalState(TypedDict):
    task: str
    draft: str
    approval_status: str   # "pending", "approved", "rejected"
    feedback: str
    final_output: str

# --- 2. LLM ---
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-20b",
    temperature=0.3
)
print("☁️  Connected to Groq\n")

# --- 3. NODES ---
def drafter_node(state: ApprovalState):
    """Drafts content (or revises based on feedback)."""
    if state.get("feedback"):
        print("✍️  Revising draft based on feedback...")
        prompt = f"""Task: {state['task']}
Previous draft:
{state['draft']}
Feedback from human:
{state['feedback']}
Revise the draft. Return ONLY the revised draft."""
    else:
        print("✍️  Drafting initial version...")
        prompt = f"Write a first draft for: {state['task']}. Return ONLY the draft."
    
    response = llm.invoke(prompt)
    return {"draft": response.content, "approval_status": "pending"}

def publisher_node(state: ApprovalState):
    """Publishes ONLY if approved."""
    print("📤 Publishing final output...")
    return {"final_output": f"[PUBLISHED]\n\n{state['draft']}"}

# --- 4. ROUTER ---
def check_approval(state: ApprovalState) -> Literal["drafter", "publisher", "end"]:
    status = state.get("approval_status", "pending")
    if status == "approved":
        print("✅ Human approved. Publishing.")
        return "publisher"
    elif status == "rejected":
        print("❌ Human rejected. Redrafting.")
        return "drafter"
    else:
        print("⏸️  Awaiting human approval...")
        return "end"  # Or wait — we'll handle this in the run section

# --- 5. BUILD GRAPH ---
builder = StateGraph(ApprovalState)
builder.add_node("drafter", drafter_node)
builder.add_node("publisher", publisher_node)
builder.set_entry_point("drafter")
builder.add_conditional_edges(
    "drafter",
    check_approval,
    {"drafter": "drafter", "publisher": "publisher", "end": END}
)
builder.add_edge("publisher", END)

# --- 6. COMPILE WITH MEMORY (required for interrupts) ---
memory = MemorySaver()
graph = builder.compile(checkpointer=memory, interrupt_before=["publisher"])  # 👈 KEY LINE

# --- 7. RUN WITH HUMAN APPROVAL LOOP ---
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "approval_session_1"}}
    
    initial_state = {
        "task": "Write a 2-line rejection email to a vendor whose proposal was over budget",
        "draft": "",
        "approval_status": "pending",
        "feedback": "",
        "final_output": ""
    }
    
    print("=" * 60)
    print("🚀 HUMAN-IN-THE-LOOP WORKFLOW")
    print("=" * 60 + "\n")
    
    # Step 1: Agent generates first draft
    graph.invoke(initial_state, config=config)
    
    state = graph.get_state(config)
    print("\n📋 DRAFT FOR REVIEW:")
    print("-" * 60)
    print(state.values["draft"])
    print("-" * 60)
    
    # Step 2: Human reviews and decides
    print("\n🤔 HUMAN REVIEW:")
    print("  [1] Approve")
    print("  [2] Reject with feedback")
    
    # Simulate a human decision (in a real app, this comes from a UI)
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        graph.update_state(config, {"approval_status": "approved"})
        print("\n✅ Approved! Resuming graph...")
        final = graph.invoke(None, config=config)
        print("\n" + "=" * 60)
        print("✅ FINAL OUTPUT:")
        print("=" * 60)
        print(final["final_output"])
    else:
        feedback = input("\nEnter your feedback: ").strip()
        graph.update_state(config, {"approval_status": "rejected", "feedback": feedback})
        print("\n🔁 Rejected. Resuming graph to redraft...")
        final = graph.invoke(None, config=config)
        print("\n" + "=" * 60)
        print("📋 REVISED DRAFT:")
        print("=" * 60)
        print(final["draft"])