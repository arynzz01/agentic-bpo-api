# day4_langgraph.py
from typing import TypedDict, Literal, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# --- 1. STATE ---
class TeamState(TypedDict):
    task: str
    next_worker: str
    research_notes: str
    draft: str
    final_output: str
    history: List[str]

# --- 2. LLM ---
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-20b",
    temperature=0.3
)
print("☁️  Connected to Groq\n")

# --- 3. WORKER NODES ---
def researcher_node(state: TeamState):
    print("🔬 Researcher working...")
    prompt = f"""You are a researcher. Provide 5 key facts about:
{state['task']}
Return ONLY a bulleted list of 5 concise facts."""
    response = llm.invoke(prompt)
    history = state.get("history", []) + ["researcher: 5 facts gathered"]
    return {"research_notes": response.content, "history": history}

def writer_node(state: TeamState):
    print("✍️  Writer working...")
    prompt = f"""You are a writer. Using these research notes, write a short article.

Topic: {state['task']}
Research notes:
{state['research_notes']}

Return ONLY the article (3 paragraphs maximum)."""
    response = llm.invoke(prompt)
    history = state.get("history", []) + ["writer: draft created"]
    return {"draft": response.content, "history": history}

def critic_node(state: TeamState):
    print("🔍 Critic working...")
    prompt = f"""You are a critic. Review this draft and produce the FINAL version.

Topic: {state['task']}
Draft:
{state['draft']}

Add any missing nuance, tighten the language, and produce the final version.
Return ONLY the final polished article."""
    response = llm.invoke(prompt)
    history = state.get("history", []) + ["critic: final version produced"]
    return {"final_output": response.content, "history": history}

# --- 4. SUPERVISOR ---
def supervisor_node(state: TeamState):
    print("👔 Supervisor thinking...")
    if not state.get("research_notes"):
        next_worker = "researcher"
    elif not state.get("draft"):
        next_worker = "writer"
    elif not state.get("final_output"):
        next_worker = "critic"
    else:
        next_worker = "FINISH"
    print(f"👔 Supervisor assigned: {next_worker}")
    return {"next_worker": next_worker}

# --- 5. ROUTER ---
def route_from_supervisor(state: TeamState) -> Literal["researcher", "writer", "critic", "end"]:
    next_worker = state.get("next_worker", "")
    if next_worker == "FINISH":
        print("👔 Supervisor: All work complete. Ending.\n")
        return "end"
    return next_worker

# --- 6. BUILD GRAPH ---
builder = StateGraph(TeamState)
builder.add_node("supervisor", supervisor_node)
builder.add_node("researcher", researcher_node)
builder.add_node("writer", writer_node)
builder.add_node("critic", critic_node)

builder.set_entry_point("supervisor")
builder.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {"researcher": "researcher", "writer": "writer", "critic": "critic", "end": END}
)

builder.add_edge("researcher", "supervisor")
builder.add_edge("writer", "supervisor")
builder.add_edge("critic", "supervisor")

graph = builder.compile()

# --- 7. RUN ---
if __name__ == "__main__":
    initial_state = {
        "task": "The impact of Agentic AI on Indian BPO industry",
        "next_worker": "",
        "research_notes": "",
        "draft": "",
        "final_output": "",
        "history": []
    }
    print("=" * 60)
    print("🚀 MULTI-AGENT SYSTEM STARTING")
    print("=" * 60 + "\n")
    final_state = graph.invoke(initial_state)
    print("=" * 60)
    print("📊 WORKFLOW HISTORY:")
    for step in final_state["history"]:
        print(f"  ✓ {step}")
    print("\n" + "=" * 60)
    print("✅ FINAL OUTPUT:")
    print("=" * 60)
    print(final_state["final_output"])