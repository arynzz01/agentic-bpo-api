# week1_llm_supervisor.py
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
    iteration_count: int

# --- 2. LLM (Auto-detects Groq or Ollama with fallback) ---
def get_llm():
    """Returns the best available LLM: Groq if it works, otherwise Ollama."""
    if os.getenv("GROQ_API_KEY"):
        try:
            groq_llm = ChatOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.getenv("GROQ_API_KEY"),
                model="openai/gpt-oss-20b",
                temperature=0.3
            )
            groq_llm.invoke("test")
            print("☁️  Connected to Groq\n")
            return groq_llm
        except Exception:
            print("⚠️  Groq unavailable. Falling back to Ollama.\n")
    
    ollama_llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="llama3.2:latest",
        temperature=0.3
    )
    print("🔗 Using Ollama (llama3.2:latest)\n")
    return ollama_llm

llm = get_llm()

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

Return ONLY the final polished article."""
    response = llm.invoke(prompt)
    history = state.get("history", []) + ["critic: final version produced"]
    return {"final_output": response.content, "history": history}

# --- 4. HYBRID SUPERVISOR (State Machine as primary, LLM as optional) ---
def llm_supervisor_node(state: TeamState):
    """
    Supervisor with safety limits.
    - Primary decision: deterministic state machine (never fails)
    - LLM is optional and only used when it doesn't break the flow
    """
    print("👔 Supervisor reasoning...")
    
    iteration = state.get("iteration_count", 0)
    has_research = bool(state.get("research_notes"))
    has_draft = bool(state.get("draft"))
    has_final = bool(state.get("final_output"))
    
    # 🔒 SAFETY RULE 1: Force FINISH if final output exists
    if has_final:
        print("👔 [SAFETY] Final output exists. Forcing FINISH.")
        return {"next_worker": "FINISH", "iteration_count": iteration + 1}
    
    # 🔒 SAFETY RULE 2: Force FINISH after 6 iterations
    if iteration >= 6:
        print("👔 [SAFETY] Max iterations reached. Forcing FINISH.")
        return {"next_worker": "FINISH", "iteration_count": iteration + 1}
    
    # 🧠 PRIMARY DECISION: Deterministic state machine
    if not has_research:
        next_worker = "researcher"
    elif not has_draft:
        next_worker = "writer"
    elif not has_final:
        next_worker = "critic"
    else:
        next_worker = "FINISH"
    
    print(f"👔 Supervisor decided: {next_worker} (iteration {iteration + 1})")
    return {"next_worker": next_worker, "iteration_count": iteration + 1}

# --- 5. ROUTER ---
def route_from_supervisor(state: TeamState) -> Literal["researcher", "writer", "critic", "end"]:
    if state.get("next_worker") == "FINISH":
        print("👔 Supervisor: All work complete. Ending.\n")
        return "end"
    return state.get("next_worker", "end")

# --- 6. BUILD GRAPH ---
builder = StateGraph(TeamState)
builder.add_node("supervisor", llm_supervisor_node)
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
        "task": "The future of remote work in India after AI agents",
        "next_worker": "",
        "research_notes": "",
        "draft": "",
        "final_output": "",
        "history": [],
        "iteration_count": 0
    }
    print("=" * 60)
    print("🚀 LLM-POWERED SUPERVISOR SYSTEM")
    print("=" * 60 + "\n")
    
    final_state = graph.invoke(initial_state)
    
    print("=" * 60)
    print("📊 WORKFLOW HISTORY:")
    for step in final_state["history"]:
        print(f"  ✓ {step}")
    print(f"\n🔄 Total supervisor iterations: {final_state['iteration_count']}")
    print("\n" + "=" * 60)
    print("✅ FINAL OUTPUT:")
    print("=" * 60)
    print(final_state["final_output"])