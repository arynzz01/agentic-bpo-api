# main.py
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import re
import logging
from datetime import datetime
from dotenv import load_dotenv
from openai import AsyncOpenAI

# --- Rate Limiting Imports ---
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

# ============================================================
# SECTION 1: LOGGING SETUP (Observability - Topic 7)
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("agentic_bpo")

# ============================================================
# SECTION 2: GLOBAL STATE
# ============================================================
DATA_FILE = "data.json"
employees = {}
managers = {}

# ============================================================
# SECTION 3: OOP CLASSES
# ============================================================
class Employee:
    def __init__(self, emp_id, name, role):
        self.emp_id = emp_id
        self.role = role
        self.name = name
        self.tasks = []
        self.shift = "Morning"

    def assign_task(self, task_name):
        self.tasks.append(task_name)
        return f"Task '{task_name}' assigned to {self.name}"

    def complete_task(self, task_name):
        if task_name in self.tasks:
            self.tasks.remove(task_name)
            return f"Task '{task_name}' completed by {self.name}"
        return f"Task '{task_name}' not found"

    def display_info(self):
        return f"ID:{self.emp_id} | {self.name} | Role:{self.role} | Shift:{self.shift} | Tasks:{self.tasks}"


class Manager(Employee):
    def __init__(self, emp_id, name, role, team_size):
        super().__init__(emp_id, name, role)
        self.team_size = team_size
        self.team_members = []

    def add_team_member(self, employee):
        if employee not in self.team_members:
            self.team_members.append(employee)
            return f"{employee.name} added to team."
        return f"{employee.name} is already in the team."

    def delegate_task(self, employee, task_name):
        if employee in self.team_members:
            employee.assign_task(task_name)
            return f"Task '{task_name}' delegated to {employee.name}"
        return f"{employee.name} not in team!"

    def remove_team_member(self, employee):
        if employee in self.team_members:
            self.team_members.remove(employee)
            return f"{employee.name} removed from team."
        return f"{employee.name} not found."

    def display_info(self):
        base = super().display_info()
        member_names = [m.name for m in self.team_members]
        return f"{base} | Team Size: {self.team_size} | Team: {member_names}"

# ============================================================
# SECTION 4: PERSISTENCE
# ============================================================
def save_data():
    data = {"employees": {}, "managers": {}}
    for emp_id, emp in employees.items():
        data["employees"][emp_id] = {
            "emp_id": emp.emp_id, "name": emp.name, "role": emp.role,
            "tasks": emp.tasks, "shift": emp.shift
        }
    for mgr_id, mgr in managers.items():
        data["managers"][mgr_id] = {
            "emp_id": mgr.emp_id, "name": mgr.name, "role": mgr.role,
            "tasks": mgr.tasks, "shift": mgr.shift,
            "team_size": mgr.team_size,
            "team_member_ids": [m.emp_id for m in mgr.team_members]
        }
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def load_data():
    if not os.path.exists(DATA_FILE):
        return
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    for emp_id, emp_data in data["employees"].items():
        emp = Employee(emp_data["emp_id"], emp_data["name"], emp_data["role"])
        emp.tasks = emp_data["tasks"]
        emp.shift = emp_data["shift"]
        employees[int(emp_id)] = emp
    for mgr_id, mgr_data in data["managers"].items():
        mgr = Manager(mgr_data["emp_id"], mgr_data["name"], mgr_data["role"], mgr_data["team_size"])
        mgr.tasks = mgr_data["tasks"]
        mgr.shift = mgr_data["shift"]
        for member_id in mgr_data["team_member_ids"]:
            if member_id in employees:
                mgr.team_members.append(employees[member_id])
        managers[int(mgr_id)] = mgr

# ============================================================
# SECTION 5: FASTAPI APP + RATE LIMITER (Topic 8)
# ============================================================
app = FastAPI(title="Agentic BPO API with Local Ollama")

# --- Rate Limiter Setup ---
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Load existing data
load_data()
print(f"✅ Loaded {len(employees)} employees and {len(managers)} managers from memory.")

# ============================================================
# SECTION 6: UNIVERSAL AI CLIENT (Ollama or Groq)
# ============================================================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if GROQ_API_KEY:
    print("☁️  GROQ_API_KEY found. Running in CLOUD MODE with Groq.")
    ai_client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=GROQ_API_KEY,
    )
    AI_MODEL = "llama-3.1-8b-instant"
else:
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    print(f"🔗 No Groq key found. Running in LOCAL MODE with Ollama at {OLLAMA_BASE_URL}")
    ai_client = AsyncOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
    )
    AI_MODEL = "llama3.2:latest"

# ============================================================
# SECTION 7: PYDANTIC MODELS
# ============================================================
class EmployeeCreate(BaseModel):
    emp_id: int
    name: str
    role: str

class TaskAssign(BaseModel):
    task_name: str

class DelegateTask(BaseModel):
    manager_id: int
    employee_id: int
    task_name: str

class ChatRequest(BaseModel):
    message: str

class DelegateAIRequest(BaseModel):
    manager_id: int
    task_description: str

# ============================================================
# SECTION 8: AI CHAT ENDPOINT (With Logging + Rate Limiting)
# ============================================================
@app.post("/ai/chat/")
@limiter.limit("5/minute")
async def ai_chat(request: Request, chat_request: ChatRequest):
    try:
        logger.info(f"📥 AI Chat Request: {chat_request.message}")
        
        system_prompt = """You are an AI assistant for a BPO Task Management System. 
        You help employees understand their tasks, suggest optimal task delegation, 
        and provide guidance on workflow management. Keep responses concise and actionable."""
        
        completion = await ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chat_request.message}
            ],
            temperature=0.3,
        )
        
        logger.info(f"✅ AI Chat Response received")
        
        return {
            "response": completion.choices[0].message.content,
            "model": AI_MODEL,
            "usage": {
                "prompt_tokens": completion.usage.prompt_tokens if completion.usage else 0,
                "completion_tokens": completion.usage.completion_tokens if completion.usage else 0,
                "total_tokens": completion.usage.total_tokens if completion.usage else 0
            }
        }
    except Exception as e:
        logger.error(f"❌ AI Chat Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")

# ============================================================
# SECTION 9: AI DELEGATE ENDPOINT (With Logging + Rate Limiting)
# ============================================================
@app.post("/ai/delegate/")
@limiter.limit("5/minute")
async def ai_delegate(request: Request, delegate_request: DelegateAIRequest):
    if delegate_request.manager_id not in managers:
        raise HTTPException(status_code=404, detail="Manager not found")
    
    manager = managers[delegate_request.manager_id]
    
    if not employees:
        raise HTTPException(status_code=400, detail="No employees available to assign tasks.")
    
    employee_list_str = ""
    for emp_id, emp in employees.items():
        task_count = len(emp.tasks)
        employee_list_str += f"ID: {emp.emp_id}, Name: {emp.name}, Role: {emp.role}, Current Tasks: {task_count}\n"

    prompt = f"""
You are an intelligent task delegation agent. 
You must assign the following task to the most suitable employee.

Task: "{delegate_request.task_description}"

Available Employees:
{employee_list_str}

Consider their roles (Agent, Senior Agent, Team Lead) and their current workload (Current Tasks).
Select the best employee ID for this task.

IMPORTANT: Return ONLY the employee ID number (integer) in your response.
    """

    try:
        logger.info(f"🎯 AI Delegation Request: '{delegate_request.task_description}' | Manager: {manager.name}")
        
        completion = await ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": "You are a precise task routing AI. You only output integer IDs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )

        response_text = completion.choices[0].message.content.strip()
        numbers = re.findall(r'\d+', response_text)
        if not numbers:
            raise HTTPException(status_code=500, detail=f"AI didn't return a valid employee ID. Response: {response_text}")
        
        selected_emp_id = int(numbers[0])

        if selected_emp_id not in employees:
            raise HTTPException(status_code=404, detail=f"AI selected ID {selected_emp_id}, but this employee does not exist.")
        
        employee = employees[selected_emp_id]
        logger.info(f"🤖 AI Selected Employee ID: {selected_emp_id} ({employee.name})")
        
        result = manager.delegate_task(employee, delegate_request.task_description)
        save_data()

        return {
            "message": "Task delegated successfully by AI!",
            "selected_employee": {"id": employee.emp_id, "name": employee.name, "role": employee.role},
            "ai_response": response_text,
            "delegation_result": result
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"❌ AI Delegation Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI Delegation Error: {str(e)}")

# ============================================================
# SECTION 10: CRUD ENDPOINTS (Unchanged)
# ============================================================
@app.get("/")
def root():
    return {"message": "Agentic BPO API is LIVE!", "status": "🚀"}

@app.post("/employees/")
def create_employee(emp: EmployeeCreate):
    if emp.emp_id in employees:
        raise HTTPException(status_code=400, detail="Employee ID already exists")
    new_emp = Employee(emp.emp_id, emp.name, emp.role)
    employees[emp.emp_id] = new_emp
    save_data()
    logger.info(f"👤 Employee created: {emp.name} (ID: {emp.emp_id})")
    return {"message": f"Employee {emp.name} created", "emp_id": emp.emp_id}

@app.post("/managers/")
def create_manager(emp: EmployeeCreate, team_size: int = 5):
    if emp.emp_id in managers:
        raise HTTPException(status_code=400, detail="Manager ID already exists")
    new_mgr = Manager(emp.emp_id, emp.name, emp.role, team_size)
    managers[emp.emp_id] = new_mgr
    save_data()
    logger.info(f"👔 Manager created: {emp.name} (ID: {emp.emp_id})")
    return {"message": f"Manager {emp.name} created", "manager_id": emp.emp_id}

@app.post("/managers/{manager_id}/add_member/")
def add_team_member(manager_id: int, employee_id: int):
    if manager_id not in managers:
        raise HTTPException(status_code=404, detail="Manager not found")
    if employee_id not in employees:
        raise HTTPException(status_code=404, detail="Employee not found")
    manager = managers[manager_id]
    employee = employees[employee_id]
    result = manager.add_team_member(employee)
    save_data()
    return {"message": result}

@app.post("/employees/{emp_id}/tasks/")
def assign_task(emp_id: int, task: TaskAssign):
    if emp_id not in employees:
        raise HTTPException(status_code=404, detail="Employee not found")
    result = employees[emp_id].assign_task(task.task_name)
    save_data()
    return {"message": result}

@app.post("/delegate/")
def delegate_task(data: DelegateTask):
    if data.manager_id not in managers:
        raise HTTPException(status_code=404, detail="Manager not found")
    if data.employee_id not in employees:
        raise HTTPException(status_code=404, detail="Employee not found")
    manager = managers[data.manager_id]
    employee = employees[data.employee_id]
    result = manager.delegate_task(employee, data.task_name)
    save_data()
    return {"message": result}

@app.get("/employees/")
def list_employees():
    return {"employees": [e.display_info() for e in employees.values()]}

@app.get("/managers/")
def list_managers():
    return {"managers": [m.display_info() for m in managers.values()]}

@app.delete("/employees/{emp_id}")
def delete_employee(emp_id: int):
    if emp_id not in employees:
        raise HTTPException(status_code=404, detail="Employee not found")
    del employees[emp_id]
    save_data()
    return {"message": f"Employee {emp_id} deleted"}