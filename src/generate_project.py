import os
import json
import subprocess
import sys
import venv
import time
import uuid
import logging
from langgraph.graph import StateGraph
from pydantic import BaseModel
from typing import List, Dict, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Initialize Groq LLM
llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

# Define Pydantic models for requirements.json
class Endpoint(BaseModel):
    method: str
    path: str
    description: str
    parameters: Dict
    response: Dict

class DatabaseTable(BaseModel):
    name: str
    fields: List[Dict[str, str]]
    relationships: List[Dict[str, str]]

class Requirements(BaseModel):
    endpoints: List[Endpoint]
    database_schema: List[DatabaseTable]
    auth_requirements: str
    business_logic: str

class GraphState(BaseModel):
    requirements: Requirements = None
    generated_files: Dict[str, str] = {}
    project_structure: Dict[str, str] = {}
    langsmith_run_id: Optional[str] = None

# Define WorkflowState for LangGraph
class WorkflowState(TypedDict):
    state: GraphState

# Node 1: Load requirements.json
def load_requirements_node(state: WorkflowState) -> WorkflowState:
    requirements_path = "outputs/requirements.json"
    if not os.path.exists(requirements_path):
        raise FileNotFoundError(f"requirements.json not found at {requirements_path}")
    
    with open(requirements_path, "r") as f:
        requirements_dict = json.load(f)
    
    state["state"].requirements = Requirements(**requirements_dict)
    logger.info("Loaded requirements from JSON")
    return state

# Node 2: Generate project structure
def generate_project_structure_node(state: WorkflowState) -> WorkflowState:
    project_dir = "generated_project"
    structure = {
        "app": {
            "__init__.py": "",
            "api": {
                "routes": ["user.py", "leave.py", "pod.py", "pod_members.py", "__init__.py"],
                "__init__.py": ""
            },
            "models": ["user.py", "leave.py", "pod.py", "pod_members.py", "__init__.py"],
            "services": ["auth.py", "leave.py", "pod.py", "__init__.py"],
            "database.py": "",
            "main.py": ""
        },
        "tests": ["test_user.py", "test_leave.py", "test_pod.py", "test_pod_members.py"],
        "migrations": {},
        "podman-compose.yml": "",
        "requirements.txt": "",
        ".env": "",
        "README.md": "",
        "api_docs.md": "",
        "workflow.mmd": ""
    }

    def create_structure(base_path, items):
        for name, content in items.items():
            path = os.path.join(base_path, name)
            if isinstance(content, dict):
                os.makedirs(path, exist_ok=True)
                create_structure(path, content)
            elif isinstance(content, list):
                os.makedirs(path, exist_ok=True)
                for file in content:
                    file_path = os.path.join(path, file)
                    if file_path not in state["state"].generated_files:
                        with open(file_path, "w") as f:
                            f.write("")
                        state["state"].generated_files[file_path] = ""
            else:
                if path not in state["state"].generated_files:
                    with open(path, "w") as f:
                        f.write(content)
                    state["state"].generated_files[path] = content

    os.makedirs(project_dir, exist_ok=True)
    create_structure(project_dir, structure)
    state["state"].project_structure = structure
    logger.info("Project structure generated")
    return state

# Node 3: Generate project files
def generate_project_files_node(state: WorkflowState) -> WorkflowState:
    requirements = state["state"].requirements
    project_dir = "generated_project"

    # Generate podman-compose.yml
    podman_compose_content = """
version: '3.8'
services:
  postgres:
    image: docker.io/postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: your_password
      POSTGRES_DB: srs_fastapi
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
volumes:
  postgres_data:
"""
    podman_compose_path = os.path.join(project_dir, "podman-compose.yml")
    with open(podman_compose_path, "w") as f:
        f.write(podman_compose_content)
    state["state"].generated_files[podman_compose_path] = podman_compose_content

    # Generate .env
    env_content = "DATABASE_URL=postgresql+psycopg://postgres:your_password@localhost:5432/srs_fastapi"
    env_path = os.path.join(project_dir, ".env")
    with open(env_path, "w") as f:
        f.write(env_content)
    state["state"].generated_files[env_path] = env_content

    # Generate requirements.txt
    requirements_content = """
fastapi==0.110.0
uvicorn==0.29.0
sqlalchemy==2.0.25
psycopg[binary]==3.2.1
alembic==1.13.1
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.1
pytest==8.1.1
langchain==0.1.14
langchain-groq==0.0.1
pydantic==2.6.4
"""
    req_path = os.path.join(project_dir, "requirements.txt")
    with open(req_path, "w") as f:
        f.write(requirements_content)
    state["state"].generated_files[req_path] = requirements_content

    # Generate database.py
    database_content = """
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
import os

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
"""
    db_path = os.path.join(project_dir, "app", "database.py")
    with open(db_path, "w") as f:
        f.write(database_content)
    state["state"].generated_files[db_path] = database_content

    # Generate main.py
    main_content = """
from fastapi import FastAPI
from app.api.routes.user import router as user_router
from app.api.routes.leave import router as leave_router
from app.api.routes.pod import router as pod_router
from app.api.routes.pod_members import router as pod_members_router
from app.database import Base, engine

app = FastAPI(title="Leave Management System")

app.include_router(user_router)
app.include_router(leave_router)
app.include_router(pod_router)
app.include_router(pod_members_router)

Base.metadata.create_all(bind=engine)
"""
    main_content = main_content.strip()
    main_path = os.path.join(project_dir, "app", "main.py")
    with open(main_path, "w") as f:
        f.write(main_content)
    state["state"].generated_files[main_path] = main_content

    # Generate model files
    model_files_dict = {
        "app/models/user.py": """
from app.database import Base
from sqlalchemy import Column, Integer, String, Enum
import enum

class UserRole(enum.Enum):
    manager = "manager"
    employee = "employee"

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    name = Column(String)
    role = Column(Enum(UserRole))
""".strip(),
        "app/models/leave.py": """
from app.database import Base
from sqlalchemy import Column, Integer, String, Date, ForeignKey
from sqlalchemy.orm import relationship

class Leave(Base):
    __tablename__ = "leaves"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    start_date = Column(Date)
    end_date = Column(Date)
    reason = Column(String)
    status = Column(String)
    user = relationship("User", backref="leaves")
""".strip(),
        "app/models/pod.py": """
from app.database import Base
from sqlalchemy import Column, Integer, String

class Pod(Base):
    __tablename__ = "pods"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
""".strip(),
        "app/models/pod_members.py": """
from app.database import Base
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

class PodMember(Base):
    __tablename__ = "pod_members"
    id = Column(Integer, primary_key=True, index=True)
    pod_id = Column(Integer, ForeignKey("pods.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    role = Column(String)
    pod = relationship("Pod", backref="pod_members")
    user = relationship("User", backref="pod_members")
""".strip()
    }

    for file_path, code in model_files_dict.items():
        full_path = os.path.join(project_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(code)
        state["state"].generated_files[full_path] = code

    # Generate route files with full CRUD
    route_files_dict = {
        "app/api/routes/user.py": """
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.user import User, UserRole
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/users", tags=["users"])

class UserCreate(BaseModel):
    email: str
    password: str
    name: str
    role: str

class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    role: str

    class Config:
        orm_mode = True

@router.get("/", response_model=List[UserResponse])
async def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return users

@router.post("/", response_model=UserResponse)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    try:
        role = UserRole[user.role]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid role")
    db_user = User(
        email=user.email,
        password=user.password,
        name=user.name,
        role=role
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.delete("/{user_id}")
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {"message": "User deleted"}
""".strip(),
        "app/api/routes/leave.py": """
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.leave import Leave
from app.models.user import User
from pydantic import BaseModel
from typing import List
from datetime import date

router = APIRouter(prefix="/leaves", tags=["leaves"])

class LeaveCreate(BaseModel):
    user_id: int
    start_date: date
    end_date: date
    reason: str
    status: str

class LeaveResponse(BaseModel):
    id: int
    user_id: int
    start_date: date
    end_date: date
    reason: str
    status: str

    class Config:
        orm_mode = True

@router.get("/", response_model=List[LeaveResponse])
async def get_leaves(db: Session = Depends(get_db)):
    leaves = db.query(Leave).all()
    return leaves

@router.post("/", response_model=LeaveResponse)
async def create_leave(leave: LeaveCreate, db: Session = Depends(get_db)):
    if not db.query(User).filter(User.id == leave.user_id).first():
        raise HTTPException(status_code=404, detail="User not found")
    db_leave = Leave(
        user_id=leave.user_id,
        start_date=leave.start_date,
        end_date=leave.end_date,
        reason=leave.reason,
        status=leave.status
    )
    db.add(db_leave)
    db.commit()
    db.refresh(db_leave)
    return db_leave

@router.get("/{leave_id}", response_model=LeaveResponse)
async def get_leave(leave_id: int, db: Session = Depends(get_db)):
    leave = db.query(Leave).filter(Leave.id == leave_id).first()
    if not leave:
        raise HTTPException(status_code=404, detail="Leave not found")
    return leave

@router.delete("/{leave_id}")
async def delete_leave(leave_id: int, db: Session = Depends(get_db)):
    leave = db.query(Leave).filter(Leave.id == leave_id).first()
    if not leave:
        raise HTTPException(status_code=404, detail="Leave not found")
    db.delete(leave)
    db.commit()
    return {"message": "Leave deleted"}
""".strip(),
        "app/api/routes/pod.py": """
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.pod import Pod
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/pods", tags=["pods"])

class PodCreate(BaseModel):
    name: str

class PodResponse(BaseModel):
    id: int
    name: str

    class Config:
        orm_mode = True

@router.get("/", response_model=List[PodResponse])
async def get_pods(db: Session = Depends(get_db)):
    pods = db.query(Pod).all()
    return pods

@router.post("/", response_model=PodResponse)
async def create_pod(pod: PodCreate, db: Session = Depends(get_db)):
    db_pod = Pod(name=pod.name)
    db.add(db_pod)
    db.commit()
    db.refresh(db_pod)
    return db_pod

@router.get("/{pod_id}", response_model=PodResponse)
async def get_pod(pod_id: int, db: Session = Depends(get_db)):
    pod = db.query(Pod).filter(Pod.id == pod_id).first()
    if not pod:
        raise HTTPException(status_code=404, detail="Pod not found")
    return pod

@router.delete("/{pod_id}")
async def delete_pod(pod_id: int, db: Session = Depends(get_db)):
    pod = db.query(Pod).filter(Pod.id == pod_id).first()
    if not pod:
        raise HTTPException(status_code=404, detail="Pod not found")
    db.delete(pod)
    db.commit()
    return {"message": "Pod deleted"}
""".strip(),
        "app/api/routes/pod_members.py": """
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.pod_members import PodMember
from app.models.pod import Pod
from app.models.user import User
from pydantic import BaseModel
from typing import List

router = APIRouter(prefix="/pod-members", tags=["pod-members"])

class PodMemberCreate(BaseModel):
    pod_id: int
    user_id: int
    role: str

class PodMemberResponse(BaseModel):
    id: int
    pod_id: int
    user_id: int
    role: str

    class Config:
        orm_mode = True

@router.get("/", response_model=List[PodMemberResponse])
async def get_pod_members(db: Session = Depends(get_db)):
    pod_members = db.query(PodMember).all()
    return pod_members

@router.post("/", response_model=PodMemberResponse)
async def create_pod_member(pod_member: PodMemberCreate, db: Session = Depends(get_db)):
    if not db.query(Pod).filter(Pod.id == pod_member.pod_id).first():
        raise HTTPException(status_code=404, detail="Pod not found")
    if not db.query(User).filter(User.id == pod_member.user_id).first():
        raise HTTPException(status_code=404, detail="User not found")
    db_pod_member = PodMember(
        pod_id=pod_member.pod_id,
        user_id=pod_member.user_id,
        role=pod_member.role
    )
    db.add(db_pod_member)
    db.commit()
    db.refresh(db_pod_member)
    return db_pod_member

@router.get("/{pod_member_id}", response_model=PolyMemberResponse)
async def get_pod_member(pod_member_id: int, db: Session = Depends(get_db)):
    pod_member = db.query(PodMember).filter(PodMember.id == pod_member_id).first()
    if not pod_member:
        raise HTTPException(status_code=404, detail="Pod member not found")
    return pod_member

@router.delete("/{pod_member_id}")
async def delete_pod_member(pod_member_id: int, db: Session = Depends(get_db)):
    pod_member = db.query(PodMember).filter(PodMember.id == pod_member_id).first()
    if not pod_member:
        raise HTTPException(status_code=404, detail="Pod member not found")
    db.delete(pod_member)
    db.commit()
    return {"message": "Pod member deleted"}
""".strip()
    }

    for file_path, code in route_files_dict.items():
        full_path = os.path.join(project_dir, file_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(code)
        state["state"].generated_files[full_path] = code

    logger.info("Project files generated")
    return state

# Node 4: Generate documentation
def generate_documentation_node(state: WorkflowState) -> WorkflowState:
    project_dir = "generated_project"
    requirements = state["state"].requirements

    # LLM prompt for Mermaid workflow diagram
    mermaid_prompt = PromptTemplate(
        input_variables=["workflow_description"],
        template="""
Generate a Mermaid diagram in the format `graph TD` to visualize a LangGraph workflow for a FastAPI project generation process. The workflow consists of four sequential nodes:
1. Load Requirements: Loads requirements from a JSON file.
2. Generate Project Structure: Creates the project directory and file structure.
3. Generate Project Files: Populates files with FastAPI code and configurations.
4. Generate Documentation: Creates documentation files (Mermaid diagram, README, API docs).

The nodes should be connected in this order with arrows (-->). Use square brackets for node labels (e.g., [Load Requirements]). Output only the Mermaid code, starting with `graph TD`, without any additional text or code fences.

Example:
graph TD
    A[Load Requirements] --> B[Generate Project Structure]
    B --> C[Generate Project Files]
    C --> D[Generate Documentation]

Workflow Description: {workflow_description}
"""
    )

    # Generate Mermaid diagram with LLM
    try:
        chain = mermaid_prompt | llm | StrOutputParser()
        mermaid_content = chain.invoke({
            "workflow_description": "A LangGraph workflow that sequentially loads requirements from a JSON file, generates a FastAPI project structure, populates project files with code, and generates documentation including a Mermaid diagram, README, and API documentation."
        })
        # Validate Mermaid output
        if not mermaid_content.startswith("graph TD"):
            raise ValueError("Invalid Mermaid diagram generated by LLM")
        # Ensure consistent formatting
        mermaid_content = mermaid_content.strip()
        mermaid_path = os.path.join(project_dir, "workflow.mmd")
        with open(mermaid_path, "w") as f:
            f.write(mermaid_content)
        state["state"].generated_files[mermaid_path] = mermaid_content
        logger.info("Generated Mermaid workflow diagram with LLM")
    except Exception as e:
        logger.error(f"Failed to generate Mermaid diagram: {e}")
        # Fallback to hardcoded diagram
        mermaid_content = """
graph TD
    A[Load Requirements] --> B[Generate Project Structure]
    B --> C[Generate Project Files]
    C --> D[Generate Documentation]
"""
        mermaid_path = os.path.join(project_dir, "workflow.mmd")
        with open(mermaid_path, "w") as f:
            f.write(mermaid_content)
        state["state"].generated_files[mermaid_path] = mermaid_content
        logger.info("Used fallback hardcoded Mermaid diagram")

    # LLM prompt for README.md
    readme_prompt = PromptTemplate(
        input_variables=["requirements", "project_structure"],
        template="""
Generate a comprehensive README.md in Markdown format for a FastAPI project generated from an SRS document. Include:

1. **Project Overview**: Describe the project as a leave management system with user, leave, pod, and pod member management.
2. **Setup Instructions**: Detailed steps to set up the project, including Python version, virtual environment, Podman for PostgreSQL, and running the FastAPI server.
3. **Usage**: How to start the server and access the API at http://localhost:8000/docs.
4. **Project Structure**: Explain the directory structure based on the provided project_structure.
5. **Troubleshooting**: Common issues and solutions (e.g., Podman setup, Alembic migrations, missing dependencies).
6. **License**: Use MIT License.

Use the provided requirements and project_structure to tailor the content. Ensure the tone is professional and the instructions are clear for a Windows user.

Requirements: {requirements}
Project Structure: {project_structure}
"""
    )

    # Generate README.md with LLM
    try:
        chain = readme_prompt | llm | StrOutputParser()
        readme_content = chain.invoke({
            "requirements": requirements.dict(),
            "project_structure": json.dumps(state["state"].project_structure, indent=2)
        })
        # Clean up LLM output (remove code fences if present)
        readme_content = readme_content.replace("```markdown\n", "").replace("\n```", "").strip()
        readme_path = os.path.join(project_dir, "README.md")
        with open(readme_path, "w") as f:
            f.write(readme_content)
        state["state"].generated_files[readme_path] = readme_content
        logger.info("Generated enhanced README.md with LLM")
    except Exception as e:
        logger.error(f"Failed to generate README.md: {e}")
        raise

    # LLM prompt for API documentation
    api_docs_prompt = PromptTemplate(
        input_variables=["endpoints"],
        template="""
Generate API documentation in Markdown format for a FastAPI project. Include:

1. **Overview**: Describe the API as a leave management system with endpoints for users, leaves, pods, and pod members.
2. **Endpoints**: For each endpoint, provide:
   - Method and path (e.g., GET /users/).
   - Description.
   - Request parameters (if any).
   - Response format (example JSON).
   - Example curl command.
3. **Accessing the API**: Note that the API is available at http://localhost:8000/docs with Swagger UI.

Use the provided endpoints from requirements.json to generate detailed documentation. Ensure the tone is professional and examples are accurate.

Endpoints: {endpoints}
"""
    )

    # Generate api_docs.md with LLM
    try:
        chain = api_docs_prompt | llm | StrOutputParser()
        api_docs_content = chain.invoke({
            "endpoints": json.dumps([e.dict() for e in requirements.endpoints], indent=2)
        })
        # Clean up LLM output
        api_docs_content = api_docs_content.replace("```markdown\n", "").replace("\n```", "").strip()
        api_docs_path = os.path.join(project_dir, "api_docs.md")
        with open(api_docs_path, "w") as f:
            f.write(api_docs_content)
        state["state"].generated_files[api_docs_path] = api_docs_content
        logger.info("Generated API documentation (api_docs.md) with LLM")
    except Exception as e:
        logger.error(f"Failed to generate api_docs.md: {e}")
        raise

    return state

# Define LangGraph workflow
workflow = StateGraph(WorkflowState)
workflow.add_node("load_requirements", load_requirements_node)
workflow.add_node("generate_project_structure", generate_project_structure_node)
workflow.add_node("generate_project_files", generate_project_files_node)
workflow.add_node("generate_documentation", generate_documentation_node)

# Define edges
workflow.add_edge("load_requirements", "generate_project_structure")
workflow.add_edge("generate_project_structure", "generate_project_files")
workflow.add_edge("generate_project_files", "generate_documentation")

# Set entry point
workflow.set_entry_point("load_requirements")

# Compile the workflow
app = workflow.compile()

# Run the workflow
if __name__ == "__main__":
    initial_state = WorkflowState(state=GraphState(langsmith_run_id=str(uuid.uuid4())))
    try:
        result = app.invoke(initial_state)
        print("Project setup complete. Files generated and database initialized in generated_project/")
        print("Documentation generated: README.md, api_docs.md, workflow.mmd")
        print("To start the FastAPI server:")
        print("cd generated_project")
        print("venv\\Scripts\\activate")
        print("uvicorn app.main:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        print(f"Setup failed: {e}")