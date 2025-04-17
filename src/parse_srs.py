import os
import re
from docx import Document
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph
from pydantic import BaseModel
from typing import List, Dict, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Initialize LLM
llm = ChatGroq(model_name="llama3-70b-8192", api_key=GROQ_API_KEY)

# Define Pydantic models for structured output
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
    srs_text: str = ""
    requirements: Requirements = None

# Define WorkflowState for LangGraph
class WorkflowState(TypedDict):
    state: GraphState

# Node 1: Read SRS document
def read_srs_node(state: WorkflowState) -> WorkflowState:
    srs_path = "srs/Python Gen AI SRD backend 14th 18th Apr (1).docx"
    if not os.path.exists(srs_path):
        raise FileNotFoundError(f"SRS file not found at {srs_path}")
    
    doc = Document(srs_path)
    text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    state["state"].srs_text = text
    return state

# Node 2: Analyze SRS with LLM
def analyze_srs_node(state: WorkflowState) -> WorkflowState:
    prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Analyze the following SRS document text and extract the following in JSON format. Return ONLY valid JSON, with no additional text, comments, or code block markers (e.g., ```). The JSON must adhere strictly to the structure provided below.

        Extract:
        1. API endpoints (method, path, description, parameters, response).
        2. Database schema (tables with fields and relationships).
        3. Authentication and authorization requirements.
        4. Business logic (rules and computations).

        Database schema notes:
        - Use 'users' table for employees and managers, not 'employees'.
        - Include a 'pod_members' junction table for the many-to-many relationship between 'users' and 'pods'.
        - Ensure relationships are correctly defined (e.g., foreign keys).

        JSON structure:
        {{
            "endpoints": [
                {{
                    "method": "GET",
                    "path": "/example",
                    "description": "Description",
                    "parameters": {{}},
                    "response": {{}}
                }}
            ],
            "database_schema": [
                {{
                    "name": "table_name",
                    "fields": [{{"name": "field", "type": "type", "constraints": "constraints"}}],
                    "relationships": [{{"field": "field", "references": "table.field"}}]
                }}
            ],
            "auth_requirements": "Description of auth",
            "business_logic": "Description of logic"
        }}

        SRS Text:
        {text}
        """
    )
    
    chain = prompt | llm
    response = chain.invoke({"text": state["state"].srs_text})
    
    # Extract JSON content using regex
    json_pattern = r'\{[\s\S]*\}'
    match = re.search(json_pattern, response.content)
    if not match:
        print(f"Failed to extract JSON from LLM response: {response.content}")
        raise ValueError("No valid JSON found in LLM response")
    
    json_str = match.group(0)
    try:
        requirements_dict = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON extracted: {json_str}")
        raise ValueError("Failed to parse extracted JSON") from e
    
    try:
        state["state"].requirements = Requirements(**requirements_dict)
    except ValueError as e:
        print(f"Pydantic validation failed for requirements: {requirements_dict}")
        raise ValueError("Requirements do not match expected schema") from e
    
    return state

# Node 3: Save requirements to JSON
def save_requirements_node(state: WorkflowState) -> WorkflowState:
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "requirements.json")
    
    with open(output_path, "w") as f:
        f.write(state["state"].requirements.model_dump_json(indent=2))
    
    return state

# Define LangGraph workflow
workflow = StateGraph(WorkflowState)
workflow.add_node("read_srs", read_srs_node)
workflow.add_node("analyze_srs", analyze_srs_node)
workflow.add_node("save_requirements", save_requirements_node)

# Define edges
workflow.add_edge("read_srs", "analyze_srs")
workflow.add_edge("analyze_srs", "save_requirements")

# Set entry point
workflow.set_entry_point("read_srs")

# Compile the workflow
app = workflow.compile()

# Run the workflow
if __name__ == "__main__":
    initial_state = WorkflowState(state=GraphState())
    result = app.invoke(initial_state)
    print("Requirements saved to outputs/requirements.json")