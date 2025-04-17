# SRS to FastAPI Project

This project builds an AI-powered backend assistant that analyzes Software Requirements Specification (SRS) documents and generates a FastAPI-based backend system. It leverages LangGraph for workflow orchestration, LangChain for LLM integration, and modern software engineering practices like modularity, testing, and persistence. The project is structured to meet milestones outlined in the assignment, starting with SRS analysis.

## Current Status

This repository currently implements **Milestone 1**: Analysis of SRS documents to extract structured requirements (API endpoints, database schema, authentication, and business logic) using a LangGraph workflow.

## Repository Contents

- `.gitignore`: Excludes sensitive files (e.g., `.env`, `venv/`, `generated_project/`).
- `parse_srs.py`: Implements Milestone 1, analyzing SRS documents and generating `requirements.json`.
- `requirements.txt`: Lists Python dependencies for the project.

## Setup

Follow these steps to set up the project on Windows:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/your-username/srs-to-fastapi.git
   cd srs-to-fastapi
   ```

2. **Create and Activate a Virtual Environment**:

   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` includes:

   - `python-docx==0.8.11`
   - `langchain==0.1.14`
   - `langgraph==0.0.24`
   - `langchain-groq==0.0.1`
   - `pydantic==2.6.4`
   - `python-dotenv==1.0.1`

4. **Set Up Environment Variables**: Create a `.env` file in the project root (`C:\Users\prastripathi\Desktop\srs-to-fastapi`) with your Groq API key:

   ```
   GROQ_API_KEY=your_groq_api_key
   ```

   Obtain the key from Groq.

## Milestone 1: SRS Analysis

**Objective**: Build an AI workflow using LangGraph to analyze an SRS document and extract structured software requirements.

**Implementation** (`parse_srs.py`):

- **Input**: A `.docx` SRS document (e.g., `srs/Python Gen AI SRD backend 14th 18th Apr (1).docx`).
- **Process**:
  - Reads the SRS document using `python-docx`.
  - Uses a LangGraph workflow with nodes:
    - `read_srs`: Extracts text from the SRS document.
    - `analyze_srs`: Processes text with Llama 3 (via `langchain-groq`) to extract:
      - API endpoints (method, path, description, parameters, response).
      - Database schema (tables, fields, relationships).
      - Authentication and authorization requirements.
      - Business logic.
    - `save_requirements`: Saves extracted requirements as `requirements.json` in the `outputs/` directory.
- **Output**: A JSON file (`outputs/requirements.json`) with structured requirements.
- **Technologies**:
  - LangGraph for workflow orchestration.
  - LangChain with Groq’s Llama 3 (70B) for LLM-driven analysis.
  - Pydantic for structured data validation.

**Usage**:

1. Place the SRS `.docx` file in the `srs/` directory (e.g., `srs/Python Gen AI SRD backend 14th 18th Apr (1).docx`).

2. Run the script:

   ```bash
   python parse_srs.py
   ```

3. Check the output in `outputs/requirements.json`.

**Example Output** (`requirements.json`):

```json
{
  "endpoints": [
    {
      "method": "GET",
      "path": "/example",
      "description": "Retrieve example data",
      "parameters": {},
      "response": {}
    }
  ],
  "database_schema": [
    {
      "name": "users",
      "fields": [{"name": "id", "type": "Integer", "constraints": "primary_key"}],
      "relationships": []
    }
  ],
  "auth_requirements": "JWT authentication",
  "business_logic": "Example business rules"
}
```

## Future Milestones

- **Milestone 2**: Generate FastAPI project structure.
- **Milestone 3**: Autonomous coding with unit tests.
- **Milestone 4**: Persistence and iterative improvements.
- **Milestone 5**: Deployment via zip archive.
- **Milestone 6**: Documentation with Mermaid diagrams.
- **Milestone 7**: LangSmith logging.
- **Milestone 8**: FastAPI endpoint for SRS input.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

MIT License