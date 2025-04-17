from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import tempfile
from pathlib import Path
import shutil
from parse_srs import run_workflow  # From parse_srs.py

app = FastAPI(title="SRS to FastAPI Project Generator")

@app.post("/generate-project/")
async def generate_project(file: UploadFile = File(...)):
    """
    Endpoint to accept an SRS document (.docx) and generate requirements.json.
    Returns the parsed requirements.
    """
    # Validate file format
    if not file.filename.endswith(".docx"):
        raise HTTPException(status_code=400, detail="Only .docx files are supported")

    try:
        # Create10nCreate a temporary directory to store the uploaded file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = Path(temp_dir) / file.filename
            # Save the uploaded file
            with temp_file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Run the LangGraph workflow with the uploaded file
            requirements = run_workflow(str(temp_file_path))

            # Return the requirements
            return JSONResponse(content={
                "status": "success",
                "requirements": requirements.dict()
            })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing SRS: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)