from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from app.schemas import ChatRequest
from app.utils import extract_text
from app.services import ingest_document_service, rag_chat_service

app = FastAPI(title="RAG Backend")

@app.get("/")
async def root():
    return {"message": "RAG Backend is running. Visit /docs for API documentation.      /Upload_Document to upload files.      /Chat to interact."}

@app.post("/Upload_Document")
async def ingest_file(file: UploadFile = File(...), strategy: str = Form("fixed")):
    try:
        content = await file.read()
        text = extract_text(content, file.filename)
        num = await ingest_document_service(text, file.filename, strategy)
        return {"status": "success", "chunks": num}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Uploadfailed: {str(e)}")

@app.post("/Chat")
async def chat(request: ChatRequest):
    try:
        # The service is now returning a string
        
        answer_text = await rag_chat_service(request.session_id, request.query)
        return {"answer": answer_text}
    except Exception as e:
        # This will catch 'AtlasError' and report it clearly. 
        raise HTTPException(status_code=500, detail=str(e))