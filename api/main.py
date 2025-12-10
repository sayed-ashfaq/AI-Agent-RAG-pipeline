"""
FastAPI application for AI Agent with RAG
Main server implementation
"""
import sys
from pathlib import Path

# Add project root to path for imports to work from any directory
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import (
    FastAPI, 
    HTTPException, 
    File, 
    UploadFile, 
    Depends, 
    Request,
    Header
)
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional, Dict, List
import asyncio
from contextlib import asynccontextmanager

from src.schemas import (
    ChatRequest,
    ChatResponse,
    InitSessionResponse,
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
    ClearChatResponse,
    ErrorResponse,
    DocumentMetadata,
    ChatMessage
)
from src.session_manager import session_manager, SessionManager
from src.document_handler import DocumentHandler
from src.tools.rag_agent.retriever import Retriever
from src.workflow.agent_workflow import ReActAgent
from custom_logger import GLOBAL_LOGGER as logger


# ============== Global State ==============
# These will be initialized at startup
retriever: Optional[Retriever] = None
agent: Optional[ReActAgent] = None
session_retrievers: Dict[str, Retriever] = {}  # Session-specific retrievers
session_agents: Dict[str, ReActAgent] = {}  # Session-specific agents


# ============== Startup/Shutdown ==============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events
    """
    # Startup
    global retriever, agent
    logger.info("🚀 Starting FastAPI application...")
    
    try:
        logger.info("Initializing Vector Store...")
        retriever = Retriever()
        
        logger.info("Initializing AI Agent...")
        agent = ReActAgent(retriever=retriever)
        
        logger.info("✅ Application initialized successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down FastAPI application...")
    try:
        # Cleanup can be added here if needed
        logger.info("✅ Application shutdown successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# ============== FastAPI App Setup ==============
app = FastAPI(
    title="AI Agent with RAG",
    description="FastAPI backend for AI Agent with RAG pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files and templates
BASE_DIR = Path(__file__).resolve().parent.parent
static_dir = BASE_DIR / "static"
templates_dir = BASE_DIR / "templates"

static_dir.mkdir(exist_ok=True)
templates_dir.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(templates_dir))


# ============== Dependencies ==============
def get_session_id(x_session_id: Optional[str] = Header(None)) -> str:
    """
    Get session ID from header or create new session
    """
    if x_session_id and session_manager.session_exists(x_session_id):
        return x_session_id
    
    # Create new session if not exists
    new_session_id = session_manager.create_session()
    return new_session_id


# ============== Routes ==============

@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Serve the main chat UI"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/session/init", response_model=InitSessionResponse)
async def init_session():
    """Initialize a new session"""
    try:
        session_id = session_manager.create_session()
        logger.info(f"Session initialized: {session_id}")
        
        return InitSessionResponse(
            session_id=session_id,
            status="initialized",
            message="Session created successfully"
        )
    except Exception as e:
        logger.error(f"Error initializing session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, session_id: str = Depends(get_session_id)):
    """
    Send a message to the agent and get a response
    """
    if not session_manager.session_exists(session_id):
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        # Add user message to history
        session_manager.add_message(session_id, "user", request.message)
        
        logger.info(f"[{session_id}] Processing message: {request.message[:50]}...")
        
        # Run agent with session context
        response = agent.run(
            query=request.message,
            thread_id=session_id
        )
        
        # Add assistant response to history
        session_manager.add_message(session_id, "assistant", response)
        
        logger.info(f"[{session_id}] Response generated successfully")
        
        return ChatResponse(
            response=response,
            session_id=session_id,
            timestamp=str(__import__('datetime').datetime.now().isoformat())
        )
    
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Depends(get_session_id)
):
    """
    Upload a document to the vector store
    """
    if not session_manager.session_exists(session_id):
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        filename = file.filename
        
        # Check if file type is supported
        if not DocumentHandler.is_supported(filename):
            logger.warning(f"Unsupported file type: {filename}")
            return DocumentUploadResponse(
                filename=filename,
                status="error",
                error=f"File type not supported. Supported types: {', '.join(DocumentHandler.SUPPORTED_TYPES)}"
            )
        
        # Check if already uploaded
        documents = session_manager.get_documents_metadata(session_id)
        if documents and filename in documents:
            logger.warning(f"Document already uploaded: {filename}")
            return DocumentUploadResponse(
                filename=filename,
                status="already_exists",
                chunks=documents[filename]["chunks"],
                message="Document already uploaded"
            )
        
        logger.info(f"[{session_id}] Uploading document: {filename}")
        
        # Read file content
        content = await file.read()
        
        # Save file
        file_path = DocumentHandler.save_uploaded_file(content, filename)
        if not file_path:
            return DocumentUploadResponse(
                filename=filename,
                status="error",
                error="Failed to save file"
            )
        
        # Process and add to vector store
        uuids = DocumentHandler.process_and_add_document(file_path, filename, retriever)
        
        # Clean up temp file
        DocumentHandler.cleanup_temp_file(file_path)
        
        if not uuids:
            return DocumentUploadResponse(
                filename=filename,
                status="error",
                error="Failed to process document"
            )
        
        # Store metadata
        session_manager.add_document_metadata(session_id, filename, uuids, len(uuids))
        
        logger.info(f"[{session_id}] Document uploaded successfully: {filename}")
        
        return DocumentUploadResponse(
            filename=filename,
            status="success",
            chunks=len(uuids),
            message=f"Successfully uploaded {filename} with {len(uuids)} chunks"
        )
    
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        return DocumentUploadResponse(
            filename=file.filename,
            status="error",
            error=str(e)
        )


@app.get("/api/documents", response_model=DocumentListResponse)
async def list_documents(session_id: str = Depends(get_session_id)):
    """
    Get list of uploaded documents for a session
    """
    if not session_manager.session_exists(session_id):
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        docs_meta = session_manager.get_documents_metadata(session_id) or {}
        
        documents = [
            DocumentMetadata(
                filename=filename,
                uuids=meta["uuids"],
                upload_time=meta["upload_time"],
                chunks=meta["chunks"]
            )
            for filename, meta in docs_meta.items()
        ]
        
        total_chunks = sum(doc.chunks for doc in documents)
        
        return DocumentListResponse(
            documents=documents,
            total_chunks=total_chunks
        )
    
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/documents/{filename}", response_model=DocumentDeleteResponse)
async def delete_document(
    filename: str,
    session_id: str = Depends(get_session_id)
):
    """
    Delete a document from the vector store
    """
    if not session_manager.session_exists(session_id):
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        uuids = session_manager.get_document_uuids(session_id, filename)
        
        if not uuids:
            logger.warning(f"Document not found: {filename}")
            return DocumentDeleteResponse(
                filename=filename,
                status="not_found",
                message="Document not found in this session"
            )
        
        # Delete from vector store
        success = DocumentHandler.delete_document(retriever, uuids)
        
        if not success:
            return DocumentDeleteResponse(
                filename=filename,
                status="error",
                message="Failed to delete document from vector store"
            )
        
        # Remove from session metadata
        session_manager.remove_document(session_id, filename)
        
        logger.info(f"[{session_id}] Document deleted: {filename}")
        
        return DocumentDeleteResponse(
            filename=filename,
            status="success",
            message=f"Successfully deleted {filename}"
        )
    
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return DocumentDeleteResponse(
            filename=filename,
            status="error",
            message=str(e)
        )


@app.get("/api/chat/history", response_model=List[ChatMessage])
async def get_chat_history(session_id: str = Depends(get_session_id)):
    """
    Get chat history for a session
    """
    if not session_manager.session_exists(session_id):
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        chat_history = session_manager.get_chat_history(session_id) or []
        
        return [
            ChatMessage(role=msg["role"], content=msg["content"])
            for msg in chat_history
        ]
    
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/clear", response_model=ClearChatResponse)
async def clear_chat_history(session_id: str = Depends(get_session_id)):
    """
    Clear chat history for a session
    """
    if not session_manager.session_exists(session_id):
        logger.warning(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session_manager.clear_chat_history(session_id)
        
        logger.info(f"[{session_id}] Chat history cleared")
        
        return ClearChatResponse(
            session_id=session_id,
            status="success",
            message="Chat history cleared successfully"
        )
    
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "AI Agent with RAG",
        "version": "1.0.0"
    }


# ============== Error Handling ==============
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            status="error",
            message=exc.detail,
            error_type="HTTPException"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            status="error",
            message="Internal server error",
            error_type=type(exc).__name__
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    logger.info("Starting AI Agent with RAG FastAPI server...")
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
