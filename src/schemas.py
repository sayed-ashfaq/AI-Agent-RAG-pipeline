"""
Pydantic schemas for FastAPI application
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class ChatMessage(BaseModel):
    """Schema for a single chat message"""
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    """Schema for incoming chat requests"""
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="User's message")


class ChatResponse(BaseModel):
    """Schema for outgoing chat responses"""
    response: str = Field(..., description="Assistant's response")
    session_id: str = Field(..., description="Session ID")
    timestamp: str = Field(..., description="Response timestamp")


class DocumentMetadata(BaseModel):
    """Schema for document metadata"""
    filename: str
    uuids: List[str]
    upload_time: str
    chunks: int


class SessionInfo(BaseModel):
    """Schema for session information"""
    session_id: str
    chat_history: List[ChatMessage]
    documents_metadata: Dict[str, DocumentMetadata]
    created_at: str
    last_activity: str


class InitSessionResponse(BaseModel):
    """Schema for session initialization response"""
    session_id: str
    status: str = "initialized"
    message: str = "Session created successfully"


class DocumentUploadResponse(BaseModel):
    """Schema for document upload response"""
    filename: str
    status: str
    chunks: int = 0
    message: str = ""
    error: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Schema for listing documents in a session"""
    documents: List[DocumentMetadata]
    total_chunks: int


class DocumentDeleteResponse(BaseModel):
    """Schema for document deletion response"""
    filename: str
    status: str
    message: str


class ClearChatResponse(BaseModel):
    """Schema for clearing chat history response"""
    session_id: str
    status: str
    message: str


class ErrorResponse(BaseModel):
    """Schema for error responses"""
    status: str = "error"
    message: str
    error_type: Optional[str] = None
