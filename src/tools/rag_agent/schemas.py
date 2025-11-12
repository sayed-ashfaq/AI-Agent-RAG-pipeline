from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class FileType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    XLSX = "xlsx"
    CSV = "csv"
    IMAGE = "image"
    DATABASE = "database"
    MARKDOWN = "md"
    PPT = "ppt"
    PPTX = "pptx"
    DB = "db"
    SQLITE = "sqlite"
    SQLITE3 = "sqlite3"

class ChatScope(str, Enum):
    SINGLE_DOCUMENT = "single"
    MULTIPLE_DOCUMENTS = "multiple"
    ALL_DOCUMENTS = "all"

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    file_type: FileType
    processed_at: datetime
    chunk_count: int
    has_images: bool = False
    image_count: int = 0

class BatchUploadResponse(BaseModel):
    total_files: int
    successful_uploads: List[DocumentInfo]
    failed_uploads: List[Dict[str, str]]
    collection_id: str

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.now)

class ChatRequest(BaseModel):
    document_id: Optional[str] = None
    document_ids: Optional[List[str]] = None
    collection_id: Optional[str] = None
    question: str
    scope: ChatScope = ChatScope.SINGLE_DOCUMENT

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence_score: Optional[float] = None
    response_time_ms: int