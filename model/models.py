from pydantic import BaseModel, Field
from typing import Annotated
from enum import Enum

class ChatAnswer(BaseModel):
    answer: Annotated[str, Field(min_length=1, max_length=4096)]


class PromptType(str, Enum):
    CONTEXTUALIZE_QUESTION = "some variable or prompt name:"
    CONTEXT_QA = "PROMPT NAME" 


class UploadResponse(BaseModel):
    session_id : str
    indexed: bool 
    messages: str | None = None

