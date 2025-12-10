"""
Document handling utilities for FastAPI
"""
from pathlib import Path
from typing import Optional, List
from uuid import uuid4
import shutil

from src.tools.rag_agent.document_loader import DocumentService
from src.tools.rag_agent.retriever import Retriever
from custom_logger import GLOBAL_LOGGER as logger


class DocumentHandler:
    """Handles document upload, processing, and deletion"""
    
    SUPPORTED_TYPES = ['pdf', 'txt', 'docx', 'doc', 'md']
    TEMP_DIR = Path("temp_uploads")
    
    @staticmethod
    def ensure_temp_dir():
        """Ensure temp directory exists"""
        DocumentHandler.TEMP_DIR.mkdir(exist_ok=True)
    
    @staticmethod
    def get_file_extension(filename: str) -> str:
        """Extract file extension without the dot"""
        return Path(filename).suffix[1:].lower()
    
    @staticmethod
    def is_supported(filename: str) -> bool:
        """Check if file type is supported"""
        ext = DocumentHandler.get_file_extension(filename)
        return ext in DocumentHandler.SUPPORTED_TYPES
    
    @staticmethod
    def save_uploaded_file(file_content: bytes, filename: str) -> Optional[str]:
        """Save uploaded file to temporary location and return path"""
        try:
            DocumentHandler.ensure_temp_dir()
            file_path = DocumentHandler.TEMP_DIR / filename
            
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            logger.info(f"File saved temporarily at: {file_path}")
            return str(file_path)
        
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            return None
    
    @staticmethod
    def process_and_add_document(
        file_path: str, 
        filename: str, 
        retriever: Retriever
    ) -> Optional[List[str]]:
        """Process document and add to vector store, returns list of document UUIDs"""
        try:
            file_extension = DocumentHandler.get_file_extension(filename)
            
            if file_extension not in DocumentHandler.SUPPORTED_TYPES:
                logger.warning(f"File type '.{file_extension}' may not be supported.")
                return None
            
            document_service = DocumentService()
            retrieved_docs = document_service.process_single_file(file_path, file_extension)
            
            if not retrieved_docs:
                logger.warning(f"No documents retrieved from {filename}")
                return None
            
            # Generate UUIDs and add to vector store
            uuids = [str(uuid4()) for _ in range(len(retrieved_docs))]
            retriever.vector_store.add_documents(documents=retrieved_docs, ids=uuids)
            
            logger.info(f"Successfully added {len(retrieved_docs)} chunks from {filename} to vector store")
            
            return uuids
        
        except Exception as e:
            logger.error(f"Error processing document {filename}: {e}")
            return None
    
    @staticmethod
    def delete_document(retriever: Retriever, uuids: List[str]) -> bool:
        """Delete document chunks from vector store"""
        try:
            for uuid in uuids:
                retriever.delete_document(uuid)
            
            logger.info(f"Successfully deleted {len(uuids)} document chunks from vector store")
            return True
        
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    @staticmethod
    def cleanup_temp_file(file_path: str):
        """Clean up temporary file"""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.debug(f"Cleaned up temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Error cleaning up temp file {file_path}: {e}")
