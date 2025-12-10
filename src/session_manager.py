"""
Session management for FastAPI application
Handles session state, chat history, and document metadata
"""
from typing import Dict, Optional, List
from datetime import datetime
import uuid
from dataclasses import dataclass, field, asdict
from threading import Lock

from custom_logger import GLOBAL_LOGGER as logger


@dataclass
class Session:
    """Represents a user session"""
    session_id: str
    chat_history: List[dict] = field(default_factory=list)
    documents_metadata: Dict[str, dict] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self):
        """Convert session to dictionary"""
        return asdict(self)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now().isoformat()


class SessionManager:
    """Manages user sessions"""
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = Lock()
    
    def create_session(self) -> str:
        """Create a new session and return session ID"""
        session_id = str(uuid.uuid4())
        with self._lock:
            self._sessions[session_id] = Session(session_id=session_id)
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID"""
        with self._lock:
            return self._sessions.get(session_id)
    
    def session_exists(self, session_id: str) -> bool:
        """Check if session exists"""
        with self._lock:
            return session_id in self._sessions
    
    def add_message(self, session_id: str, role: str, content: str) -> bool:
        """Add a message to session chat history"""
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False
        
        with self._lock:
            session.chat_history.append({
                "role": role,
                "content": content
            })
            session.update_activity()
        
        logger.debug(f"Added message to session {session_id}")
        return True
    
    def get_chat_history(self, session_id: str) -> Optional[List[dict]]:
        """Get chat history for a session"""
        session = self.get_session(session_id)
        if not session:
            return None
        return session.chat_history
    
    def add_document_metadata(
        self, 
        session_id: str, 
        filename: str, 
        uuids: List[str], 
        chunks: int
    ) -> bool:
        """Add document metadata to session"""
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False
        
        with self._lock:
            session.documents_metadata[filename] = {
                "uuids": uuids,
                "upload_time": datetime.now().isoformat(),
                "chunks": chunks
            }
            session.update_activity()
        
        logger.info(f"Added document metadata for {filename} in session {session_id}")
        return True
    
    def get_documents_metadata(self, session_id: str) -> Optional[Dict[str, dict]]:
        """Get all document metadata for a session"""
        session = self.get_session(session_id)
        if not session:
            return None
        return session.documents_metadata
    
    def get_document_uuids(self, session_id: str, filename: str) -> Optional[List[str]]:
        """Get UUIDs for a specific document"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        doc_meta = session.documents_metadata.get(filename)
        return doc_meta["uuids"] if doc_meta else None
    
    def remove_document(self, session_id: str, filename: str) -> bool:
        """Remove document metadata from session"""
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False
        
        with self._lock:
            if filename in session.documents_metadata:
                del session.documents_metadata[filename]
                session.update_activity()
                logger.info(f"Removed document {filename} from session {session_id}")
                return True
        
        logger.warning(f"Document not found: {filename} in session {session_id}")
        return False
    
    def clear_chat_history(self, session_id: str) -> bool:
        """Clear chat history for a session"""
        session = self.get_session(session_id)
        if not session:
            logger.warning(f"Session not found: {session_id}")
            return False
        
        with self._lock:
            session.chat_history = []
            session.update_activity()
        
        logger.info(f"Cleared chat history for session {session_id}")
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Deleted session: {session_id}")
                return True
        
        logger.warning(f"Session not found for deletion: {session_id}")
        return False
    
    def get_all_sessions(self) -> Dict[str, Session]:
        """Get all sessions (for debugging/monitoring)"""
        with self._lock:
            return dict(self._sessions)


# Global session manager instance
session_manager = SessionManager()
