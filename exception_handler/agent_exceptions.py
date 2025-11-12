import sys
import traceback
import os
from typing import Optional, Tuple, Type, cast


class AgentRagException(Exception):
    """
    Base enterprise exception for AgentRag system.
    Captures and formats detailed context such as file name, line number, and traceback.
    """

    def __init__(self, message: str | BaseException, details: Optional[object] = None):
        # Normalize message
        self.error_message = str(message)

        # Extract exception info
        exc_type: Optional[Type[BaseException]] = None
        exc_value: Optional[BaseException] = None
        exc_tb = None

        if details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        elif hasattr(details, "exc_info"):
            exc_info_obj = cast(sys, details)
            exc_type, exc_value, exc_tb = exc_info_obj.exc_info()
        elif isinstance(details, BaseException):
            exc_type, exc_value, exc_tb = type(details), details, details.__traceback__
        else:
            exc_type, exc_value, exc_tb = sys.exc_info()

        # Find last frame for accurate location
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1
        if os.getenv("DEBUG_MODE", "false").lower() == 'true':
            self.traceback_str = (
                "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
                if exc_type and exc_tb
                else ""
            )
        else:
            self.traceback_str = ""

        super().__init__(self.__str__())


    # Defines how the exception appears in log

    def __str__(self) -> str:
        base = f"Error in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"
        return f"{base}\n{self.traceback_str}" if self.traceback_str else base

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(file='{self.file_name}', "
            f"lineno={self.lineno}, message='{self.error_message}')"
        )


# ---- Example Custom Exceptions ----
class WeatherAPIError(AgentRagException):
    """Raised when weather API calls fail."""
    pass

class FileProcessingError(AgentRagException):
    """Raised when there is document processing error"""
    pass


class DataProcessingError(AgentRagException):
    """Raised when RAG data pipeline processing fails."""
    pass

class QdrantServiceError(AgentRagException):
    """Raised when Qdrant service fails."""
    pass

class WorkflowError(AgentRagException):
    """Raised when workflow fails."""
    pass

class RetrieverError(AgentRagException):
    """Raised when retriever fails."""
    pass


# ---- Example Usage ----
if __name__ == "__main__":
    try:
        a = int("Str")  # Simulate failure
    except Exception as e:
        raise WeatherAPIError("Conversion failed", e)
