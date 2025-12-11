class OpenNotebookError(Exception):
    """Base exception class for Open Notebook errors."""

    pass


class DatabaseOperationError(OpenNotebookError):
    """Raised when a database operation fails."""

    pass


class UnsupportedTypeException(OpenNotebookError):
    """Raised when an unsupported type is provided."""

    pass


class InvalidInputError(OpenNotebookError):
    """Raised when invalid input is provided."""

    pass


class NotFoundError(OpenNotebookError):
    """Raised when a requested resource is not found."""

    pass


class AuthenticationError(OpenNotebookError):
    """Raised when there's an authentication problem."""

    pass


class ConfigurationError(OpenNotebookError):
    """Raised when there's a configuration problem."""

    pass


class ExternalServiceError(OpenNotebookError):
    """Raised when an external service (e.g., AI model) fails."""

    pass


class RateLimitError(OpenNotebookError):
    """Raised when a rate limit is exceeded."""

    pass


class FileOperationError(OpenNotebookError):
    """Raised when a file operation fails."""

    pass


class NetworkError(OpenNotebookError):
    """Raised when a network operation fails."""

    pass


class NoTranscriptFound(OpenNotebookError):
    """Raised when no transcript is found for a video."""

    pass


# Agentic RAG related exceptions
class ToolExecutionError(OpenNotebookError):
    """Raised when a tool execution fails."""

    pass


class ToolNotFoundError(OpenNotebookError):
    """Raised when a requested tool is not found."""

    pass


class EvaluationError(OpenNotebookError):
    """Raised when evaluation service fails."""

    pass


class IterationLimitError(OpenNotebookError):
    """Raised when maximum iteration limit is reached."""

    pass


class TokenLimitError(OpenNotebookError):
    """Raised when token usage limit is exceeded."""

    pass


class TimeoutError(OpenNotebookError):
    """Raised when execution timeout is exceeded."""

    pass


class CircularReasoningError(OpenNotebookError):
    """Raised when circular reasoning is detected."""

    pass