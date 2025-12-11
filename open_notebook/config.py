import os
from typing import Optional


# ROOT DATA FOLDER
DATA_FOLDER = "./data"

# LANGGRAPH CHECKPOINT FILE
sqlite_folder = f"{DATA_FOLDER}/sqlite-db"
os.makedirs(sqlite_folder, exist_ok=True)
LANGGRAPH_CHECKPOINT_FILE = f"{sqlite_folder}/checkpoints.sqlite"

# UPLOADS FOLDER
UPLOADS_FOLDER = f"{DATA_FOLDER}/uploads"
os.makedirs(UPLOADS_FOLDER, exist_ok=True)

# TIKTOKEN CACHE FOLDER
TIKTOKEN_CACHE_DIR = f"{DATA_FOLDER}/tiktoken-cache"
os.makedirs(TIKTOKEN_CACHE_DIR, exist_ok=True)

# FILE UPLOAD LIMITS
# Maximum file upload size in bytes (default: 100MB)
# Can be overridden via MAX_UPLOAD_SIZE environment variable
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "104857600"))  # 100MB default
MAX_UPLOAD_SIZE_MB = MAX_UPLOAD_SIZE / (1024 * 1024)


# QDRANT CONFIGURATION
class QdrantConfig:
    """Configuration for Qdrant vector database service."""
    
    host: str = os.getenv("QDRANT_HOST", "qdrant")
    port: int = int(os.getenv("QDRANT_PORT", "6333"))
    api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    timeout: int = int(os.getenv("QDRANT_TIMEOUT", "60"))
    prefer_grpc: bool = os.getenv("QDRANT_PREFER_GRPC", "false").lower() == "true"
    store_content_in_payload: bool = os.getenv("QDRANT_STORE_CONTENT", "true").lower() == "true"
    
    @classmethod
    def get_grpc_port(cls) -> int:
        """Get gRPC port (typically HTTP port + 1)."""
        return cls.port + 1 if cls.prefer_grpc else cls.port