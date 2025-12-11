#!/usr/bin/env python3
"""
Startup script for Open Notebook API server.
"""

import os
import sys
from pathlib import Path

import uvicorn

# Add the current directory to Python path so imports work
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

if __name__ == "__main__":
    # Default configuration
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "5055"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"

    print(f"Starting Open Notebook API server on {host}:{port}")
    print(f"Reload mode: {reload}")

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
        reload_dirs=[str(current_dir)] if reload else None,
        timeout_keep_alive=300,  # 保持連接活躍 300 秒（5 分鐘）
        timeout_graceful_shutdown=30,  # 優雅關閉超時 30 秒
    )
