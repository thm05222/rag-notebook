"""
Error message utilities for cleaning and formatting error messages.

This module provides functions to clean error messages, extract key information,
and provide user-friendly error messages for common error types.
"""
import re
import traceback
from typing import Any, Optional

from loguru import logger


def clean_error_message(error: Exception, max_length: int = 500) -> str:
    """
    Clean error message by removing traceback and extracting key information.
    
    Provides user-friendly error messages for common error types.
    
    Args:
        error: Exception object
        max_length: Maximum length of error message (default: 500)
        
    Returns:
        Cleaned, user-friendly error message
    """
    error_type = type(error).__name__
    error_str = str(error)
    
    # Handle common error types with friendly messages
    friendly_message = _get_friendly_message(error_type, error_str, error)
    
    if friendly_message:
        return friendly_message[:max_length]
    
    # Extract key information from error message
    # Remove traceback if present
    if "Traceback" in error_str or "\n  File" in error_str:
        # Extract the last line (usually the actual error)
        lines = error_str.split("\n")
        for line in reversed(lines):
            if line.strip() and not line.strip().startswith("File"):
                error_str = line.strip()
                break
    
    # Remove common technical details
    error_str = re.sub(r"at 0x[0-9a-f]+", "", error_str)
    error_str = re.sub(r"line \d+", "", error_str)
    error_str = re.sub(r"File ['\"].+['\"]", "", error_str)
    
    # Clean up whitespace
    error_str = " ".join(error_str.split())
    
    # If still too long, truncate intelligently
    if len(error_str) > max_length:
        # Try to keep the beginning (usually more informative)
        error_str = error_str[:max_length - 3] + "..."
    
    return error_str


def _get_friendly_message(
    error_type: str, 
    error_str: str, 
    error: Exception
) -> Optional[str]:
    """
    Get user-friendly message for common error types.
    
    Args:
        error_type: Name of the exception class
        error_str: String representation of the error
        error: The exception object
        
    Returns:
        User-friendly message or None if no mapping exists
    """
    error_lower = error_str.lower()
    
    # OCR/Image processing errors
    if "ocr" in error_lower or "tesseract" in error_lower or "image" in error_lower:
        if "cannot" in error_lower or "unable" in error_lower or "failed" in error_lower:
            return "無法識別文字，請確認 PDF 是否清晰或嘗試上傳純文字版本"
    
    # PageIndex/Structure building errors
    if "pageindex" in error_lower or "structure" in error_lower:
        if "timeout" in error_lower or "timed out" in error_lower:
            return "文件結構過於複雜或處理超時，建議將文件拆分成較小的章節重新上傳"
        if "failed" in error_lower or "error" in error_lower:
            return "無法建立文件結構索引，請檢查文件格式是否正確"
    
    # Token/Model limit errors
    if "token" in error_lower and ("limit" in error_lower or "exceed" in error_lower or "too long" in error_lower):
        return "文件長度超過模型限制，請嘗試將文件拆分或使用較短的內容"
    
    # Qdrant/Vector database errors
    if "qdrant" in error_lower or "vector" in error_lower:
        if "connection" in error_lower or "connect" in error_lower or "refused" in error_lower:
            return "向量資料庫連接失敗，請檢查服務狀態"
        if "dimension" in error_lower or "dim" in error_lower:
            return "向量維度不匹配，請重新建立索引"
    
    # SurrealDB errors
    if "surrealdb" in error_lower or "database" in error_lower:
        if "connection" in error_lower or "connect" in error_lower:
            return "資料庫連接失敗，請檢查服務狀態"
        if "not found" in error_lower:
            return "找不到指定的資料，可能已被刪除"
    
    # File/IO errors
    if "file" in error_lower or "io" in error_lower:
        if "not found" in error_lower or "no such file" in error_lower:
            return "找不到指定的檔案，請確認檔案是否存在"
        if "permission" in error_lower:
            return "檔案權限不足，無法讀取檔案"
    
    # Network/HTTP errors
    if "http" in error_lower or "network" in error_lower or "connection" in error_lower:
        if "timeout" in error_lower:
            return "網路請求超時，請稍後重試"
        if "refused" in error_lower or "unreachable" in error_lower:
            return "無法連接到伺服器，請檢查網路連接"
    
    # Generic ValueError with dimension info
    if error_type == "ValueError" and "dimension" in error_lower:
        return "向量維度不匹配，這通常發生在切換嵌入模型時。請刪除集合或使用「重建嵌入」來遷移資料"
    
    # If no specific mapping, return None to use default cleaning
    return None


def extract_error_summary(error: Exception) -> dict[str, Any]:
    """
    Extract summary information from an error.
    
    Args:
        error: Exception object
        
    Returns:
        Dictionary with error summary information
    """
    return {
        "type": type(error).__name__,
        "message": str(error),
        "friendly_message": clean_error_message(error),
    }

