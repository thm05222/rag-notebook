"""
Database repository functions for SurrealDB operations.

Compatibility Notes:
- Designed for SurrealDB v2
- Uses UPSERT for create-or-update operations (v2 requirement)
- UPDATE only modifies existing records (v2 behavior)
- Error handling adapted for v2 response formats
- Handles various error response formats (string, dict, list with error fields)
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypeVar, Union

from loguru import logger
from surrealdb import AsyncSurreal, RecordID  # type: ignore

from open_notebook.exceptions import DatabaseOperationError, NotFoundError

# Import connection pool functions
from open_notebook.database.connection_pool import (
    db_connection as _db_connection,
    close_pool,
    get_database_url,
    get_database_password,
)

T = TypeVar("T", Dict[str, Any], List[Dict[str, Any]])

# Re-export for backward compatibility
__all__ = [
    "db_connection",
    "repo_query",
    "repo_create",
    "repo_relate",
    "repo_upsert",
    "repo_update",
    "repo_delete",
    "repo_insert",
    "parse_record_ids",
    "ensure_record_id",
    "repo_add_message",
    "repo_get_chat_history",
]


def parse_record_ids(obj: Any) -> Any:
    """Recursively parse and convert RecordIDs into strings.
    
    Handles None, dict, list, and RecordID types safely.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: parse_record_ids(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [parse_record_ids(item) for item in obj]
    elif isinstance(obj, RecordID):
        return str(obj)
    return obj


def ensure_record_id(value: Union[str, RecordID]) -> RecordID:
    """Ensure a value is a RecordID."""
    if isinstance(value, RecordID):
        return value
    return RecordID.parse(value)


# Use connection pool instead of creating new connections
db_connection = _db_connection


async def repo_query(
    query_str: str, vars: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Execute a SurrealQL query and return the results.
    
    Handles SurrealDB v2 response formats including error detection
    in various formats (string, dict, list with error fields).
    """
    async with db_connection() as connection:
        try:
            result = await connection.query(query_str, vars)
            
            # Handle None or empty results
            if result is None:
                logger.warning(f"Query returned None: {query_str[:200]}")
                return []
            
            # Parse result and check for errors
            # parse_record_ids handles None by returning None, so check before parsing
            parsed_result = parse_record_ids(result)
            
            # SurrealDB v2 may return errors in different formats
            # Check if result is an error string
            if isinstance(parsed_result, str):
                error_msg_lower = parsed_result.lower()
                if "not found" in error_msg_lower or "does not exist" in error_msg_lower:
                    raise NotFoundError(parsed_result)
                raise DatabaseOperationError(parsed_result)
            
            # Check if result is a list containing error information
            if isinstance(parsed_result, list):
                if len(parsed_result) == 0:
                    return []
                first_item = parsed_result[0]
                if isinstance(first_item, dict):
                    # Check for error fields in the result (SurrealDB v2 error format)
                    if "error" in first_item or "code" in first_item:
                        error_msg = str(first_item.get("error", first_item.get("code", "")))
                        error_msg_lower = error_msg.lower()
                        if "not found" in error_msg_lower or "does not exist" in error_msg_lower:
                            raise NotFoundError(error_msg)
                        raise DatabaseOperationError(error_msg)
            
            # Handle None after parsing (shouldn't happen, but defensive)
            if parsed_result is None:
                logger.warning(f"Parsed result is None: {query_str[:200]}")
                return []
            
            return parsed_result
            
        except (NotFoundError, DatabaseOperationError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Query failed: {query_str[:200]} | vars: {vars} | Error: {e}")
            logger.exception(e)
            error_msg = str(e).lower()
            if "not found" in error_msg or "does not exist" in error_msg:
                raise NotFoundError(f"Record not found: {str(e)}") from e
            elif "connection" in error_msg or "timeout" in error_msg or "network" in error_msg:
                raise DatabaseOperationError(f"Database connection error: {str(e)}") from e
            else:
                raise DatabaseOperationError(f"Database query failed: {str(e)}") from e


async def repo_create(table: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new record in the specified table"""
    # Remove 'id' attribute if it exists in data
    data.pop("id", None)
    data["created"] = datetime.now(timezone.utc)
    data["updated"] = datetime.now(timezone.utc)
    try:
        async with db_connection() as connection:
            return parse_record_ids(await connection.insert(table, data))
    except (NotFoundError, DatabaseOperationError):
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        logger.exception(e)
        error_msg = str(e).lower()
        if "connection" in error_msg or "timeout" in error_msg or "network" in error_msg:
            raise DatabaseOperationError(f"Database connection error while creating record: {str(e)}") from e
        else:
            raise DatabaseOperationError(f"Failed to create record: {str(e)}") from e


async def repo_relate(
    source: str, relationship: str, target: str, data: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Create a relationship between two records with optional data"""
    if data is None:
        data = {}
    query = f"RELATE {source}->{relationship}->{target} CONTENT $data;"
    # logger.debug(f"Relate query: {query}")

    return await repo_query(
        query,
        {
            "data": data,
        },
    )


async def repo_upsert(
    table: str, id: Optional[str], data: Dict[str, Any], add_timestamp: bool = False
) -> List[Dict[str, Any]]:
    """Create or update a record in the specified table"""
    data.pop("id", None)
    if add_timestamp:
        data["updated"] = datetime.now(timezone.utc)
    query = f"UPSERT {id if id else table} MERGE $data;"
    return await repo_query(query, {"data": data})


async def repo_update(
    table: str, id: str, data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Update an existing record by table and id"""
    # If id already contains the table name, use it as is
    try:
        if isinstance(id, RecordID) or (":" in id and id.startswith(f"{table}:")):
            record_id = id
        else:
            record_id = f"{table}:{id}"
        data.pop("id", None)
        if "created" in data and isinstance(data["created"], str):
            data["created"] = datetime.fromisoformat(data["created"])
        data["updated"] = datetime.now(timezone.utc)
        query = f"UPDATE {record_id} MERGE $data;"
        # logger.debug(f"Update query: {query}")
        result = await repo_query(query, {"data": data})
        # Check if update returned empty result (record not found)
        if not result or (isinstance(result, list) and len(result) == 0):
            raise NotFoundError(f"Record {record_id} not found for update")
        # if isinstance(result, list):
        #     return [_return_data(item) for item in result]
        return parse_record_ids(result)
    except (NotFoundError, DatabaseOperationError):
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg or "does not exist" in error_msg:
            raise NotFoundError(f"Record not found for update: {str(e)}") from e
        elif "connection" in error_msg or "timeout" in error_msg or "network" in error_msg:
            raise DatabaseOperationError(f"Database connection error while updating record: {str(e)}") from e
        else:
            raise DatabaseOperationError(f"Failed to update record: {str(e)}") from e


async def repo_get_news_by_jota_id(jota_id: str) -> Dict[str, Any]:
    try:
        results = await repo_query(
            "SELECT * omit embedding FROM news where jota_id=$jota_id",
            {"jota_id": jota_id},
        )
        if not results or len(results) == 0:
            raise NotFoundError(f"News record with jota_id {jota_id} not found")
        return parse_record_ids(results)
    except (NotFoundError, DatabaseOperationError):
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        logger.exception(e)
        error_msg = str(e).lower()
        if "not found" in error_msg or "does not exist" in error_msg:
            raise NotFoundError(f"Record not found: {str(e)}") from e
        elif "connection" in error_msg or "timeout" in error_msg or "network" in error_msg:
            raise DatabaseOperationError(f"Database connection error while fetching record: {str(e)}") from e
        else:
            raise DatabaseOperationError(f"Failed to fetch record: {str(e)}") from e


async def repo_delete(record_id: Union[str, RecordID]):
    """Delete a record by record id"""

    try:
        async with db_connection() as connection:
            result = await connection.delete(ensure_record_id(record_id))
            # Check if delete returned empty result (record not found)
            if result is None or (isinstance(result, list) and len(result) == 0):
                raise NotFoundError(f"Record {record_id} not found for deletion")
            return result
    except (NotFoundError, DatabaseOperationError):
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        logger.exception(e)
        error_msg = str(e).lower()
        if "not found" in error_msg or "does not exist" in error_msg:
            raise NotFoundError(f"Record not found for deletion: {str(e)}") from e
        elif "connection" in error_msg or "timeout" in error_msg or "network" in error_msg:
            raise DatabaseOperationError(f"Database connection error while deleting record: {str(e)}") from e
        else:
            raise DatabaseOperationError(f"Failed to delete record: {str(e)}") from e


async def repo_insert(
    table: str, data: List[Dict[str, Any]], ignore_duplicates: bool = False
) -> List[Dict[str, Any]]:
    """Create a new record in the specified table"""
    try:
        async with db_connection() as connection:
            return parse_record_ids(await connection.insert(table, data))
    except (NotFoundError, DatabaseOperationError):
        # Re-raise our custom exceptions
        raise
    except Exception as e:
        if ignore_duplicates and "already contains" in str(e):
            return []
        logger.exception(e)
        error_msg = str(e).lower()
        if "connection" in error_msg or "timeout" in error_msg or "network" in error_msg:
            raise DatabaseOperationError(f"Database connection error while inserting record: {str(e)}") from e
        else:
            raise DatabaseOperationError(f"Failed to create record: {str(e)}") from e


async def repo_add_message(
    session_id: str, 
    role: str, 
    content: str, 
    thinking_process: Optional[Dict[str, Any]] = None,
    reasoning_content: Optional[str] = None
) -> Dict[str, Any]:
    """
    儲存單條對話訊息到 SurrealDB
    
    Args:
        session_id: Chat session ID (支援 'chat_session:xxx' 或純 'xxx' 格式)
        role: 角色 ('user' 或 'ai')
        content: 訊息內容
        thinking_process: AI 的思考過程 (AgentThinkingProcess 的 dict 格式)
        reasoning_content: 純文字版思考過程（用於簡化顯示）
    """
    clean_id = session_id.split(":")[-1] if ":" in session_id else session_id
    
    sql = """
    CREATE message SET 
        session_id = type::thing('chat_session', $session_id),
        role = $role,
        content = $content,
        thinking_process = $thinking_process,
        reasoning_content = $reasoning_content,
        created_at = time::now();
    """
    try:
        result = await repo_query(sql, {
            "session_id": clean_id, 
            "role": role, 
            "content": content,
            "thinking_process": thinking_process,
            "reasoning_content": reasoning_content
        })
        return result[0] if result else {}
    except Exception as e:
        logger.error(f"Failed to add message: {e}")
        raise DatabaseOperationError(f"Failed to add message: {str(e)}") from e


async def repo_get_chat_history(session_id: str) -> List[Dict[str, Any]]:
    """
    獲取特定 Session 的所有對話紀錄，按時間排序
    
    Args:
        session_id: Chat session ID (支援 'chat_session:xxx' 或純 'xxx' 格式)
    
    Returns:
        List of message dicts with keys: id, session_id, role, content, thinking_process, reasoning_content, created_at
    """
    clean_id = session_id.split(":")[-1] if ":" in session_id else session_id
    
    sql = """
    SELECT * FROM message 
    WHERE session_id = type::thing('chat_session', $session_id)
    ORDER BY created_at ASC;
    """
    try:
        return await repo_query(sql, {"session_id": clean_id})
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        # Return empty list instead of raising to avoid breaking the chat flow
        return []
