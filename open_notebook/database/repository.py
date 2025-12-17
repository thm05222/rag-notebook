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
]


def parse_record_ids(obj: Any) -> Any:
    """Recursively parse and convert RecordIDs into strings."""
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
    """Execute a SurrealQL query and return the results"""

    async with db_connection() as connection:
        try:
            result = parse_record_ids(await connection.query(query_str, vars))
            if isinstance(result, str):
                # Check if the error message indicates a not found condition
                error_msg_lower = result.lower()
                if "not found" in error_msg_lower or "does not exist" in error_msg_lower:
                    raise NotFoundError(result)
                raise DatabaseOperationError(result)
            return result
        except (NotFoundError, DatabaseOperationError):
            # Re-raise our custom exceptions
            raise
        except Exception as e:
            logger.error(f"Query: {query_str[:200]} vars: {vars}")
            logger.exception(e)
            # Check error message to determine exception type
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
