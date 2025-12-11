"""
Database connection pool for SurrealDB.

This module provides a simple connection pool to reuse database connections
and avoid creating a new connection for every query.

Supports multiple event loops by maintaining a separate connection for each event loop.
"""
import asyncio
import os
import threading
from contextlib import asynccontextmanager
from typing import Dict, Optional

from loguru import logger
from surrealdb import AsyncSurreal

# Connection pool per event loop (keyed by event loop id)
_connection_pools: Dict[int, AsyncSurreal] = {}
# Thread-safe lock to protect the connection pool dictionary
_pool_lock = threading.Lock()


def get_database_url():
    """Get database URL with backward compatibility"""
    surreal_url = os.getenv("SURREAL_URL")
    if surreal_url:
        return surreal_url

    # Fallback to old format - WebSocket URL format
    address = os.getenv("SURREAL_ADDRESS", "localhost")
    port = os.getenv("SURREAL_PORT", "8000")
    return f"ws://{address}/rpc:{port}"


def get_database_password():
    """Get password with backward compatibility"""
    return os.getenv("SURREAL_PASSWORD") or os.getenv("SURREAL_PASS")


async def _get_or_create_connection() -> AsyncSurreal:
    """
    Get or create a database connection from the pool for the current event loop.
    
    Each event loop gets its own connection to avoid "attached to a different loop" errors.
    Returns a connection that should be reused across queries in the same event loop.
    """
    global _connection_pools
    
    # Get the current event loop ID
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        # No event loop running, create a new one (shouldn't happen in normal operation)
        loop = asyncio.new_event_loop()
        loop_id = id(loop)
        logger.warning("No event loop running, creating new one for database connection")
    
    # Check if we already have a connection for this event loop
    with _pool_lock:
        if loop_id in _connection_pools:
            return _connection_pools[loop_id]
    
    # Create a new connection for this event loop
    # Note: We don't hold the lock during connection creation to avoid blocking
    logger.debug(f"Creating new database connection for event loop {loop_id}")
    connection = AsyncSurreal(get_database_url())
    await connection.signin(
        {
            "username": os.environ.get("SURREAL_USER"),
            "password": get_database_password(),
        }
    )
    await connection.use(
        os.environ.get("SURREAL_NAMESPACE"),
        os.environ.get("SURREAL_DATABASE")
    )
    logger.debug(f"Database connection established for event loop {loop_id}")
    
    # Store the connection for this event loop
    with _pool_lock:
        _connection_pools[loop_id] = connection
    
    return connection


async def close_pool():
    """Close all connections in the pool."""
    global _connection_pools
    
    with _pool_lock:
        connections_to_close = list(_connection_pools.values())
        _connection_pools.clear()
    
    # Close all connections outside the lock to avoid blocking
    for connection in connections_to_close:
        try:
            await connection.close()
        except Exception as e:
            logger.warning(f"Error closing database connection: {e}")
    
    if connections_to_close:
        logger.info(f"Closed {len(connections_to_close)} database connection(s)")


@asynccontextmanager
async def db_connection():
    """
    Get a database connection from the pool.
    
    This context manager reuses a shared connection instead of creating
    a new one for each query, improving performance and avoiding connection
    exhaustion.
    """
    connection = await _get_or_create_connection()
    try:
        yield connection
    except Exception as e:
        # If connection is broken, reset the connection for this event loop
        if "connection" in str(e).lower() or "closed" in str(e).lower():
            try:
                loop = asyncio.get_running_loop()
                loop_id = id(loop)
                logger.warning(f"Database connection error detected, resetting connection for event loop {loop_id}")
                with _pool_lock:
                    if loop_id in _connection_pools:
                        del _connection_pools[loop_id]
            except RuntimeError:
                # No event loop running, can't identify which connection to reset
                logger.warning("Database connection error detected, but no event loop to identify connection")
        raise

