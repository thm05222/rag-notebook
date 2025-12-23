#!/usr/bin/env python3
"""
One-time script to initialize message table for chat history persistence.

This script creates the message table in SurrealDB to store chat history
with thinking process data. Run this script once after deploying the new code.

Usage:
    cd /home/qiyoo/rag-notebook
    python scripts/init_message_table.py
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


async def init_message_table():
    """Initialize the message table in SurrealDB."""
    from open_notebook.database.repository import repo_query
    from loguru import logger
    
    sql = """
    -- Message table for chat history persistence
    DEFINE TABLE message SCHEMAFULL;
    DEFINE FIELD session_id ON TABLE message TYPE record<chat_session>;
    DEFINE FIELD role ON TABLE message TYPE string ASSERT $value INSIDE ["user", "ai"];
    DEFINE FIELD content ON TABLE message TYPE string;
    DEFINE FIELD thinking_process ON TABLE message FLEXIBLE TYPE option<object>;
    DEFINE FIELD created_at ON TABLE message TYPE datetime DEFAULT time::now();
    
    -- Index for efficient history retrieval
    DEFINE INDEX idx_message_session ON TABLE message COLUMNS session_id;
    """
    
    try:
        logger.info("Initializing message table...")
        await repo_query(sql)
        logger.success("Message table initialized successfully!")
        print("\n✅ Message table initialized successfully!")
        print("\nYou can now restart the backend container to apply the new code.")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize message table: {e}")
        print(f"\n❌ Failed to initialize message table: {e}")
        return False


async def verify_table():
    """Verify the message table was created correctly."""
    from open_notebook.database.repository import repo_query
    from loguru import logger
    
    try:
        # Try to query the message table
        result = await repo_query("INFO FOR TABLE message;")
        logger.info(f"Message table info: {result}")
        print("\n📋 Message table structure verified:")
        print(f"   {result}")
        return True
    except Exception as e:
        logger.warning(f"Could not verify message table: {e}")
        print(f"\n⚠️ Could not verify message table: {e}")
        return False


async def main():
    """Main entry point."""
    print("=" * 60)
    print("Chat History Persistence - Message Table Initialization")
    print("=" * 60)
    
    # Initialize table
    success = await init_message_table()
    
    if success:
        # Verify table
        await verify_table()
    
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

