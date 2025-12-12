"""
Migration script: Add processing_status and error_message fields to source table.

This script updates existing source records to have a default processing_status
of 'completed' if they don't have one set.

Usage:
    python -m open_notebook.database.migrations.migrate_add_processing_status
"""
import asyncio
from loguru import logger

from open_notebook.database.repository import repo_query


async def migrate():
    """Run the migration to set default processing_status for existing sources."""
    try:
        logger.info("Starting migration: Add processing_status to existing sources")
        
        # Update existing sources without processing_status to 'completed'
        result = await repo_query(
            "UPDATE source SET processing_status = 'completed' WHERE processing_status IS NONE RETURN COUNT"
        )
        
        # Extract count from result
        count = 0
        if result and len(result) > 0:
            if isinstance(result[0], dict) and "count" in result[0]:
                count = result[0]["count"]
            elif isinstance(result[0], list) and len(result[0]) > 0:
                if isinstance(result[0][0], dict) and "count" in result[0][0]:
                    count = result[0][0]["count"]
        
        logger.info(f"Migration complete: Updated {count} source records")
        return count
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        logger.exception(e)
        raise


if __name__ == "__main__":
    asyncio.run(migrate())

