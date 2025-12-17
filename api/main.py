from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.auth import PasswordAuthMiddleware
from api.middleware.idempotency import IdempotencyMiddleware
from api.routers import (
    auth,
    chat,
    config,
    context,
    embedding,
    embedding_rebuild,
    insights,
    mcp,
    models,
    notebooks,
    search,
    settings,
    source_chat,
    sources,
    transformations,
)
from api.routers import commands as commands_router

# Import commands to register them in the API process
try:

    logger.info("Commands imported in API process")
except Exception as e:
    logger.error(f"Failed to import commands in API process: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for the FastAPI application.
    Initializes database schema automatically on startup.
    """
    # Startup: Initialize database schema
    logger.info("Starting API initialization...")

    try:
        from open_notebook.database.schema_init import init_schema, needs_init
        
        if await needs_init():
            logger.info("Initializing database schema...")
            success = await init_schema()
            if success:
                logger.success("Database schema initialized successfully")
            else:
                logger.error("CRITICAL: Database schema initialization failed")
                raise RuntimeError("Failed to initialize database schema")
        else:
            logger.info("Database schema already initialized")
    except Exception as e:
        logger.error(f"CRITICAL: Database schema initialization failed: {str(e)}")
        logger.exception(e)
        # Fail fast - don't start the API with an uninitialized database schema
        raise RuntimeError(f"Failed to initialize database schema: {str(e)}") from e

    # Initialize Qdrant collections
    try:
        from open_notebook.services.qdrant_service import qdrant_service
        logger.info("Initializing Qdrant collections...")
        await qdrant_service._ensure_client()
        await qdrant_service._ensure_collections()
        logger.success("Qdrant collections initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize Qdrant collections: {str(e)}")
        logger.warning("Qdrant collections will be created on first use")
    
    # Initialize and register tools for Agentic RAG
    try:
        from open_notebook.services.tool_service import (
            InternetSearchTool,
            TextSearchTool,
            VectorSearchTool,
            tool_registry,
        )
        
        logger.info("Registering built-in tools...")
        await tool_registry.register(VectorSearchTool())
        await tool_registry.register(TextSearchTool())
        await tool_registry.register(InternetSearchTool())
        logger.success("Built-in tools registered successfully")
        
        # Register PageIndex tool if available
        try:
            from open_notebook.services.pageindex_service import PageIndexSearchTool, pageindex_service
            
            await pageindex_service._ensure_initialized()
            if pageindex_service.is_available():
                await tool_registry.register(PageIndexSearchTool())
                logger.success("PageIndex tool registered successfully")
            else:
                logger.info("PageIndex not available, skipping registration")
        except Exception as e:
            logger.warning(f"Failed to register PageIndex tool: {e}")
        
        # MCP servers - auto-connect pageindex as default tool, others remain manual
        try:
            from open_notebook.services.mcp_service import mcp_manager
            from open_notebook.services.tool_service import MCPToolWrapper, tool_registry
            
            if mcp_manager and mcp_manager.available:
                mcp_servers = mcp_manager.get_servers()
                logger.info(f"Found {len(mcp_servers)} MCP servers in config")
                
                # Auto-connect and register pageindex server as default tool
                if "pageindex" in mcp_servers:
                    try:
                        logger.info("Auto-connecting pageindex MCP server (default tool)...")
                        connected = await mcp_manager.connect_server("pageindex")
                        if connected:
                            tools = await mcp_manager.get_tools("pageindex")
                            registered_count = 0
                            for tool_info in tools:
                                try:
                                    tool_name = tool_info.get("name", "unknown")
                                    tool_description = tool_info.get("description", "")
                                    mcp_wrapper = MCPToolWrapper(tool_info)
                                    await tool_registry.register(mcp_wrapper)
                                    registered_count += 1
                                    logger.success(f"Registered pageindex MCP tool: {mcp_wrapper.name} (original: {tool_name}, description: {tool_description[:100] if tool_description else 'None'}...)")
                                except Exception as e:
                                    logger.warning(f"Failed to register pageindex tool {tool_info.get('name')}: {e}")
                            logger.success(f"PageIndex MCP server connected and {registered_count} tool(s) registered successfully")
                        else:
                            logger.warning("Failed to connect to pageindex MCP server")
                    except Exception as e:
                        logger.warning(f"Failed to auto-connect pageindex MCP server: {e}")
                else:
                    logger.info("PageIndex MCP server not found in config, skipping auto-connect")
            else:
                logger.info("MCP not available, skipping MCP tool registration")
        except Exception as e:
            logger.warning(f"Failed to check MCP servers: {e}")
            
    except Exception as e:
        logger.warning(f"Failed to initialize tools: {e}")
        logger.warning("Tool registration will be skipped, but Agentic RAG may not work properly")

    # Start background cleanup tasks
    cleanup_tasks = []
    
    # Task 1: Idempotency records cleanup
    try:
        from api.idempotency_service import IdempotencyService
        
        async def idempotency_cleanup_loop():
            """Background task to clean up expired idempotency records"""
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    count = await IdempotencyService.cleanup_expired_records()
                    if count > 0:
                        logger.info(f"Idempotency cleanup: removed {count} expired records")
                except asyncio.CancelledError:
                    logger.info("Idempotency cleanup task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in idempotency cleanup task: {e}")
        
        idempotency_task = asyncio.create_task(idempotency_cleanup_loop())
        cleanup_tasks.append(idempotency_task)
        logger.info("Started background idempotency cleanup task")
    except Exception as e:
        logger.warning(f"Failed to start idempotency cleanup task: {e}")
    
    # Task 2: Orphaned insights cleanup (Qdrant)
    try:
        from open_notebook.services.qdrant_service import qdrant_service
        import os
        
        # 從環境變數讀取清理間隔（默認 6 小時）
        cleanup_interval = int(os.getenv("ORPHANED_INSIGHTS_CLEANUP_INTERVAL", "21600"))  # 6 hours in seconds
        
        async def orphaned_insights_cleanup_loop():
            """Background task to clean up orphaned insights in Qdrant"""
            # 首次啟動後等待 1 小時再執行第一次清理（避免與啟動時的其他操作衝突）
            await asyncio.sleep(3600)
            
            while True:
                try:
                    stats = await qdrant_service.cleanup_orphaned_insights(
                        batch_size=100,
                        max_cleanup=None  # 不限制清理數量
                    )
                    if stats["orphaned"] > 0:
                        logger.info(
                            f"Orphaned insights cleanup: checked {stats['checked']}, "
                            f"found {stats['orphaned']} orphaned, cleaned {stats['cleaned']}, "
                            f"failed {stats['failed']}"
                        )
                    
                    await asyncio.sleep(cleanup_interval)
                except asyncio.CancelledError:
                    logger.info("Orphaned insights cleanup task cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in orphaned insights cleanup task: {e}")
                    # 即使出錯也繼續運行，等待下次清理
                    await asyncio.sleep(cleanup_interval)
        
        orphaned_insights_task = asyncio.create_task(orphaned_insights_cleanup_loop())
        cleanup_tasks.append(orphaned_insights_task)
        logger.info(f"Started background orphaned insights cleanup task (interval: {cleanup_interval}s)")
    except Exception as e:
        logger.warning(f"Failed to start orphaned insights cleanup task: {e}")

    # Initialize AsyncSqliteSaver checkpointer for chat graph
    # Use standard LangGraph approach: create aiosqlite connection in lifespan
    # IMPORTANT: db_connection must be in outer scope for shutdown cleanup
    db_connection = None
    try:
        import aiosqlite
        import os
        from open_notebook.graphs.chat import initialize_checkpointer
        from open_notebook.config import LANGGRAPH_CHECKPOINT_FILE
        
        # Ensure directory exists
        checkpoint_path = os.path.abspath(LANGGRAPH_CHECKPOINT_FILE)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if checkpoint_dir:  # Only create if dirname is not empty
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create aiosqlite connection
        db_connection = await aiosqlite.connect(checkpoint_path)
        
        # Initialize checkpointer with the connection (this will call setup() internally)
        await initialize_checkpointer(db_connection)
        logger.success("Chat graph checkpointer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chat graph checkpointer: {e}")
        logger.exception(e)
        logger.warning("Chat functionality may not work properly")
        # Close connection if initialization failed
        if db_connection:
            try:
                await db_connection.close()
            except Exception as close_error:
                logger.warning(f"Error closing checkpointer connection: {close_error}")
            db_connection = None

    logger.success("API initialization completed successfully")

    # Yield control to the application
    yield

    # Shutdown: cleanup background tasks
    for task in cleanup_tasks:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    
    # Cleanup chat graph checkpointer connection
    if db_connection:
        try:
            await db_connection.close()
            logger.info("Chat graph checkpointer connection closed")
        except Exception as e:
            logger.warning(f"Error closing checkpointer connection during shutdown: {e}")
    
    # Close database connection pool
    try:
        from open_notebook.database.connection_pool import close_pool
        await close_pool()
    except Exception as e:
        logger.warning(f"Error closing database connection pool: {e}")
    
    logger.info("API shutdown complete")


app = FastAPI(
    title="Research Assistant API",
    description="API for Research Assistant",
    version="0.2.2",
    lifespan=lifespan,
)

# Add idempotency middleware (before authentication)
app.add_middleware(IdempotencyMiddleware)

# Add password authentication middleware
# Exclude /api/auth/status and /api/config from authentication
app.add_middleware(PasswordAuthMiddleware, excluded_paths=["/", "/health", "/docs", "/openapi.json", "/redoc", "/api/auth/status", "/api/config"])

# Add CORS middleware last (so it processes first)
# Configure CORS from environment variable
import os

# Get environment settings
env_type = os.getenv("ENVIRONMENT", "development").lower()
cors_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()

allow_origins = []

# Logic:
# 1. If explicitly set to "*" or in development and not set, allow all
if cors_origins_env == "*" or (not cors_origins_env and env_type == "development"):
    allow_origins = ["*"]
    if env_type == "production":
        logger.warning("SECURITY WARNING: CORS is accepting all origins ('*') in production!")
# 2. If specific origins are set, parse them
elif cors_origins_env:
    allow_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
    logger.info(f"CORS allowed origins set to: {allow_origins}")
# 3. Production environment and not set, keep empty list (most secure)
else:
    logger.warning("CORS_ALLOWED_ORIGINS is not set in production. API may be inaccessible from browsers.")
    # Keep empty list to force explicit configuration
    allow_origins = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(config.router, prefix="/api", tags=["config"])
app.include_router(notebooks.router, prefix="/api", tags=["notebooks"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(transformations.router, prefix="/api", tags=["transformations"])
app.include_router(embedding.router, prefix="/api", tags=["embedding"])
app.include_router(embedding_rebuild.router, prefix="/api/embeddings", tags=["embeddings"])
app.include_router(settings.router, prefix="/api", tags=["settings"])
app.include_router(context.router, prefix="/api", tags=["context"])
app.include_router(sources.router, prefix="/api", tags=["sources"])
app.include_router(insights.router, prefix="/api", tags=["insights"])
app.include_router(commands_router.router, prefix="/api", tags=["commands"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(source_chat.router, prefix="/api", tags=["source-chat"])
app.include_router(mcp.router, prefix="/api", tags=["MCP"])


@app.get("/")
async def root():
    return {"message": "API is running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
