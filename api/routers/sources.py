import os
from pathlib import Path
from typing import Any, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse, Response
from loguru import logger
from surreal_commands import execute_command_sync

from api.command_service import CommandService
from api.models import (
    AssetModel,
    CreateSourceInsightRequest,
    SourceCreate,
    SourceInsightResponse,
    SourceListResponse,
    SourceResponse,
    SourceStatusResponse,
    SourceUpdate,
)
from commands.source_commands import SourceProcessingInput
from open_notebook.config import UPLOADS_FOLDER, MAX_UPLOAD_SIZE
from open_notebook.database.repository import ensure_record_id, repo_query
from open_notebook.domain.notebook import Notebook, Source
from open_notebook.domain.transformation import Transformation
from open_notebook.exceptions import InvalidInputError

router = APIRouter()


def generate_unique_filename(original_filename: str, upload_folder: str) -> str:
    """Generate unique filename like Streamlit app (append counter if file exists)."""
    file_path = Path(upload_folder)
    file_path.mkdir(parents=True, exist_ok=True)

    # Split filename and extension
    stem = Path(original_filename).stem
    suffix = Path(original_filename).suffix

    # Check if file exists and generate unique name
    counter = 0
    while True:
        if counter == 0:
            new_filename = original_filename
        else:
            new_filename = f"{stem} ({counter}){suffix}"

        full_path = file_path / new_filename
        if not full_path.exists():
            return str(full_path)
        counter += 1


async def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file to uploads folder and return file path."""
    if not upload_file.filename:
        raise ValueError("No filename provided")

    # Check file size before reading
    # Note: FastAPI's UploadFile.size may not be available for streaming uploads
    # We'll check during read with chunked processing
    # Generate unique filename
    file_path = generate_unique_filename(upload_file.filename, UPLOADS_FOLDER)

    try:
        # Save file with size checking (streaming to avoid memory issues)
        total_size = 0
        with open(file_path, "wb") as f:
            while True:
                # Read in chunks of 1MB to avoid memory issues
                chunk = await upload_file.read(1024 * 1024)
                if not chunk:
                    break
                
                total_size += len(chunk)
                
                # Check size limit during upload
                if total_size > MAX_UPLOAD_SIZE:
                    # Clean up partial file
                    f.close()
                    if os.path.exists(file_path):
                        os.unlink(file_path)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File size exceeds maximum allowed size of {MAX_UPLOAD_SIZE / (1024 * 1024):.1f}MB"
                    )
                
                f.write(chunk)

        logger.info(f"Saved uploaded file to: {file_path} (size: {total_size / (1024 * 1024):.2f}MB)")
        return file_path
    except HTTPException:
        # Re-raise HTTP exceptions (size limit)
        raise
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        # Clean up partial file if it exists
        if os.path.exists(file_path):
            os.unlink(file_path)
        raise


def parse_source_form_data(
    type: str = Form(...),
    notebook_id: Optional[str] = Form(None),
    notebooks: Optional[str] = Form(None),  # JSON string of notebook IDs
    url: Optional[str] = Form(None),
    content: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    transformations: Optional[str] = Form(None),  # JSON string of transformation IDs
    embed: str = Form("false"),  # Accept as string, convert to bool
    build_pageindex: str = Form("false"),  # Accept as string, convert to bool
    delete_source: str = Form("false"),  # Accept as string, convert to bool
    async_processing: str = Form("false"),  # Accept as string, convert to bool
    file: Optional[UploadFile] = File(None),
) -> tuple[SourceCreate, Optional[UploadFile]]:
    """Parse form data into SourceCreate model and return upload file separately."""
    import json

    # Convert string booleans to actual booleans
    def str_to_bool(value: str) -> bool:
        return value.lower() in ("true", "1", "yes", "on")

    embed_bool = str_to_bool(embed)
    build_pageindex_bool = str_to_bool(build_pageindex)
    delete_source_bool = str_to_bool(delete_source)
    async_processing_bool = str_to_bool(async_processing)

    # Parse JSON strings
    notebooks_list = None
    if notebooks:
        try:
            notebooks_list = json.loads(notebooks)
        except json.JSONDecodeError:
            logger.error(f"DEBUG - Invalid JSON in notebooks field: {notebooks}")
            raise ValueError("Invalid JSON in notebooks field")

    transformations_list = []
    if transformations:
        try:
            transformations_list = json.loads(transformations)
        except json.JSONDecodeError:
            logger.error(
                f"DEBUG - Invalid JSON in transformations field: {transformations}"
            )
            raise ValueError("Invalid JSON in transformations field")

    # Create SourceCreate instance
    try:
        source_data = SourceCreate(
            type=type,
            notebook_id=notebook_id,
            notebooks=notebooks_list,
            url=url,
            content=content,
            title=title,
            file_path=None,  # Will be set later if file is uploaded
            transformations=transformations_list,
            embed=embed_bool,
            build_pageindex=build_pageindex_bool,
            delete_source=delete_source_bool,
            async_processing=async_processing_bool,
        )
        pass  # SourceCreate instance created successfully
    except Exception as e:
        logger.error(f"Failed to create SourceCreate instance: {e}")
        raise

    return source_data, file


@router.get("/sources", response_model=List[SourceListResponse])
async def get_sources(
    notebook_id: Optional[str] = Query(None, description="Filter by notebook ID"),
    limit: int = Query(50, ge=1, le=100, description="Number of sources to return (1-100)"),
    offset: int = Query(0, ge=0, description="Number of sources to skip"),
    sort_by: str = Query("updated", description="Field to sort by (created or updated)"),
    sort_order: str = Query("desc", description="Sort order (asc or desc)"),
):
    """Get sources with pagination and sorting support."""
    try:
        # Validate sort parameters
        if sort_by not in ["created", "updated"]:
            raise HTTPException(status_code=400, detail="sort_by must be 'created' or 'updated'")
        if sort_order.lower() not in ["asc", "desc"]:
            raise HTTPException(status_code=400, detail="sort_order must be 'asc' or 'desc'")

        # Build ORDER BY clause
        order_clause = f"ORDER BY {sort_by} {sort_order.upper()}"

        # Build the query
        if notebook_id:
            # Verify notebook exists first
            notebook = await Notebook.get(notebook_id)
            if not notebook:
                raise HTTPException(status_code=404, detail="Notebook not found")

            # Query sources for specific notebook - include command field
            # Note: embedded status will be calculated in Python from Qdrant
            query = f"""
                SELECT id, asset, created, title, updated, topics, command, pageindex_structure, pageindex_built_at,
                (SELECT VALUE count() FROM source_insight WHERE source = $parent.id GROUP ALL)[0].count OR 0 AS insights_count
                FROM (select value in from reference where out=$notebook_id)
                {order_clause}
                LIMIT $limit START $offset
            """
            result = await repo_query(
                query, {
                    "notebook_id": ensure_record_id(notebook_id),
                    "limit": limit,
                    "offset": offset
                }
            )
        else:
            # Query all sources - include command field
            # Note: embedded status will be calculated in Python from Qdrant
            query = f"""
                SELECT id, asset, created, title, updated, topics, command, pageindex_structure, pageindex_built_at,
                (SELECT VALUE count() FROM source_insight WHERE source = $parent.id GROUP ALL)[0].count OR 0 AS insights_count
                FROM source
                {order_clause}
                LIMIT $limit START $offset
            """
            result = await repo_query(query, {"limit": limit, "offset": offset})

        # Extract command IDs and source IDs for batch fetching
        command_ids = []
        command_to_source = {}
        source_ids = []

        for row in result:
            source_ids.append(row["id"])
            command = row.get("command")
            if command:
                command_str = str(command)
                command_ids.append(command_str)
                command_to_source[command_str] = row["id"]

        # Batch fetch embedded status from Qdrant
        embedded_status = {}
        try:
            from open_notebook.services.qdrant_service import qdrant_service
            from qdrant_client.models import FieldCondition, Filter, MatchAny
            
            if qdrant_service.client is None:
                await qdrant_service._ensure_client()
            
            client = qdrant_service.client
            if client and source_ids:
                # Scroll through all points and check which sources have embeddings
                scroll_result = client.scroll(
                    collection_name="source_embeddings",
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="source_id",
                                match=MatchAny(any=source_ids)
                            )
                        ]
                    ),
                    limit=10000,
                    with_payload=True,
                    with_vectors=False
                )
                
                # Mark sources that have at least one embedding
                if scroll_result and scroll_result[0]:
                    for point in scroll_result[0]:
                        source_id = point.payload.get("source_id")
                        if source_id:
                            embedded_status[source_id] = True
        except Exception as e:
            logger.warning(f"Failed to fetch embedded status from Qdrant: {e}")

        # Batch fetch command statuses
        command_statuses = {}
        if command_ids:
            try:
                # Get status for all commands in batch (if the library supports it)
                # If not, we'll fall back to individual calls, but limit concurrent requests
                import asyncio

                from surreal_commands import get_command_status

                async def get_status_safe(command_id: str):
                    try:
                        status = await get_command_status(command_id)
                        return (command_id, status)
                    except Exception as e:
                        logger.warning(
                            f"Failed to get status for command {command_id}: {e}"
                        )
                        return (command_id, None)

                # Limit concurrent requests to avoid overwhelming the command service
                semaphore = asyncio.Semaphore(10)

                async def get_status_with_limit(command_id: str):
                    async with semaphore:
                        return await get_status_safe(command_id)

                # Fetch statuses concurrently but with limit
                status_tasks = [get_status_with_limit(cmd_id) for cmd_id in command_ids]
                status_results = await asyncio.gather(
                    *status_tasks, return_exceptions=True
                )

                # Process results
                for result_item in status_results:
                    if isinstance(result_item, Exception):
                        continue
                    if isinstance(result_item, tuple) and len(result_item) == 2:
                        cmd_id, status = result_item
                        command_statuses[cmd_id] = status

            except Exception as e:
                logger.warning(f"Failed to batch fetch command statuses: {e}")

        # Convert result to response model
        response_list = []
        for row in result:
            command = row.get("command")
            command_id = str(command) if command else None
            status = None
            processing_info = None

            # Get status information if command exists
            if command_id and command_id in command_statuses:
                status_obj = command_statuses[command_id]
                if status_obj:
                    status = status_obj.status
                    # Extract execution metadata from nested result structure
                    result_data: dict[str, Any] | None = getattr(status_obj, "result", None)
                    execution_metadata: dict[str, Any] = result_data.get("execution_metadata", {}) if isinstance(result_data, dict) else {}
                    processing_info = {
                        "started_at": execution_metadata.get("started_at"),
                        "completed_at": execution_metadata.get("completed_at"),
                        "error": getattr(status_obj, "error_message", None),
                    }
            elif command_id:
                # Command exists but status couldn't be fetched
                status = "unknown"

            # Extract PageIndex fields
            pageindex_structure = row.get("pageindex_structure")
            pageindex_built_at = row.get("pageindex_built_at")
            
            response_list.append(
                SourceListResponse(
                    id=row["id"],
                    title=row.get("title"),
                    topics=row.get("topics") or [],
                    asset=AssetModel(
                        file_path=row["asset"].get("file_path")
                        if row.get("asset")
                        else None,
                        url=row["asset"].get("url") if row.get("asset") else None,
                    )
                    if row.get("asset")
                    else None,
                    embedded=embedded_status.get(row["id"], False),
                    embedded_chunks=0,  # Removed from query - not needed in list view
                    insights_count=row.get("insights_count", 0),
                    created=str(row["created"]),
                    updated=str(row["updated"]),
                    processing_status=row.get("processing_status"),
                    error_message=row.get("error_message"),
                    # Status fields
                    command_id=command_id,
                    status=status,
                    processing_info=processing_info,
                    # PageIndex fields
                    has_pageindex=pageindex_structure is not None,
                    pageindex_built_at=str(pageindex_built_at) if pageindex_built_at else None,
                )
            )

        return response_list
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching sources: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching sources: {str(e)}")


@router.post("/sources", response_model=SourceResponse)
async def create_source(
    form_data: tuple[SourceCreate, Optional[UploadFile]] = Depends(
        parse_source_form_data
    ),
):
    """Create a new source with support for both JSON and multipart form data."""
    source_data, upload_file = form_data
    file_path = None  # Initialize early to avoid UnboundLocalError in exception handlers

    try:
        # Verify all specified notebooks exist (backward compatibility support)
        for notebook_id in (source_data.notebooks or []):
            notebook = await Notebook.get(notebook_id)
            if not notebook:
                raise HTTPException(
                    status_code=404, detail=f"Notebook {notebook_id} not found"
                )

        # Handle file upload if provided
        # Only process file upload if type is explicitly "upload"
        if upload_file and source_data.type == "upload":
            try:
                file_path = await save_uploaded_file(upload_file)
                logger.info(f"File saved successfully: {file_path}")
            except Exception as e:
                logger.error(f"File upload failed: {e}")
                raise HTTPException(
                    status_code=400, detail=f"File upload failed: {str(e)}"
                )

        # Prepare content_state for processing
        content_state: dict[str, Any] = {}

        if source_data.type == "link":
            if not source_data.url:
                raise HTTPException(
                    status_code=400, detail="URL is required for link type"
                )
            content_state["url"] = source_data.url
        elif source_data.type == "upload":
            # Use uploaded file path or provided file_path (backward compatibility)
            final_file_path = file_path or source_data.file_path
            if not final_file_path:
                logger.error(f"Upload type but no file_path: file_path={file_path}, source_data.file_path={source_data.file_path}, upload_file={upload_file}")
                raise HTTPException(
                    status_code=400,
                    detail="File upload or file_path is required for upload type",
                )
            logger.info(f"Setting content_state file_path to: {final_file_path}")
            content_state["file_path"] = final_file_path
            content_state["delete_source"] = source_data.delete_source
        elif source_data.type == "text":
            if not source_data.content:
                raise HTTPException(
                    status_code=400, detail="Content is required for text type"
                )
            content_state["content"] = source_data.content
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid source type. Must be link, upload, or text",
            )

        # Validate transformations exist
        transformation_ids = source_data.transformations or []
        for trans_id in transformation_ids:
            transformation = await Transformation.get(trans_id)
            if not transformation:
                raise HTTPException(
                    status_code=404, detail=f"Transformation {trans_id} not found"
                )

        # Branch based on processing mode
        if source_data.async_processing:
            # ASYNC PATH: Create source record first, then queue command
            logger.info("Using async processing path")

            # Create minimal source record - let SurrealDB generate the ID
            source = Source(
                title=source_data.title or "Processing...",
                topics=[],
            )
            await source.save()

            # Add source to notebooks immediately so it appears in the UI
            # The source_graph will skip adding duplicates
            for notebook_id in (source_data.notebooks or []):
                await source.add_to_notebook(notebook_id)

            try:
                # Import command modules to ensure they're registered
                import commands.source_commands  # noqa: F401

                # Submit command for background processing
                command_input = SourceProcessingInput(
                    source_id=str(source.id),
                    content_state=content_state,
                    notebook_ids=source_data.notebooks,
                    transformations=transformation_ids,
                    embed=source_data.embed,
                    build_pageindex=source_data.build_pageindex,
                )

                command_id = await CommandService.submit_command_job(
                    "open_notebook",  # app name
                    "process_source",  # command name
                    command_input.model_dump(),
                )

                logger.info(f"Submitted async processing command: {command_id}")

                # Update source with command reference immediately
                # command_id already includes 'command:' prefix
                source.command = ensure_record_id(command_id)
                await source.save()

                # Return source with command info
                return SourceResponse(
                    id=source.id or "",
                    title=source.title,
                    topics=source.topics or [],
                    asset=None,  # Will be populated after processing
                    full_text=None,  # Will be populated after processing
                    embedded=False,  # Will be updated after processing
                    embedded_chunks=0,
                    created=str(source.created),
                    updated=str(source.updated),
                    command_id=command_id,
                    status="new",
                    processing_info={"async": True, "queued": True},
                    processing_status=source.processing_status or "pending",
                    error_message=source.error_message,
                )

            except Exception as e:
                logger.error(f"Failed to submit async processing command: {e}")
                # Clean up source record on command submission failure
                try:
                    await source.delete()
                except Exception:
                    pass
                # Clean up uploaded file if we created it
                if file_path and upload_file:
                    try:
                        os.unlink(file_path)
                    except Exception:
                        pass
                raise HTTPException(
                    status_code=500, detail=f"Failed to queue processing: {str(e)}"
                )

        else:
            # SYNC PATH: Execute synchronously using execute_command_sync
            logger.info("Using sync processing path")

            try:
                # Import command modules to ensure they're registered
                import commands.source_commands  # noqa: F401

                # Create source record - let SurrealDB generate the ID
                source = Source(
                    title=source_data.title or "Processing...",
                    topics=[],
                )
                await source.save()

                # Add source to notebooks immediately so it appears in the UI
                # The source_graph will skip adding duplicates
                for notebook_id in (source_data.notebooks or []):
                    await source.add_to_notebook(notebook_id)

                # Execute command synchronously
                command_input = SourceProcessingInput(
                    source_id=str(source.id),
                    content_state=content_state,
                    notebook_ids=source_data.notebooks,
                    transformations=transformation_ids,
                    embed=source_data.embed,
                    build_pageindex=source_data.build_pageindex,
                )

                result = execute_command_sync(
                    "open_notebook",  # app name
                    "process_source",  # command name
                    command_input.model_dump(),
                    timeout=300,  # 5 minute timeout for sync processing
                )

                if not result.is_success():
                    logger.error(f"Sync processing failed: {result.error_message}")
                    # Clean up source record
                    try:
                        await source.delete()
                    except Exception:
                        pass
                    # Clean up uploaded file if we created it
                    if file_path and upload_file:
                        try:
                            os.unlink(file_path)
                        except Exception:
                            pass
                    raise HTTPException(
                        status_code=500,
                        detail=f"Processing failed: {result.error_message}",
                    )

                # Get the processed source
                if not source.id:
                    raise HTTPException(
                        status_code=500, detail="Source ID is missing"
                    )
                processed_source = await Source.get(source.id)
                if not processed_source:
                    raise HTTPException(
                        status_code=500, detail="Processed source not found"
                    )

                embedded_chunks = await processed_source.get_embedded_chunks()
                return SourceResponse(
                    id=processed_source.id or "",
                    title=processed_source.title,
                    topics=processed_source.topics or [],
                    asset=AssetModel(
                        file_path=processed_source.asset.file_path
                        if processed_source.asset
                        else None,
                        url=processed_source.asset.url
                        if processed_source.asset
                        else None,
                    )
                    if processed_source.asset
                    else None,
                    full_text=processed_source.full_text,
                    embedded=embedded_chunks > 0,
                    processing_status=processed_source.processing_status,
                    error_message=processed_source.error_message,
                    embedded_chunks=embedded_chunks,
                    created=str(processed_source.created),
                    updated=str(processed_source.updated),
                    # No command_id or status for sync processing (legacy behavior)
                )

            except Exception as e:
                logger.error(f"Sync processing failed: {e}")
                # Clean up uploaded file if we created it
                if file_path and upload_file:
                    try:
                        os.unlink(file_path)
                    except Exception:
                        pass
                raise

    except HTTPException:
        # Clean up uploaded file on HTTP exceptions if we created it
        if file_path and upload_file:
            try:
                os.unlink(file_path)
            except Exception:
                pass
        raise
    except InvalidInputError as e:
        # Clean up uploaded file on validation errors if we created it
        if file_path and upload_file:
            try:
                os.unlink(file_path)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating source: {str(e)}")
        # Clean up uploaded file on unexpected errors if we created it
        if file_path and upload_file:
            try:
                os.unlink(file_path)
            except Exception:
                pass
        raise HTTPException(status_code=500, detail=f"Error creating source: {str(e)}")


@router.post("/sources/json", response_model=SourceResponse)
async def create_source_json(source_data: SourceCreate):
    """Create a new source using JSON payload (legacy endpoint for backward compatibility)."""
    # Convert to form data format and call main endpoint
    form_data = (source_data, None)
    return await create_source(form_data)


@router.post("/sources/{source_id}/build-pageindex")
async def build_pageindex_for_source(source_id: str):
    """
    為指定的 source 建立 PageIndex 索引。
    
    如果索引已存在，會直接返回成功（不會重新建立）。
    """
    try:
        from open_notebook.services.pageindex_service import pageindex_service
        from open_notebook.database.repository import ensure_record_id
        
        # Ensure PageIndex service is initialized before checking availability
        await pageindex_service._ensure_initialized()
        
        if not pageindex_service.is_available():
            raise HTTPException(
                status_code=503, 
                detail="PageIndex is not available. Please ensure PageIndex is installed."
            )
        
        # Ensure source ID has table prefix
        full_source_id = (
            source_id if source_id.startswith("source:")
            else f"source:{source_id}"
        )
        source = await Source.get(full_source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        
        if not source.full_text:
            raise HTTPException(
                status_code=400, 
                detail="Source has no content to index. Please ensure the source has been processed."
            )
        
        # 建立索引（如果已存在會直接返回緩存的索引）
        try:
            await pageindex_service._get_or_create_index_for_source(source.id, source)
            return {
                "status": "success",
                "message": f"PageIndex built for source {source_id}",
                "source_id": source_id,
                "source_title": source.title
            }
        except Exception as e:
            # 檢查是否是因為文件結構不適合 PageIndex
            error_msg = str(e)
            is_unsuitable = (
                "hierarchical structure" in error_msg.lower() or 
                "empty structure" in error_msg.lower() or
                "not have a hierarchical structure" in error_msg.lower()
            )
            
            if is_unsuitable:
                # 這是預期的失敗（文件不適合 PageIndex），返回警告而不是錯誤
                logger.info(f"PageIndex not suitable for source {source_id}: {error_msg}")
                return {
                    "status": "not_suitable",
                    "message": "PageIndex is not suitable for this document type. Use vector search instead.",
                    "source_id": source_id,
                    "source_title": source.title,
                    "reason": error_msg
                }
            else:
                # 其他錯誤，正常拋出
                logger.error(f"Failed to build PageIndex for source {source_id}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to build PageIndex: {str(e)}"
                )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building PageIndex for source {source_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _resolve_source_file(source_id: str) -> tuple[str, str]:
    source = await Source.get(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    file_path = source.asset.file_path if source.asset else None
    if not file_path:
        raise HTTPException(status_code=404, detail="Source has no file to download")

    safe_root = os.path.realpath(UPLOADS_FOLDER)
    resolved_path = os.path.realpath(file_path)

    if not resolved_path.startswith(safe_root):
        logger.warning(
            f"Blocked download outside uploads directory for source {source_id}: {resolved_path}"
        )
        raise HTTPException(status_code=403, detail="Access to file denied")

    if not os.path.exists(resolved_path):
        raise HTTPException(status_code=404, detail="File not found on server")

    filename = os.path.basename(resolved_path)
    return resolved_path, filename


def _is_source_file_available(source: Source) -> Optional[bool]:
    if not source or not source.asset or not source.asset.file_path:
        return None

    file_path = source.asset.file_path
    safe_root = os.path.realpath(UPLOADS_FOLDER)
    resolved_path = os.path.realpath(file_path)

    if not resolved_path.startswith(safe_root):
        return False

    return os.path.exists(resolved_path)


@router.get("/sources/{source_id}", response_model=SourceResponse)
async def get_source(source_id: str):
    """Get a specific source by ID."""
    try:
        # Ensure source ID has table prefix
        full_source_id = (
            source_id if source_id.startswith("source:")
            else f"source:{source_id}"
        )
        source = await Source.get(full_source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Get status information if command exists
        status = None
        processing_info = None
        if source.command:
            try:
                status = await source.get_status()
                processing_info = await source.get_processing_progress()
            except Exception as e:
                logger.warning(f"Failed to get status for source {source_id}: {e}")
                status = "unknown"

        embedded_chunks = await source.get_embedded_chunks()

        # Get associated notebooks
        notebooks_query = await repo_query(
            "SELECT VALUE out FROM reference WHERE in = $source_id",
            {"source_id": ensure_record_id(source.id or source_id)}
        )
        notebook_ids = [str(nb_id) for nb_id in notebooks_query] if notebooks_query else []

        return SourceResponse(
            id=source.id or "",
            title=source.title,
            topics=source.topics or [],
            asset=AssetModel(
                file_path=source.asset.file_path if source.asset else None,
                url=source.asset.url if source.asset else None,
            )
            if source.asset
            else None,
            full_text=source.full_text,
            embedded=embedded_chunks > 0,
            embedded_chunks=embedded_chunks,
            file_available=_is_source_file_available(source),
            created=str(source.created),
            updated=str(source.updated),
            # Status fields
            command_id=str(source.command) if source.command else None,
            status=status,
            processing_info=processing_info,
            processing_status=source.processing_status,
            error_message=source.error_message,
            # Notebook associations
            notebooks=notebook_ids,
            # PageIndex fields
            has_pageindex=source.pageindex_structure is not None,
            pageindex_built_at=str(source.pageindex_built_at) if source.pageindex_built_at else None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching source {source_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching source: {str(e)}")


@router.head("/sources/{source_id}/download")
async def check_source_file(source_id: str):
    """Check if a source has a downloadable file."""
    try:
        await _resolve_source_file(source_id)
        return Response(status_code=200)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking file for source {source_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to verify file")


@router.get("/sources/{source_id}/download")
async def download_source_file(source_id: str):
    """Download the original file associated with an uploaded source."""
    try:
        resolved_path, filename = await _resolve_source_file(source_id)
        return FileResponse(
            path=resolved_path,
            filename=filename,
            media_type="application/octet-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file for source {source_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to download source file")


@router.get("/sources/{source_id}/status", response_model=SourceStatusResponse)
async def get_source_status(source_id: str):
    """Get processing status for a source."""
    try:
        # First, verify source exists
        source = await Source.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Check if this is a legacy source (no command)
        if not source.command:
            return SourceStatusResponse(
                status=None,
                message="Legacy source (completed before async processing)",
                processing_info=None,
                command_id=None,
            )

        # Get command status and processing info
        try:
            status = await source.get_status()
            processing_info = await source.get_processing_progress()

            # Generate descriptive message based on status
            if status == "completed":
                message = "Source processing completed successfully"
            elif status == "failed":
                message = "Source processing failed"
            elif status == "running":
                message = "Source processing in progress"
            elif status == "queued":
                message = "Source processing queued"
            elif status == "unknown":
                message = "Source processing status unknown"
            else:
                message = f"Source processing status: {status}"

            return SourceStatusResponse(
                status=status,
                message=message,
                processing_info=processing_info,
                command_id=str(source.command) if source.command else None,
            )

        except Exception as e:
            logger.warning(f"Failed to get status for source {source_id}: {e}")
            return SourceStatusResponse(
                status="unknown",
                message="Failed to retrieve processing status",
                processing_info=None,
                command_id=str(source.command) if source.command else None,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching status for source {source_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching source status: {str(e)}"
        )


@router.put("/sources/{source_id}", response_model=SourceResponse)
async def update_source(source_id: str, source_update: SourceUpdate):
    """Update a source."""
    try:
        source = await Source.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Update only provided fields
        if source_update.title is not None:
            source.title = source_update.title
        if source_update.topics is not None:
            source.topics = source_update.topics

        await source.save()

        embedded_chunks = await source.get_embedded_chunks()
        return SourceResponse(
            id=source.id or "",
            title=source.title,
            topics=source.topics or [],
            asset=AssetModel(
                file_path=source.asset.file_path if source.asset else None,
                url=source.asset.url if source.asset else None,
            )
            if source.asset
            else None,
            full_text=source.full_text,
            embedded=embedded_chunks > 0,
            embedded_chunks=embedded_chunks,
            created=str(source.created),
            updated=str(source.updated),
            processing_status=source.processing_status,
            error_message=source.error_message,
        )
    except HTTPException:
        raise
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating source {source_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating source: {str(e)}")


@router.post("/sources/{source_id}/retry", response_model=SourceResponse)
async def retry_source_processing(source_id: str):
    """Retry processing for a failed or stuck source."""
    try:
        # First, verify source exists
        source = await Source.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Check if source already has a running command
        if source.command:
            try:
                status = await source.get_status()
                if status in ["running", "queued"]:
                    raise HTTPException(
                        status_code=400,
                        detail="Source is already processing. Cannot retry while processing is active.",
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to check current status for source {source_id}: {e}"
                )
                # Continue with retry if we can't check status

        # Get notebooks that this source belongs to
        query = "SELECT notebook FROM reference WHERE source = $source_id"
        references = await repo_query(query, {"source_id": source_id})
        notebook_ids = [str(ref["notebook"]) for ref in references]

        if not notebook_ids:
            raise HTTPException(
                status_code=400, detail="Source is not associated with any notebooks"
            )

        # Prepare content_state based on source asset
        content_state = {}
        if source.asset:
            if source.asset.file_path:
                content_state = {
                    "file_path": source.asset.file_path,
                    "delete_source": False,  # Don't delete on retry
                }
            elif source.asset.url:
                content_state = {"url": source.asset.url}
            else:
                raise HTTPException(
                    status_code=400, detail="Source asset has no file_path or url"
                )
        else:
            # Check if it's a text source by trying to get full_text
            if source.full_text:
                content_state = {"content": source.full_text}
            else:
                raise HTTPException(
                    status_code=400, detail="Cannot determine source content for retry"
                )

        try:
            # Import command modules to ensure they're registered
            import commands.source_commands  # noqa: F401

            # Submit new command for background processing
            command_input = SourceProcessingInput(
                source_id=str(source.id),
                content_state=content_state,
                notebook_ids=notebook_ids,
                transformations=[],  # Use default transformations on retry
                embed=True,  # Always embed on retry
            )

            command_id = await CommandService.submit_command_job(
                "open_notebook",  # app name
                "process_source",  # command name
                command_input.model_dump(),
            )

            logger.info(
                f"Submitted retry processing command: {command_id} for source {source_id}"
            )

            # Update source with new command ID and clear error status
            source.command = ensure_record_id(f"command:{command_id}")
            # Clear error message and set status to pending
            await source.update_processing_status("pending")
            await source.save()

            # Get current embedded chunks count
            embedded_chunks = await source.get_embedded_chunks()

            # Return updated source response
            return SourceResponse(
                id=source.id or "",
                title=source.title,
                topics=source.topics or [],
                asset=AssetModel(
                    file_path=source.asset.file_path if source.asset else None,
                    url=source.asset.url if source.asset else None,
                )
                if source.asset
                else None,
                full_text=source.full_text,
                embedded=embedded_chunks > 0,
                embedded_chunks=embedded_chunks,
                created=str(source.created),
                updated=str(source.updated),
                command_id=command_id,
                status="queued",
                processing_info={"retry": True, "queued": True},
                processing_status=source.processing_status or "pending",
                error_message=source.error_message,
            )

        except Exception as e:
            logger.error(
                f"Failed to submit retry processing command for source {source_id}: {e}"
            )
            raise HTTPException(
                status_code=500, detail=f"Failed to queue retry processing: {str(e)}"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrying source processing for {source_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error retrying source processing: {str(e)}"
        )


@router.delete("/sources/{source_id}")
async def delete_source(source_id: str):
    """Delete a source and its associated uploaded file if exists."""
    try:
        source = await Source.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Delete associated file if it exists
        if source.asset and source.asset.file_path:
            file_path = source.asset.file_path
            safe_root = os.path.realpath(UPLOADS_FOLDER)
            
            # Resolve the file path - handle both relative and absolute paths
            # Use the same logic as _resolve_source_file for consistency
            resolved_path = None
            
            # Try multiple path resolution strategies
            if os.path.isabs(file_path):
                # Absolute path
                resolved_path = os.path.realpath(file_path)
            else:
                # Relative path - try multiple combinations
                # Strategy 1: Direct combination with UPLOADS_FOLDER
                combined_path = os.path.join(UPLOADS_FOLDER, file_path)
                if os.path.exists(combined_path):
                    resolved_path = os.path.realpath(combined_path)
                else:
                    # Strategy 2: Remove "uploads/" prefix if present
                    clean_path = file_path.replace("uploads/", "").replace("uploads\\", "")
                    if clean_path != file_path:
                        combined_path = os.path.join(UPLOADS_FOLDER, clean_path)
                        if os.path.exists(combined_path):
                            resolved_path = os.path.realpath(combined_path)
                    # Strategy 3: Try as absolute path (in case it's stored as absolute but not recognized)
                    if not resolved_path:
                        try:
                            temp_resolved = os.path.realpath(file_path)
                            if temp_resolved.startswith(safe_root):
                                resolved_path = temp_resolved
                        except Exception:
                            pass
            
            logger.info(f"Attempting to delete file for source {source_id}: original_path={file_path}, resolved_path={resolved_path}, safe_root={safe_root}")
            
            # Only delete if file is within the uploads directory (security check)
            if resolved_path and resolved_path.startswith(safe_root) and os.path.exists(resolved_path):
                try:
                    os.unlink(resolved_path)
                    logger.info(f"Successfully deleted file for source {source_id}: {resolved_path}")
                except Exception as e:
                    logger.error(f"Failed to delete file {resolved_path} for source {source_id}: {e}")
                    # Continue with source deletion even if file deletion fails
            elif resolved_path and not os.path.exists(resolved_path):
                logger.warning(f"File not found for source {source_id}: {resolved_path}")
            elif resolved_path and not resolved_path.startswith(safe_root):
                logger.warning(f"File path outside uploads directory for source {source_id}: {resolved_path} (safe_root: {safe_root})")
            else:
                logger.warning(f"Could not resolve file path for source {source_id}: original={file_path}, UPLOADS_FOLDER={UPLOADS_FOLDER}")

        # 清理 PageIndex 緩存（如果存在）
        try:
            from open_notebook.services.pageindex_service import pageindex_service
            pageindex_service.clear_cache_for_source(source_id)
        except Exception as e:
            logger.warning(f"Failed to clear PageIndex cache for source {source_id}: {e}")
            # 不影響刪除操作

        await source.delete()

        return {"message": "Source deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting source {source_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting source: {str(e)}")


@router.get("/sources/{source_id}/insights", response_model=List[SourceInsightResponse])
async def get_source_insights(source_id: str):
    """Get all insights for a specific source."""
    try:
        source = await Source.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        insights = await source.get_insights()
        return [
            SourceInsightResponse(
                id=insight.id or "",
                source_id=source_id,
                insight_type=insight.insight_type,
                content=insight.content,
                created=str(insight.created),
                updated=str(insight.updated),
            )
            for insight in insights
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching insights for source {source_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching insights: {str(e)}"
        )


@router.post("/sources/{source_id}/insights", response_model=SourceInsightResponse)
async def create_source_insight(source_id: str, request: CreateSourceInsightRequest):
    """Create a new insight for a source by running a transformation."""
    try:
        # Get source
        source = await Source.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Get transformation
        transformation = await Transformation.get(request.transformation_id)
        if not transformation:
            raise HTTPException(status_code=404, detail="Transformation not found")

        # Run transformation graph
        from open_notebook.graphs.transformation import graph as transform_graph

        await transform_graph.ainvoke(
            dict(source=source, transformation=transformation)  # type: ignore[arg-type]
        )

        # Get the newly created insight (last one)
        insights = await source.get_insights()
        if insights:
            newest = insights[-1]
            return SourceInsightResponse(
                id=newest.id or "",
                source_id=source_id,
                insight_type=newest.insight_type,
                content=newest.content,
                created=str(newest.created),
                updated=str(newest.updated),
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to create insight")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating insight for source {source_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating insight: {str(e)}")
