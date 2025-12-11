import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from api.models import NotebookCreate, NotebookResponse, NotebookUpdate
from open_notebook.database.repository import ensure_record_id, repo_query
from open_notebook.domain.notebook import Notebook, Source
from open_notebook.exceptions import InvalidInputError

router = APIRouter()


@router.get("/notebooks", response_model=List[NotebookResponse])
async def get_notebooks(
    archived: Optional[bool] = Query(None, description="Filter by archived status"),
    order_by: str = Query("updated desc", description="Order by field and direction"),
):
    """Get all notebooks with optional filtering and ordering."""
    try:
        # Build the query with counts
        query = f"""
            SELECT *,
            count(<-reference.in) as source_count
            FROM notebook
            ORDER BY {order_by}
        """

        result = await repo_query(query)

        # Filter by archived status if specified
        if archived is not None:
            result = [nb for nb in result if nb.get("archived") == archived]

        return [
            NotebookResponse(
                id=str(nb.get("id", "")),
                name=nb.get("name", ""),
                description=nb.get("description", ""),
                archived=nb.get("archived", False),
                created=str(nb.get("created", "")),
                updated=str(nb.get("updated", "")),
                source_count=nb.get("source_count", 0),
            )
            for nb in result
        ]
    except Exception as e:
        logger.error(f"Error fetching notebooks: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching notebooks: {str(e)}"
        )


@router.post("/notebooks", response_model=NotebookResponse)
async def create_notebook(notebook: NotebookCreate):
    """Create a new notebook."""
    try:
        new_notebook = Notebook(
            name=notebook.name,
            description=notebook.description,
        )
        await new_notebook.save()

        return NotebookResponse(
            id=new_notebook.id or "",
            name=new_notebook.name,
            description=new_notebook.description,
            archived=new_notebook.archived or False,
            created=str(new_notebook.created),
            updated=str(new_notebook.updated),
            source_count=0,  # New notebook has no sources
        )
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating notebook: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error creating notebook: {str(e)}"
        )


@router.get("/notebooks/{notebook_id}", response_model=NotebookResponse)
async def get_notebook(notebook_id: str):
    """Get a specific notebook by ID."""
    try:
        # Query with counts for single notebook
        query = """
            SELECT *,
            count(<-reference.in) as source_count
            FROM $notebook_id
        """
        result = await repo_query(query, {"notebook_id": ensure_record_id(notebook_id)})

        if not result:
            raise HTTPException(status_code=404, detail="Notebook not found")

        nb = result[0]
        return NotebookResponse(
            id=str(nb.get("id", "")),
            name=nb.get("name", ""),
            description=nb.get("description", ""),
            archived=nb.get("archived", False),
            created=str(nb.get("created", "")),
            updated=str(nb.get("updated", "")),
            source_count=nb.get("source_count", 0),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching notebook {notebook_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching notebook: {str(e)}"
        )


@router.put("/notebooks/{notebook_id}", response_model=NotebookResponse)
async def update_notebook(notebook_id: str, notebook_update: NotebookUpdate):
    """Update a notebook."""
    try:
        notebook = await Notebook.get(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

        # Update only provided fields
        if notebook_update.name is not None:
            notebook.name = notebook_update.name
        if notebook_update.description is not None:
            notebook.description = notebook_update.description
        if notebook_update.archived is not None:
            notebook.archived = notebook_update.archived

        await notebook.save()

        # Query with counts after update
        query = """
            SELECT *,
            count(<-reference.in) as source_count
            FROM $notebook_id
        """
        result = await repo_query(query, {"notebook_id": ensure_record_id(notebook_id)})

        if result:
            nb = result[0]
            return NotebookResponse(
                id=str(nb.get("id", "")),
                name=nb.get("name", ""),
                description=nb.get("description", ""),
                archived=nb.get("archived", False),
                created=str(nb.get("created", "")),
                updated=str(nb.get("updated", "")),
                source_count=nb.get("source_count", 0),
            )

        # Fallback if query fails
        return NotebookResponse(
            id=notebook.id or "",
            name=notebook.name,
            description=notebook.description,
            archived=notebook.archived or False,
            created=str(notebook.created),
            updated=str(notebook.updated),
            source_count=0,
        )
    except HTTPException:
        raise
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating notebook {notebook_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error updating notebook: {str(e)}"
        )


@router.post("/notebooks/{notebook_id}/sources/{source_id}")
async def add_source_to_notebook(notebook_id: str, source_id: str):
    """Add an existing source to a notebook (create the reference)."""
    try:
        # Check if notebook exists
        notebook = await Notebook.get(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

        # Check if source exists
        source = await Source.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Check if reference already exists (idempotency)
        existing_ref = await repo_query(
            "SELECT * FROM reference WHERE out = $source_id AND in = $notebook_id",
            {
                "notebook_id": ensure_record_id(notebook_id),
                "source_id": ensure_record_id(source_id),
            },
        )

        # If reference doesn't exist, create it
        if not existing_ref:
            await repo_query(
                "RELATE $source_id->reference->$notebook_id",
                {
                    "notebook_id": ensure_record_id(notebook_id),
                    "source_id": ensure_record_id(source_id),
                },
            )

        return {"message": "Source linked to notebook successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error linking source {source_id} to notebook {notebook_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Error linking source to notebook: {str(e)}"
        )


@router.delete("/notebooks/{notebook_id}/sources/{source_id}")
async def remove_source_from_notebook(notebook_id: str, source_id: str):
    """Remove a source from a notebook (delete the reference)."""
    try:
        # Check if notebook exists
        notebook = await Notebook.get(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

        # Delete the reference record linking source to notebook
        await repo_query(
            "DELETE FROM reference WHERE out = $notebook_id AND in = $source_id",
            {
                "notebook_id": ensure_record_id(notebook_id),
                "source_id": ensure_record_id(source_id),
            },
        )

        return {"message": "Source removed from notebook successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error removing source {source_id} from notebook {notebook_id}: {str(e)}"
        )
        raise HTTPException(
            status_code=500, detail=f"Error removing source from notebook: {str(e)}"
        )


@router.delete("/notebooks/{notebook_id}")
async def delete_notebook(notebook_id: str):
    """Delete a notebook."""
    try:
        notebook = await Notebook.get(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

        await notebook.delete()

        return {"message": "Notebook deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting notebook {notebook_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting notebook: {str(e)}"
        )


@router.post("/notebooks/{notebook_id}/build-pageindex")
async def build_pageindex_for_notebook(notebook_id: str):
    """
    為 notebook 中的所有 sources 建立 PageIndex 索引。
    
    這是一個批量操作，會為所有有內容的 sources 建立索引。
    已存在的索引不會重新建立。
    """
    try:
        from open_notebook.services.pageindex_service import pageindex_service
        
        logger.info(f"Building PageIndex for notebook {notebook_id}")
        
        if not pageindex_service.is_available():
            logger.error(f"PageIndex is not available for notebook {notebook_id}")
            raise HTTPException(
                status_code=503, 
                detail="PageIndex is not available. Please ensure PageIndex is installed."
            )
        
        notebook = await Notebook.get(notebook_id)
        if not notebook:
            logger.error(f"Notebook {notebook_id} not found")
            raise HTTPException(status_code=404, detail="Notebook not found")
        
        sources = await notebook.get_sources()
        logger.info(f"Found {len(sources)} sources in notebook {notebook_id}")
        
        if not sources:
            return {
                "status": "completed",
                "message": f"No sources found in notebook {notebook_id}",
                "notebook_id": notebook_id,
                "total": 0,
                "success": 0,
                "skipped": 0,
                "failed": 0,
                "results": []
            }
        
        results = []
        success_count = 0
        skipped_count = 0
        failed_count = 0
        
        for idx, source in enumerate(sources, 1):
            logger.info(f"[{idx}/{len(sources)}] Processing source {source.id} ({source.title})")
            result_item = {
                "source_id": source.id,
                "source_title": source.title,
            }
            
            try:
                # 獲取完整的 source
                full_source = await Source.get(source.id)
                
                # 檢查是否已有索引
                if full_source.pageindex_structure:
                    logger.info(f"Source {source.id} already has PageIndex structure, skipping build")
                    result_item["status"] = "skipped"
                    result_item["reason"] = "already exists"
                    skipped_count += 1
                elif not full_source.full_text:
                    # 沒有 full_text，檢查是否可以從文件路徑讀取（僅 PDF）
                    if full_source.asset and full_source.asset.file_path:
                        file_ext = os.path.splitext(full_source.asset.file_path)[1].lower()
                        if file_ext == '.pdf':
                            logger.info(f"Source {source.id} has no full_text but is PDF, attempting to build from file path")
                            try:
                                await pageindex_service._get_or_create_index_for_source(full_source.id, full_source)
                                logger.info(f"Successfully built PageIndex for source {source.id} from file path")
                                result_item["status"] = "success"
                                result_item["reason"] = "built from file path"
                                success_count += 1
                            except Exception as e:
                                logger.error(f"Failed to build PageIndex from file path for source {source.id}: {e}")
                                result_item["status"] = "error"
                                result_item["error"] = str(e)
                                result_item["reason"] = "file path read failed"
                                failed_count += 1
                        else:
                            logger.info(f"Source {source.id} has no content and is not PDF ({file_ext}), skipping")
                            result_item["status"] = "skipped"
                            result_item["reason"] = f"no content and not PDF ({file_ext})"
                            skipped_count += 1
                    else:
                        logger.info(f"Source {source.id} has no content and no file path, skipping")
                        result_item["status"] = "skipped"
                        result_item["reason"] = "no content and no file path"
                        skipped_count += 1
                else:
                    # 有 full_text，正常建立索引
                    logger.info(f"Building PageIndex for source {source.id} (content length: {len(full_source.full_text)} chars)")
                    logger.info(f"Starting PageIndex build for source {source.id}...")
                    await pageindex_service._get_or_create_index_for_source(full_source.id, full_source)
                    logger.info(f"Successfully built PageIndex for source {source.id}")
                    result_item["status"] = "success"
                    result_item["reason"] = "built from full_text"
                    success_count += 1
            except Exception as e:
                logger.error(f"Failed to build PageIndex for source {source.id}: {e}")
                logger.exception(e)
                result_item["status"] = "error"
                result_item["error"] = str(e)
                failed_count += 1
            
            results.append(result_item)
        
        logger.info(
            f"PageIndex building completed for notebook {notebook_id}: "
            f"total={len(sources)}, success={success_count}, skipped={skipped_count}, failed={failed_count}"
        )
        
        return {
            "status": "completed",
            "message": f"PageIndex building completed for notebook {notebook_id}",
            "notebook_id": notebook_id,
            "total": len(sources),
            "success": success_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "results": results
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building PageIndex for notebook {notebook_id}: {e}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))