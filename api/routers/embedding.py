from fastapi import APIRouter, HTTPException
from loguru import logger

from api.command_service import CommandService
from api.models import EmbedRequest, EmbedResponse
from open_notebook.domain.models import model_manager
from open_notebook.domain.notebook import Source

router = APIRouter()


@router.post("/embed", response_model=EmbedResponse)
async def embed_content(embed_request: EmbedRequest):
    """Embed content for vector search."""
    try:
        # Check if embedding model is available
        try:
            embedding_model = await model_manager.get_embedding_model()
            if not embedding_model:
                raise HTTPException(
                    status_code=400,
                    detail="No embedding model configured. Please configure one in the Models section.",
                )
        except ValueError as e:
            # Model configuration error (e.g., model not found)
            error_msg = str(e)
            logger.error(f"Embedding model configuration error: {error_msg}")
            raise HTTPException(
                status_code=400,
                detail=f"Embedding model configuration error: {error_msg}. Please configure a valid embedding model in Settings > Models.",
            )
        except Exception as e:
            logger.error(f"Failed to get embedding model: {e}")
            logger.exception(e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load embedding model: {str(e)}. Please check model configuration."
            )

        item_id = embed_request.item_id
        item_type = embed_request.item_type.lower()

        # Validate item type
        if item_type != "source":
            raise HTTPException(
                status_code=400, detail="Item type must be 'source'"
            )

        # Branch based on processing mode
        if embed_request.async_processing:
            # ASYNC PATH: Submit command for background processing
            logger.info(f"Using async processing for {item_type} {item_id}")

            try:
                # Import commands to ensure they're registered
                import commands.embedding_commands  # noqa: F401

                # Submit command
                command_id = await CommandService.submit_command_job(
                    "open_notebook",  # app name
                    "embed_single_item",  # command name
                    {"item_id": item_id, "item_type": item_type},
                )

                logger.info(f"Submitted async embedding command: {command_id}")

                return EmbedResponse(
                    success=True,
                    message="Embedding queued for background processing",
                    item_id=item_id,
                    item_type=item_type,
                    command_id=command_id,
                )

            except Exception as e:
                logger.error(f"Failed to submit async embedding command: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Failed to queue embedding: {str(e)}"
                )

        else:
            # SYNC PATH: Execute synchronously (existing behavior)
            logger.info(f"Using sync processing for {item_type} {item_id}")

            # Get the item and embed it
            if item_type == "source":
                source_item = await Source.get(item_id)
                if not source_item:
                    raise HTTPException(status_code=404, detail="Source not found")

                # Check if source has content
                if not source_item.full_text or not source_item.full_text.strip():
                    raise HTTPException(
                        status_code=400,
                        detail="Source has no content to embed. Please ensure the source has been processed."
                    )

                # Perform embedding (vectorize is now idempotent - safe to call multiple times)
                try:
                    await source_item.vectorize()
                    message = "Source embedded successfully"
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Failed to vectorize source {item_id}: {error_msg}")
                    logger.exception(e)
                    # Provide more detailed error message
                    if "dimension" in error_msg.lower() or "dim" in error_msg.lower():
                        # Vector dimension mismatch - usually happens when switching embedding models
                        raise HTTPException(
                            status_code=400,
                            detail=f"Vector dimension mismatch: {error_msg}. "
                                   f"This usually happens when switching embedding models. "
                                   f"The Qdrant collection was created with a different embedding model. "
                                   f"Please delete the Qdrant collections or use 'Rebuild Embeddings' to migrate data."
                        )
                    elif "All chunks failed to process" in error_msg:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to process source content: {error_msg}. Check embedding model configuration and Qdrant connection."
                        )
                    elif "Qdrant" in error_msg or "connection" in error_msg.lower():
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to connect to Qdrant: {error_msg}. Please check Qdrant service status."
                        )
                    elif "embedding model" in error_msg.lower() or "model" in error_msg.lower():
                        raise HTTPException(
                            status_code=400,
                            detail=f"Embedding model error: {error_msg}. Please configure an embedding model in Settings > Models."
                        )
                    else:
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to embed source: {error_msg}"
                        )

            else:
                raise HTTPException(status_code=400, detail=f"Invalid item_type: {item_type}. Only 'source' is supported.")

            return EmbedResponse(
                success=True, message=message, item_id=item_id, item_type=item_type, command_id=None
            )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(
            f"Error embedding {embed_request.item_type} {embed_request.item_id}: {error_msg}"
        )
        logger.exception(e)
        raise HTTPException(
            status_code=500, detail=f"Error embedding content: {error_msg}"
        )
