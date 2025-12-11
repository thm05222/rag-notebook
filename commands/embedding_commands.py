import time
from typing import Dict, List, Literal, Optional

from loguru import logger
from pydantic import BaseModel
from qdrant_client.models import FieldCondition, Filter, MatchValue
from surreal_commands import CommandInput, CommandOutput, command

from open_notebook.database.repository import ensure_record_id, repo_query
from open_notebook.domain.models import model_manager
from open_notebook.domain.notebook import Source, SourceInsight, get_source_notebook_ids


def full_model_dump(model):
    if isinstance(model, BaseModel):
        return model.model_dump()
    elif isinstance(model, dict):
        return {k: full_model_dump(v) for k, v in model.items()}
    elif isinstance(model, list):
        return [full_model_dump(item) for item in model]
    else:
        return model


class EmbedSingleItemInput(CommandInput):
    item_id: str
    item_type: Literal["source", "insight"]


class EmbedSingleItemOutput(CommandOutput):
    success: bool
    item_id: str
    item_type: str
    chunks_created: int = 0  # For sources
    processing_time: float
    error_message: Optional[str] = None


class RebuildEmbeddingsInput(CommandInput):
    mode: Literal["existing", "all"]
    include_sources: bool = True
    include_insights: bool = True


class RebuildEmbeddingsOutput(CommandOutput):
    success: bool
    total_items: int
    processed_items: int
    failed_items: int
    sources_processed: int = 0
    insights_processed: int = 0
    processing_time: float
    error_message: Optional[str] = None


@command("embed_single_item", app="open_notebook")
async def embed_single_item_command(
    input_data: EmbedSingleItemInput,
) -> EmbedSingleItemOutput:
    """
    Embed a single item (source or insight)
    """
    start_time = time.time()

    try:
        logger.info(
            f"Starting embedding for {input_data.item_type}: {input_data.item_id}"
        )

        # Check if embedding model is available
        EMBEDDING_MODEL = await model_manager.get_embedding_model()
        if not EMBEDDING_MODEL:
            raise ValueError(
                "No embedding model configured. Please configure one in the Models section."
            )

        chunks_created = 0

        if input_data.item_type == "source":
            # Get source and vectorize
            source = await Source.get(input_data.item_id)
            if not source:
                raise ValueError(f"Source '{input_data.item_id}' not found")

            await source.vectorize()

            # Count chunks created from Qdrant
            from open_notebook.services.qdrant_service import qdrant_service
            try:
                # Ensure Qdrant client is initialized
                client = await qdrant_service._ensure_client()
                # Use scroll to count points for this source
                scroll_result = client.scroll(
                    collection_name="source_embeddings",
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="source_id",
                                match=MatchValue(value=input_data.item_id)
                            )
                        ]
                    ),
                    limit=1000  # Adjust based on expected size
                )
                chunks_created = len(scroll_result[0]) if scroll_result[0] else 0
            except Exception as e:
                logger.warning(f"Could not count chunks from Qdrant: {e}")
                chunks_created = 0

            logger.info(f"Source vectorized: {chunks_created} chunks created")

        elif input_data.item_type == "insight":
            # Get insight and re-generate embedding
            insight = await SourceInsight.get(input_data.item_id)
            if not insight:
                raise ValueError(f"Insight '{input_data.item_id}' not found")

            # Generate new embedding
            embedding = (await EMBEDDING_MODEL.aembed([insight.content]))[0]

            # Get source for this insight to get notebook associations
            source = await insight.get_source()
            notebook_ids = await get_source_notebook_ids(source.id)

            # Store insight embedding in Qdrant
            from open_notebook.services.qdrant_service import qdrant_service
            await qdrant_service.store_source_insight(
                source_id=source.id,
                insight_type=insight.insight_type,
                content=insight.content,
                embedding=embedding,
                notebook_ids=notebook_ids,
                insight_id=insight.id
            )
            logger.info(f"Insight embedded: {input_data.item_id}")

        else:
            raise ValueError(
                f"Invalid item_type: {input_data.item_type}. Must be 'source' or 'insight'"
            )

        processing_time = time.time() - start_time
        logger.info(
            f"Successfully embedded {input_data.item_type} {input_data.item_id} in {processing_time:.2f}s"
        )

        return EmbedSingleItemOutput(
            success=True,
            item_id=input_data.item_id,
            item_type=input_data.item_type,
            chunks_created=chunks_created,
            processing_time=processing_time,
        )

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Embedding failed for {input_data.item_type} {input_data.item_id}: {e}")
        logger.exception(e)

        return EmbedSingleItemOutput(
            success=False,
            item_id=input_data.item_id,
            item_type=input_data.item_type,
            processing_time=processing_time,
            error_message=str(e),
        )


async def collect_items_for_rebuild(
    mode: str,
    include_sources: bool,
    include_insights: bool,
) -> Dict[str, List[str]]:
    """
    Collect items to rebuild based on mode and include flags.

    Returns:
        Dict with keys: 'sources', 'insights' containing lists of item IDs
    """
    items: Dict[str, List[str]] = {"sources": [], "insights": []}

    if include_sources:
        if mode == "existing":
            # Query sources with embeddings from Qdrant
            from open_notebook.services.qdrant_service import qdrant_service
            try:
                # Ensure Qdrant client is initialized
                client = await qdrant_service._ensure_client()
                # Get all source IDs from Qdrant source_embeddings collection
                scroll_result = client.scroll(
                    collection_name="source_embeddings",
                    limit=10000  # Large limit to get all sources
                )
                source_ids = set()
                if scroll_result[0]:
                    for point in scroll_result[0]:
                        source_ids.add(point.payload.get("source_id"))
                items["sources"] = list(source_ids)
            except Exception as e:
                logger.warning(f"Could not query sources from Qdrant: {e}")
                items["sources"] = []
        else:  # mode == "all"
            # Query all sources with content
            result = await repo_query("SELECT id FROM source WHERE full_text != none")
            items["sources"] = [str(item["id"]) for item in result] if result else []

        logger.info(f"Collected {len(items['sources'])} sources for rebuild")

    if include_insights:
        if mode == "existing":
            # Query insights with embeddings from Qdrant
            from open_notebook.services.qdrant_service import qdrant_service
            try:
                # Ensure Qdrant client is initialized
                client = await qdrant_service._ensure_client()
                scroll_result = client.scroll(
                    collection_name="source_insights",
                    limit=10000
                )
                insight_ids = set()
                if scroll_result[0]:
                    for point in scroll_result[0]:
                        # For insights, we can get insight_id directly from payload if available
                        # Otherwise fall back to querying SurrealDB
                        insight_id = point.payload.get("insight_id")
                        if insight_id:
                            insight_ids.add(insight_id)
                        else:
                            # Fallback: get insight ID from SurrealDB using source_id
                            source_id = point.payload.get("source_id")
                            if source_id:
                                # Get all insights for this source
                                insight_result = await repo_query(
                                    "SELECT id FROM source_insight WHERE source = $source_id",
                                    {"source_id": source_id}
                                )
                                if insight_result:
                                    for insight in insight_result:
                                        insight_ids.add(insight["id"])
                items["insights"] = list(insight_ids)
            except Exception as e:
                logger.warning(f"Could not query insights from Qdrant: {e}")
                items["insights"] = []
        else:  # mode == "all"
            # Query all insights
            result = await repo_query("SELECT id FROM source_insight")
            items["insights"] = [str(item["id"]) for item in result] if result else []

        logger.info(f"Collected {len(items['insights'])} insights for rebuild")

    return items


@command("rebuild_embeddings", app="open_notebook")
async def rebuild_embeddings_command(
    input_data: RebuildEmbeddingsInput,
) -> RebuildEmbeddingsOutput:
    """
    Rebuild embeddings for sources and/or insights
    """
    start_time = time.time()

    try:
        logger.info("=" * 60)
        logger.info(f"Starting embedding rebuild with mode={input_data.mode}")
        logger.info(f"Include: sources={input_data.include_sources}, insights={input_data.include_insights}")
        logger.info("=" * 60)

        # Check embedding model availability
        EMBEDDING_MODEL = await model_manager.get_embedding_model()
        if not EMBEDDING_MODEL:
            raise ValueError(
                "No embedding model configured. Please configure one in the Models section."
            )

        logger.info(f"Using embedding model: {EMBEDDING_MODEL}")
        
        # Check and recreate collections if dimension mismatch is detected
        from open_notebook.services.qdrant_service import qdrant_service
        try:
            collections_recreated = await qdrant_service.check_and_recreate_collections_if_needed()
            if collections_recreated:
                logger.info("Collections were recreated due to dimension mismatch. All existing embeddings have been deleted.")
        except Exception as e:
            logger.warning(f"Failed to check/recreate collections: {e}. Will continue with existing collections.")
            # Don't fail the rebuild if collection check fails, let it try to process and fail with clearer error
        
        # Ensure collections exist (will be created if needed)
        await qdrant_service._ensure_collections()

        # Collect items to process
        items = await collect_items_for_rebuild(
            input_data.mode,
            input_data.include_sources,
            input_data.include_insights,
        )

        total_items = (
            len(items["sources"]) + len(items["insights"])
        )
        logger.info(f"Total items to process: {total_items}")

        if total_items == 0:
            logger.warning("No items found to rebuild")
            return RebuildEmbeddingsOutput(
                success=True,
                total_items=0,
                processed_items=0,
                failed_items=0,
                processing_time=time.time() - start_time,
            )

        # Initialize counters
        sources_processed = 0
        insights_processed = 0
        failed_items = 0

        # Process sources
        logger.info(f"\nProcessing {len(items['sources'])} sources...")
        for idx, source_id in enumerate(items["sources"], 1):
            try:
                source = await Source.get(source_id)
                if not source:
                    logger.warning(f"Source {source_id} not found, skipping")
                    failed_items += 1
                    continue

                # Check if source has content to vectorize
                if not source.full_text:
                    logger.warning(f"Source {source_id} has no content to vectorize, skipping")
                    failed_items += 1
                    continue

                await source.vectorize()
                sources_processed += 1

                if idx % 10 == 0 or idx == len(items["sources"]):
                    logger.info(
                        f"  Progress: {idx}/{len(items['sources'])} sources processed"
                    )

            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to re-embed source {source_id}: {error_msg}")
                logger.exception(e)  # Log full traceback for debugging
                failed_items += 1

        # Process insights
        logger.info(f"\nProcessing {len(items['insights'])} insights...")
        for idx, insight_id in enumerate(items["insights"], 1):
            try:
                insight = await SourceInsight.get(insight_id)
                if not insight:
                    logger.warning(f"Insight {insight_id} not found, skipping")
                    failed_items += 1
                    continue

                # Re-generate embedding
                if not insight.content or not insight.content.strip():
                    logger.warning(f"Insight {insight_id} has no content, skipping")
                    failed_items += 1
                    continue

                embedding = (await EMBEDDING_MODEL.aembed([insight.content]))[0]

                # Update insight with new embedding in SurrealDB
                await repo_query(
                    "UPDATE $insight_id SET embedding = $embedding",
                    {
                        "insight_id": ensure_record_id(insight_id),
                        "embedding": embedding,
                    },
                )

                # Update insight in Qdrant
                from open_notebook.services.qdrant_service import qdrant_service
                from open_notebook.domain.notebook import get_source_notebook_ids
                source = await insight.get_source()
                if not source:
                    logger.warning(f"Source for insight {insight_id} not found, skipping Qdrant update")
                    failed_items += 1
                    continue
                
                notebook_ids = await get_source_notebook_ids(source.id)
                await qdrant_service.store_source_insight(
                    source_id=source.id,
                    insight_type=insight.insight_type,
                    content=insight.content,
                    embedding=embedding,
                    notebook_ids=notebook_ids,
                    insight_id=insight.id
                )
                insights_processed += 1

                if idx % 10 == 0 or idx == len(items["insights"]):
                    logger.info(
                        f"  Progress: {idx}/{len(items['insights'])} insights processed"
                    )

            except Exception as e:
                logger.error(f"Failed to re-embed insight {insight_id}: {e}")
                logger.exception(e)  # Log full traceback for debugging
                failed_items += 1

        processing_time = time.time() - start_time
        processed_items = sources_processed + insights_processed

        logger.info("=" * 60)
        logger.info("REBUILD COMPLETE")
        logger.info(f"  Total processed: {processed_items}/{total_items}")
        logger.info(f"  Sources: {sources_processed}")
        logger.info(f"  Insights: {insights_processed}")
        logger.info(f"  Failed: {failed_items}")
        logger.info(f"  Time: {processing_time:.2f}s")
        logger.info("=" * 60)

        return RebuildEmbeddingsOutput(
            success=True,
            total_items=total_items,
            processed_items=processed_items,
            failed_items=failed_items,
            sources_processed=sources_processed,
            insights_processed=insights_processed,
            processing_time=processing_time,
        )

    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        logger.error(f"Rebuild embeddings failed: {error_msg}")
        logger.exception(e)

        # Ensure error message is detailed and informative
        # Include exception type and message for better debugging
        detailed_error = f"{type(e).__name__}: {error_msg}"
        if hasattr(e, '__cause__') and e.__cause__:
            detailed_error += f" (caused by: {type(e.__cause__).__name__}: {str(e.__cause__)})"

        return RebuildEmbeddingsOutput(
            success=False,
            total_items=0,
            processed_items=0,
            failed_items=0,
            processing_time=processing_time,
            error_message=detailed_error,
        )
