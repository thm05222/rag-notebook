import asyncio
from datetime import datetime
from typing import Any, ClassVar, Dict, List, Literal, Optional, Tuple, Union

from loguru import logger
from pydantic import BaseModel, Field, field_validator
from surrealdb import RecordID

from open_notebook.database.repository import ensure_record_id, repo_query
from open_notebook.domain.base import ObjectModel
from open_notebook.domain.models import model_manager
from open_notebook.exceptions import DatabaseOperationError, InvalidInputError
from open_notebook.utils import split_text


class Notebook(ObjectModel):
    table_name: ClassVar[str] = "notebook"
    name: str
    description: str
    archived: Optional[bool] = False

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v):
        if not v.strip():
            raise InvalidInputError("Notebook name cannot be empty")
        return v

    async def get_sources(self) -> List["Source"]:
        try:
            srcs = await repo_query(
                """
                select * omit source.full_text from (
                select in as source from reference where out=$id
                fetch source
            ) order by source.updated desc
            """,
                {"id": ensure_record_id(self.id)},
            )
            return [Source(**src["source"]) for src in srcs] if srcs else []
        except Exception as e:
            logger.error(f"Error fetching sources for notebook {self.id}: {str(e)}")
            logger.exception(e)
            raise DatabaseOperationError(e)

    async def get_chat_sessions(self) -> List["ChatSession"]:
        try:
            srcs = await repo_query(
                """
                select * from (
                    select
                    <- chat_session as chat_session
                    from refers_to
                    where out=$id
                    fetch chat_session
                )
                order by chat_session.updated desc
            """,
                {"id": ensure_record_id(self.id)},
            )
            return (
                [ChatSession(**src["chat_session"][0]) for src in srcs] if srcs else []
            )
        except Exception as e:
            logger.error(
                f"Error fetching chat sessions for notebook {self.id}: {str(e)}"
            )
            logger.exception(e)
            raise DatabaseOperationError(e)


class Asset(BaseModel):
    file_path: Optional[str] = None
    url: Optional[str] = None


class SourceEmbedding(ObjectModel):
    table_name: ClassVar[str] = "source_embedding"
    content: str

    async def get_source(self) -> "Source":
        try:
            src = await repo_query(
                """
            select source.* from $id fetch source
            """,
                {"id": ensure_record_id(self.id)},
            )
            return Source(**src[0]["source"])
        except Exception as e:
            logger.error(f"Error fetching source for embedding {self.id}: {str(e)}")
            logger.exception(e)
            raise DatabaseOperationError(e)


class SourceInsight(ObjectModel):
    table_name: ClassVar[str] = "source_insight"
    insight_type: str
    content: str

    async def get_source(self) -> "Source":
        try:
            src = await repo_query(
                """
            select source.* from $id fetch source
            """,
                {"id": ensure_record_id(self.id)},
            )
            return Source(**src[0]["source"])
        except Exception as e:
            logger.error(f"Error fetching source for insight {self.id}: {str(e)}")
            logger.exception(e)
            raise DatabaseOperationError(e)


class Source(ObjectModel):
    table_name: ClassVar[str] = "source"
    asset: Optional[Asset] = None
    title: Optional[str] = None
    topics: Optional[List[str]] = Field(default_factory=list)
    full_text: Optional[str] = None
    command: Optional[Union[str, RecordID]] = Field(
        default=None, description="Link to surreal-commands processing job"
    )
    # PageIndex fields for persistent storage
    pageindex_structure: Optional[Dict[str, Any]] = Field(
        default=None, description="PageIndex tree structure (nested object)"
    )
    pageindex_built_at: Optional[datetime] = Field(
        default=None, description="Timestamp when PageIndex was built"
    )
    pageindex_model: Optional[str] = Field(
        default=None, description="Model ID used to build PageIndex"
    )
    pageindex_version: Optional[str] = Field(
        default=None, description="PageIndex version for compatibility"
    )

    class Config:
        arbitrary_types_allowed = True

    @field_validator("command", mode="before")
    @classmethod
    def parse_command(cls, value):
        """Parse command field to ensure RecordID format"""
        if isinstance(value, str) and value:
            return ensure_record_id(value)
        return value

    @field_validator("id", mode="before")
    @classmethod
    def parse_id(cls, value):
        """Parse id field to handle both string and RecordID inputs"""
        if value is None:
            return None
        if isinstance(value, RecordID):
            return str(value)
        return str(value) if value else None

    async def get_status(self) -> Optional[str]:
        """Get the processing status of the associated command"""
        if not self.command:
            return None

        try:
            from surreal_commands import get_command_status

            status = await get_command_status(str(self.command))
            return status.status if status else "unknown"
        except Exception as e:
            logger.warning(f"Failed to get command status for {self.command}: {e}")
            return "unknown"

    async def get_processing_progress(self) -> Optional[Dict[str, Any]]:
        """Get detailed processing information for the associated command"""
        if not self.command:
            return None

        try:
            from surreal_commands import get_command_status

            status_result = await get_command_status(str(self.command))
            if not status_result:
                return None

            # Extract execution metadata if available
            result = getattr(status_result, "result", None)
            execution_metadata = result.get("execution_metadata", {}) if isinstance(result, dict) else {}

            return {
                "status": status_result.status,
                "started_at": execution_metadata.get("started_at"),
                "completed_at": execution_metadata.get("completed_at"),
                "error": getattr(status_result, "error_message", None),
                "result": result,
            }
        except Exception as e:
            logger.warning(f"Failed to get command progress for {self.command}: {e}")
            return None

    async def get_context(
        self, context_size: Literal["short", "long"] = "short"
    ) -> Dict[str, Any]:
        insights_list = await self.get_insights()
        insights = [insight.model_dump() for insight in insights_list]
        if context_size == "long":
            return dict(
                id=self.id,
                title=self.title,
                insights=insights,
                full_text=self.full_text,
            )
        else:
            return dict(id=self.id, title=self.title, insights=insights)

    async def get_embedded_chunks(self) -> int:
        """
        Get the number of embedded chunks for this source from Qdrant
        
        Returns:
            int: Number of chunks stored in Qdrant
        """
        try:
            from open_notebook.services.qdrant_service import qdrant_service
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            
            # Ensure Qdrant client is available
            if qdrant_service.client is None:
                await qdrant_service._ensure_client()
            
            client = qdrant_service.client
            if client is None:
                logger.warning(f"Qdrant client not available for source {self.id}")
                return 0
            
            # Use scroll to count points for this source
            scroll_result = client.scroll(
                collection_name="source_embeddings",
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="source_id",
                            match=MatchValue(value=self.id)
                        )
                    ]
                ),
                limit=10000  # Large enough to get all chunks for one source
            )
            
            # scroll returns (points, next_page_offset)
            chunks_count = len(scroll_result[0]) if scroll_result and scroll_result[0] else 0
            return chunks_count
            
        except Exception as e:
            logger.warning(f"Error fetching chunks count from Qdrant for source {self.id}: {str(e)}")
            # Return 0 instead of raising to avoid breaking the API
            return 0

    async def get_insights(self) -> List[SourceInsight]:
        try:
            result = await repo_query(
                """
                SELECT * FROM source_insight WHERE source=$id
                """,
                {"id": ensure_record_id(self.id)},
            )
            return [SourceInsight(**insight) for insight in result]
        except Exception as e:
            logger.error(f"Error fetching insights for source {self.id}: {str(e)}")
            logger.exception(e)
            raise DatabaseOperationError("Failed to fetch insights for source")

    async def add_to_notebook(self, notebook_id: str) -> Any:
        if not notebook_id:
            raise InvalidInputError("Notebook ID must be provided")
        
        # Create the relationship in SurrealDB
        result = await self.relate("reference", notebook_id)
        
        # Update Qdrant with new notebook association
        try:
            from open_notebook.services.qdrant_service import qdrant_service
            notebook_ids = await get_source_notebook_ids(self.id)
            await qdrant_service.update_notebook_associations(
                item_id=self.id,
                item_type="source",
                notebook_ids=notebook_ids
            )
            logger.info(f"Updated Qdrant notebook associations for source {self.id}")
        except Exception as e:
            logger.warning(f"Failed to update Qdrant notebook associations: {e}")
        
        return result

    async def vectorize(self) -> None:
        logger.info(f"Starting vectorization for source {self.id}")
        
        # Check embedding model first
        EMBEDDING_MODEL = await model_manager.get_embedding_model()
        if not EMBEDDING_MODEL:
            raise ValueError("No embedding model configured. Please configure one in the Models section.")

        try:
            # Get notebook IDs for this source
            notebook_ids = await get_source_notebook_ids(self.id)
            logger.debug(f"Source {self.id} belongs to notebooks: {notebook_ids}")

            # Import Qdrant service
            from open_notebook.services.qdrant_service import qdrant_service

            # Ensure Qdrant client is available before proceeding
            try:
                await qdrant_service._ensure_client()
                await qdrant_service._ensure_collections()
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                raise ValueError(f"Failed to connect to Qdrant: {str(e)}. Please check Qdrant service status.")

            # DELETE EXISTING EMBEDDINGS FIRST - Makes vectorize() idempotent
            try:
                await qdrant_service.delete_source_embeddings(self.id)
                logger.info(f"Deleted existing embeddings for source {self.id}")
            except Exception as e:
                logger.warning(f"Failed to delete existing embeddings (may not exist): {e}")
                # Continue anyway - this is not critical

            if not self.full_text:
                logger.warning(f"No text to vectorize for source {self.id}")
                raise ValueError(f"Source {self.id} has no content to vectorize. Please ensure the source has been processed.")

            if not self.full_text.strip():
                logger.warning(f"Source {self.id} has empty content")
                raise ValueError(f"Source {self.id} has empty content. Please ensure the source has valid content.")

            # Split text into chunks
            try:
                chunks = split_text(
                    self.full_text,
                )
            except Exception as e:
                logger.error(f"Failed to split text for source {self.id}: {e}")
                logger.exception(e)
                raise ValueError(f"Failed to split source {self.id} into chunks: {str(e)}")
            
            chunk_count = len(chunks)
            logger.info(f"Split into {chunk_count} chunks for source {self.id}")

            if chunk_count == 0:
                logger.warning("No chunks created after splitting")
                raise ValueError(f"Failed to split source {self.id} into chunks. The content may be too short or invalid.")

            # Process chunks concurrently using async gather
            logger.info("Starting concurrent processing of chunks")

            async def process_chunk(
                idx: int, chunk: str
            ) -> Tuple[int, List[float], str]:
                logger.debug(f"Processing chunk {idx}/{chunk_count}")
                try:
                    if EMBEDDING_MODEL is None:
                        raise ValueError("EMBEDDING_MODEL is not configured")
                    
                    # Validate chunk content
                    if not chunk or not chunk.strip():
                        logger.warning(f"Chunk {idx} is empty, skipping")
                        raise ValueError(f"Chunk {idx} is empty")
                    
                    # Generate embedding
                    try:
                        embedding = (await EMBEDDING_MODEL.aembed([chunk]))[0]
                    except Exception as e:
                        logger.error(f"Embedding model failed for chunk {idx}: {e}")
                        raise ValueError(f"Failed to generate embedding for chunk {idx}: {str(e)}")
                    
                    # Validate embedding
                    if not embedding or not isinstance(embedding, list) or len(embedding) == 0:
                        logger.error(f"Invalid embedding returned for chunk {idx}: {type(embedding)}, length: {len(embedding) if isinstance(embedding, list) else 'N/A'}")
                        raise ValueError(f"Invalid embedding returned for chunk {idx}")
                    
                    cleaned_content = chunk
                    logger.debug(f"Successfully processed chunk {idx} (embedding dimension: {len(embedding)})")
                    return (idx, embedding, cleaned_content)
                except Exception as e:
                    logger.error(f"Error processing chunk {idx}: {str(e)}")
                    raise

            # Create tasks for all chunks and process them concurrently
            # Use return_exceptions=True to handle individual chunk failures gracefully
            tasks = [process_chunk(idx, chunk) for idx, chunk in enumerate(chunks)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and log them
            valid_results = []
            failed_chunks = []
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to process chunk {idx}/{chunk_count}: {result}")
                    failed_chunks.append((idx, str(result)))
                else:
                    valid_results.append(result)
            
            if not valid_results:
                error_details = "; ".join([f"chunk {idx}: {err}" for idx, err in failed_chunks[:5]])
                if len(failed_chunks) > 5:
                    error_details += f" ... and {len(failed_chunks) - 5} more"
                raise ValueError(f"All {chunk_count} chunks failed to process. Errors: {error_details}")
            
            if failed_chunks:
                logger.warning(f"Processed {len(valid_results)}/{chunk_count} chunks successfully. {len(failed_chunks)} chunks failed.")
            
            results = valid_results

            logger.info(f"Parallel processing complete. Got {len(results)} valid results out of {chunk_count} chunks")

            # Store all embeddings in Qdrant
            try:
                await qdrant_service.store_source_embeddings(
                    source_id=self.id,
                    embeddings_data=results,
                    notebook_ids=notebook_ids
                )
                logger.info(f"Vectorization complete for source {self.id}: {len(results)} chunks stored")
            except Exception as e:
                logger.error(f"Failed to store embeddings in Qdrant: {e}")
                raise ValueError(f"Failed to store embeddings in Qdrant: {str(e)}")

        except Exception as e:
            error_msg = f"Error vectorizing source {self.id}: {str(e)}"
            logger.error(error_msg)
            logger.exception(e)
            # Wrap the original exception to preserve the error message
            raise DatabaseOperationError(error_msg) from e

    async def add_insight(self, insight_type: str, content: str) -> Any:
        EMBEDDING_MODEL = await model_manager.get_embedding_model()
        if not EMBEDDING_MODEL:
            logger.warning("No embedding model found. Insight will not be searchable.")

        if not insight_type or not content:
            raise InvalidInputError("Insight type and content must be provided")
        try:
            # Get notebook IDs for this source
            notebook_ids = await get_source_notebook_ids(self.id)
            logger.debug(f"Source {self.id} belongs to notebooks: {notebook_ids}")

            # Import Qdrant service
            from open_notebook.services.qdrant_service import qdrant_service

            # Generate embedding
            embedding = None
            if EMBEDDING_MODEL:
                try:
                    embedding = (await EMBEDDING_MODEL.aembed([content]))[0]
                    logger.debug(f"Generated embedding for insight: {len(embedding)} dimensions")
                except Exception as e:
                    logger.error(f"Failed to generate embedding for insight: {e}")
                    embedding = None

            # Store insight in SurrealDB (without embedding)
            insight_result = await repo_query(
                """
                CREATE source_insight CONTENT {
                        "source": $source_id,
                        "insight_type": $insight_type,
                        "content": $content,
                };""",
                {
                    "source_id": ensure_record_id(self.id),
                    "insight_type": insight_type,
                    "content": content,
                },
            )

            # Extract insight ID from result
            insight_id = None
            if insight_result and len(insight_result) > 0:
                insight_id = str(insight_result[0].get("id", "")) if isinstance(insight_result[0], dict) else None

            # Store embedding in Qdrant (only if we have a valid embedding)
            if embedding and len(embedding) > 0:
                try:
                    await qdrant_service.store_source_insight(
                        source_id=self.id,
                        insight_type=insight_type,
                        content=content,
                        embedding=embedding,
                        notebook_ids=notebook_ids,
                        insight_id=insight_id
                    )
                    logger.info(f"Stored insight embedding for source {self.id}" + (f" (insight_id: {insight_id})" if insight_id else ""))
                except Exception as e:
                    logger.error(f"Failed to store insight embedding: {e}")
                    # Don't raise - insight was created in SurrealDB, just embedding failed
            else:
                logger.warning(f"No embedding available for insight - insight stored in SurrealDB only")

            return insight_result
        except Exception as e:
            logger.error(f"Error adding insight to source {self.id}: {str(e)}")
            raise  # DatabaseOperationError(e)

    def _prepare_save_data(self) -> dict:
        """Override to ensure command field is always RecordID format for database"""
        data = super()._prepare_save_data()

        # Ensure command field is RecordID format if not None
        if data.get("command") is not None:
            data["command"] = ensure_record_id(data["command"])

        return data


class ChatSession(ObjectModel):
    table_name: ClassVar[str] = "chat_session"
    title: Optional[str] = None
    model_override: Optional[str] = None

    async def relate_to_notebook(self, notebook_id: str) -> Any:
        if not notebook_id:
            raise InvalidInputError("Notebook ID must be provided")
        return await self.relate("refers_to", notebook_id)

    async def relate_to_source(self, source_id: str) -> Any:
        if not source_id:
            raise InvalidInputError("Source ID must be provided")
        return await self.relate("refers_to", source_id)


async def text_search(
    keyword: str, results: int, source: bool = True
):
    if not keyword:
        raise InvalidInputError("Search keyword cannot be empty")
    try:
        search_results = await repo_query(
            """
            select *
            from fn::text_search($keyword, $results, $source, false)
            """,
            {"keyword": keyword, "results": results, "source": source},
        )
        return search_results
    except Exception as e:
        logger.error(f"Error performing text search: {str(e)}")
        logger.exception(e)
        raise DatabaseOperationError(e)


async def vector_search(
    keyword: str,
    results: int,
    source: bool = True,
    minimum_score=0.2,
    notebook_ids: Optional[List[str]] = None,
):
    """
    Perform vector search using Qdrant
    
    Args:
        keyword: Search keyword
        results: Maximum number of results
        source: Whether to search source embeddings and insights
        minimum_score: Minimum similarity score
        notebook_ids: Optional list of notebook IDs to filter by
        
    Returns:
        List of search results with similarity scores
    """
    if not keyword:
        raise InvalidInputError("Search keyword cannot be empty")
    try:
        import time
        vector_search_start = time.time()
        logger.info(f"vector_search: Starting search for keyword='{keyword[:50]}...', results={results}, source={source}, minimum_score={minimum_score}, notebook_ids={notebook_ids}")
        
        EMBEDDING_MODEL = await model_manager.get_embedding_model()
        if EMBEDDING_MODEL is None:
            logger.error("vector_search: EMBEDDING_MODEL is not configured")
            raise ValueError("EMBEDDING_MODEL is not configured")
        
        # Generate query embedding
        embed_start = time.time()
        try:
            query_vector = (await EMBEDDING_MODEL.aembed([keyword]))[0]
            embed_duration = time.time() - embed_start
            logger.info(f"vector_search: Query embedding generated in {embed_duration:.2f}s, dimension={len(query_vector)}")
        except Exception as embed_error:
            logger.error(f"vector_search: Failed to generate embedding: {type(embed_error).__name__}: {embed_error}")
            logger.exception(embed_error)
            raise
        
        # Import Qdrant service
        from open_notebook.services.qdrant_service import qdrant_service
        
        # Perform search using Qdrant
        qdrant_search_start = time.time()
        try:
            search_results = await qdrant_service.vector_search(
                query_vector=query_vector,
                limit=results,
                notebook_ids=notebook_ids,
                search_sources=source,
                min_score=minimum_score
            )
            qdrant_search_duration = time.time() - qdrant_search_start
            logger.info(f"vector_search: Qdrant search completed in {qdrant_search_duration:.2f}s, found {len(search_results)} results")
        except Exception as qdrant_error:
            logger.error(f"vector_search: Qdrant search failed: {type(qdrant_error).__name__}: {qdrant_error}")
            logger.exception(qdrant_error)
            raise
        
        # 關鍵修復：驗證結果格式並記錄詳細信息
        if not search_results:
            logger.warning(f"vector_search: WARNING - Qdrant returned 0 results for keyword='{keyword[:50]}...'")
            logger.warning(f"vector_search: This may indicate:")
            logger.warning(f"vector_search:   - No matching data in Qdrant collections")
            logger.warning(f"vector_search:   - minimum_score ({minimum_score}) too high")
            logger.warning(f"vector_search:   - Notebook filter too restrictive")
            return []
        
        # Format results to match original API
        formatted_results = []
        missing_fields = []
        empty_content_count = 0
        
        for idx, result in enumerate(search_results):
            try:
                # 關鍵修復：驗證必要字段
                if not result.get("id"):
                    missing_fields.append(f"result[{idx}].id")
                if not result.get("content"):
                    empty_content_count += 1
                    logger.warning(f"vector_search: Result {idx} has empty content, id={result.get('id', 'N/A')}")
                
                formatted_result = {
                    "id": result.get("id", f"unknown_{idx}"),
                    "parent_id": result.get("parent_id", ""),
                    "title": result.get("title", ""),
                    "similarity": result.get("similarity", 0.0),
                    "matches": [result.get("content", "")]  # Wrap content in array to match original format
                }
                formatted_results.append(formatted_result)
            except Exception as format_error:
                logger.warning(f"vector_search: Failed to format result {idx}: {format_error}")
                logger.debug(f"vector_search: Result {idx} keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
        
        # 關鍵修復：記錄格式化統計
        if missing_fields:
            logger.warning(f"vector_search: {len(missing_fields)} results missing required fields: {missing_fields[:5]}")
        if empty_content_count > 0:
            logger.warning(f"vector_search: {empty_content_count} results have empty content")
        
        total_duration = time.time() - vector_search_start
        logger.info(f"vector_search: Completed in {total_duration:.2f}s, returning {len(formatted_results)} formatted results")
        
        if formatted_results:
            # 記錄樣本結果格式（用於調試）
            sample = formatted_results[0]
            logger.debug(f"vector_search: Sample result format: keys={list(sample.keys())}, has_id={bool(sample.get('id'))}, has_matches={bool(sample.get('matches'))}, matches_length={len(sample.get('matches', []))}")
        
        return formatted_results
    except InvalidInputError:
        raise
    except ValueError as e:
        logger.error(f"vector_search: Configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"vector_search: Unexpected error performing vector search: {type(e).__name__}: {e}")
        logger.exception(e)
        raise DatabaseOperationError(e)


async def get_source_notebook_ids(source_id: str) -> List[str]:
    """
    查詢 source 所屬的所有 notebook IDs
    
    Args:
        source_id: Source 的 ID
        
    Returns:
        List[str]: 該 source 所屬的所有 notebook IDs
    """
    try:
        result = await repo_query(
            """
            SELECT VALUE out FROM reference WHERE in = $source_id
            """,
            {"source_id": ensure_record_id(source_id)}
        )
        return [str(notebook_id) for notebook_id in result] if result else []
    except Exception as e:
        logger.error(f"Error fetching notebook IDs for source {source_id}: {str(e)}")
        logger.exception(e)
        raise DatabaseOperationError(e)