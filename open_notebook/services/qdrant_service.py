"""
Qdrant service for vector storage and search operations
"""

import asyncio
import os
import uuid
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import Distance, FieldCondition, Filter, MatchAny, MatchValue, PointStruct, VectorParams
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from open_notebook.config import QdrantConfig
from open_notebook.domain.models import model_manager


def async_wrap(func):
    """Wrap synchronous Qdrant client methods to run in thread pool."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)
    return wrapper


# Retry decorator for critical Qdrant operations
qdrant_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
    reraise=True,
)


class QdrantService:
    """Service for managing vector embeddings in Qdrant"""
    
    def __init__(self):
        self.client = None
        self.vector_dim = None
        self._collections_created = False
        
    async def _ensure_client(self) -> QdrantClient:
        """Ensure Qdrant client is initialized with configuration."""
        if self.client is None:
            logger.info(
                f"QdrantService: Initializing connection to {QdrantConfig.host}:{QdrantConfig.port} "
                f"(API key: {'provided' if QdrantConfig.api_key else 'not provided'}, "
                f"gRPC: {QdrantConfig.prefer_grpc}, timeout: {QdrantConfig.timeout}s)"
            )
            
            try:
                # Initialize client with configuration
                self.client = QdrantClient(
                    host=QdrantConfig.host,
                    port=QdrantConfig.port,
                    api_key=QdrantConfig.api_key,
                    timeout=QdrantConfig.timeout,
                    prefer_grpc=QdrantConfig.prefer_grpc,
                )
                # Test connection in thread pool to avoid blocking
                try:
                    collections = await asyncio.to_thread(self.client.get_collections)
                    logger.info(
                        f"QdrantService: Successfully connected to {QdrantConfig.host}:{QdrantConfig.port}, "
                        f"found {len(collections.collections)} collections"
                    )
                except Exception as test_error:
                    logger.warning(f"QdrantService: Connected but failed to test connection: {test_error}")
                logger.info(f"Connected to Qdrant at {QdrantConfig.host}:{QdrantConfig.port}")
            except Exception as e:
                logger.error(
                    f"QdrantService: Failed to connect to Qdrant at {QdrantConfig.host}:{QdrantConfig.port}: "
                    f"{type(e).__name__}: {e}"
                )
                logger.exception(e)
                raise
                
        return self.client
    
    async def delete_collection(self, collection_name: str) -> None:
        """
        Delete a Qdrant collection
        
        Args:
            collection_name: Name of the collection to delete
        """
        client = await self._ensure_client()
        
        @qdrant_retry
        def _delete_collection():
            # Check if collection exists
            collections_info = client.get_collections()
            existing_collections = [c.name for c in collections_info.collections]
            
            if collection_name in existing_collections:
                client.delete_collection(collection_name)
                logger.info(f"Deleted collection: {collection_name}")
            else:
                logger.debug(f"Collection {collection_name} does not exist, skipping deletion")
        
        try:
            await asyncio.to_thread(_delete_collection)
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise
    
    async def check_and_recreate_collections_if_needed(self) -> bool:
        """
        Check if collections need to be recreated due to dimension mismatch.
        If mismatch is detected, delete and recreate collections.
        
        Returns:
            True if collections were recreated, False otherwise
        """
        client = await self._ensure_client()
        
        # Get current embedding model dimension
        if self.vector_dim is None:
            embedding_model = await model_manager.get_embedding_model()
            if embedding_model is None:
                raise ValueError("No embedding model configured")
            test_embedding = await embedding_model.aembed(["test"])
            self.vector_dim = len(test_embedding[0])
            logger.info(f"Current embedding model dimension: {self.vector_dim}")
        
        collections_to_check = ["source_embeddings", "source_insights"]
        needs_recreation = False
        
        # Check each collection for dimension mismatch
        for collection_name in collections_to_check:
            try:
                def _get_collections():
                    return client.get_collections()
                
                collections_info = await asyncio.to_thread(_get_collections)
                existing_collections = [c.name for c in collections_info.collections]
                
                if collection_name not in existing_collections:
                    # Collection doesn't exist, will be created by _ensure_collections
                    continue
                
                # Collection exists, check dimension
                def _get_collection():
                    return client.get_collection(collection_name)
                
                collection_info = await asyncio.to_thread(_get_collection)
                existing_dim = None
                
                if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'params'):
                    params = collection_info.config.params
                    if hasattr(params, 'vectors'):
                        vectors_config = params.vectors
                        if hasattr(vectors_config, 'size'):
                            existing_dim = vectors_config.size
                        elif isinstance(vectors_config, dict) and 'size' in vectors_config:
                            existing_dim = vectors_config['size']
                
                if existing_dim is not None and existing_dim != self.vector_dim:
                    logger.warning(
                        f"Collection {collection_name} has dimension mismatch: "
                        f"expected {self.vector_dim}, got {existing_dim}. "
                        f"Will delete and recreate collection."
                    )
                    needs_recreation = True
                    break
            except Exception as e:
                logger.warning(f"Could not check collection {collection_name}: {e}")
        
        # If dimension mismatch detected, delete all collections
        if needs_recreation:
            logger.info("Dimension mismatch detected. Deleting and recreating collections...")
            for collection_name in collections_to_check:
                try:
                    await self.delete_collection(collection_name)
                except Exception as e:
                    logger.error(f"Failed to delete collection {collection_name}: {e}")
                    # Continue with other collections even if one fails
            
            # Reset collections_created flag to force recreation
            self._collections_created = False
            logger.info("Collections deleted. Will be recreated with correct dimension.")
            return True
        
        return False
    
    async def _ensure_collections(self) -> None:
        """Ensure all required collections exist"""
        if self._collections_created:
            return
            
        client = await self._ensure_client()
        
        # Get vector dimension from embedding model
        if self.vector_dim is None:
            embedding_model = await model_manager.get_embedding_model()
            if embedding_model is None:
                raise ValueError("No embedding model configured")
            # Get dimension by creating a test embedding
            test_embedding = await embedding_model.aembed(["test"])
            self.vector_dim = len(test_embedding[0])
            logger.info(f"Vector dimension set to {self.vector_dim}")
        
        collections = [
            ("source_embeddings", "Source document chunks"),
            ("source_insights", "Source insights"),
        ]
        
        for collection_name, description in collections:
            try:
                # Check if collection exists
                def _get_collections():
                    return client.get_collections()
                
                collections_info = await asyncio.to_thread(_get_collections)
                existing_collections = [c.name for c in collections_info.collections]
                
                if collection_name not in existing_collections:
                    # Collection doesn't exist, create it
                    def _create_collection():
                        client.create_collection(
                            collection_name=collection_name,
                            vectors_config=VectorParams(
                                size=self.vector_dim,
                                distance=Distance.COSINE
                            )
                        )
                    
                    await asyncio.to_thread(_create_collection)
                    logger.info(f"Created collection: {collection_name} with dimension {self.vector_dim}")
                else:
                    # Collection exists, check if dimension matches
                    try:
                        def _get_collection():
                            return client.get_collection(collection_name)
                        
                        collection_info = await asyncio.to_thread(_get_collection)
                        # Get vector dimension from collection config
                        # Qdrant collection config structure: config.params.vectors.size
                        existing_dim = None
                        if hasattr(collection_info, 'config') and hasattr(collection_info.config, 'params'):
                            params = collection_info.config.params
                            if hasattr(params, 'vectors'):
                                vectors_config = params.vectors
                                # Handle both named vectors and single vector config
                                if hasattr(vectors_config, 'size'):
                                    # Single vector config
                                    existing_dim = vectors_config.size
                                elif hasattr(vectors_config, 'named'):
                                    # Named vectors - this is more complex, skip for now
                                    logger.warning(f"Collection {collection_name} uses named vectors, dimension check skipped")
                                else:
                                    # Try to access as dict-like
                                    if isinstance(vectors_config, dict) and 'size' in vectors_config:
                                        existing_dim = vectors_config['size']
                        
                        if existing_dim is not None and existing_dim != self.vector_dim:
                            logger.warning(
                                f"Collection {collection_name} has dimension mismatch: "
                                f"current embedding model produces {self.vector_dim}-dimensional vectors, "
                                f"but collection expects {existing_dim}-dimensional vectors. "
                                f"This usually happens when switching embedding models."
                            )
                            raise ValueError(
                                f"Vector dimension mismatch for collection '{collection_name}': "
                                f"current embedding model produces {self.vector_dim}-dimensional vectors, "
                                f"but collection expects {existing_dim}-dimensional vectors. "
                                f"This happens when switching embedding models. "
                                f"Solution: Delete the collection or use 'Rebuild Embeddings' to migrate data."
                            )
                        elif existing_dim is not None:
                            logger.debug(f"Collection {collection_name} already exists with dimension {existing_dim}")
                        else:
                            logger.warning(f"Could not determine dimension for collection {collection_name}, skipping dimension check")
                    except ValueError:
                        # Re-raise dimension mismatch errors
                        raise
                    except Exception as e:
                        logger.warning(f"Could not check dimension for collection {collection_name}: {e}, skipping dimension check")
                    
            except ValueError:
                # Re-raise dimension mismatch errors
                raise
            except Exception as e:
                logger.error(f"Failed to create/check collection {collection_name}: {e}")
                raise
        
        self._collections_created = True
        logger.info("All Qdrant collections are ready")
    
    async def store_source_embeddings(
        self, 
        source_id: str, 
        embeddings_data: List[Tuple[int, List[float], str]], 
        notebook_ids: List[str]
    ) -> None:
        """
        Store source embeddings in Qdrant.
        
        New logic: Content is stored in SurrealDB source_chunk table,
        Qdrant only stores chunk_id and vector.
        
        Args:
            source_id: Source ID
            embeddings_data: List of (order, embedding, content) tuples
            notebook_ids: List of notebook IDs this source belongs to
        """
        from open_notebook.domain.notebook import SourceChunk
        
        # Step 1: Delete existing chunks for idempotency
        try:
            await SourceChunk.delete_by_source_id(source_id)
            logger.debug(f"Deleted existing chunks for source {source_id} (idempotency)")
        except Exception as e:
            logger.warning(f"Failed to delete existing chunks (may not exist): {e}")
            # Continue anyway - this is not critical
        
        await self._ensure_collections()
        client = await self._ensure_client()
        
        chunk_ids = []
        try:
            # Step 2: Store chunks in SurrealDB and get chunk IDs
            chunks_data = [(order, content) for order, embedding, content in embeddings_data]
            chunk_ids = await SourceChunk.create_batch(source_id, chunks_data)
            
            if len(chunk_ids) != len(embeddings_data):
                raise ValueError(
                    f"Mismatch: created {len(chunk_ids)} chunks but expected {len(embeddings_data)}"
                )
            
            logger.info(f"Stored {len(chunk_ids)} chunks in SurrealDB for source {source_id}")
            
            # Step 3: Store embeddings in Qdrant with chunk_id (not content)
            points = []
            for idx, (order, embedding, content) in enumerate(embeddings_data):
                point_id = str(uuid.uuid4())
                chunk_id = chunk_ids[idx] if idx < len(chunk_ids) else None
                
                if not chunk_id:
                    raise ValueError(f"Missing chunk_id for chunk {idx}")
                
                payload = {
                    "source_id": source_id,
                    "chunk_id": chunk_id,
                    "order": order,
                    "notebook_ids": notebook_ids
                }
                # Content is no longer stored in Qdrant payload
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
                points.append(point)
            
            @qdrant_retry
            def _upsert_points():
                client.upsert(
                    collection_name="source_embeddings",
                    points=points
                )
            
            await asyncio.to_thread(_upsert_points)
            logger.info(f"Stored {len(points)} embeddings in Qdrant for source {source_id}")
            
        except Exception as e:
            error_str = str(e)
            
            # Rollback: If Qdrant write failed, delete SurrealDB chunks
            if chunk_ids:
                logger.warning(
                    f"Qdrant write failed for source {source_id}, "
                    f"rolling back {len(chunk_ids)} chunks from SurrealDB"
                )
                try:
                    await SourceChunk.delete_by_source_id(source_id)
                    logger.info(f"Rolled back {len(chunk_ids)} chunks for source {source_id}")
                except Exception as rollback_error:
                    logger.error(f"Failed to rollback chunks: {rollback_error}")
                    # Continue to raise original error
            
            # Check for dimension mismatch error
            if "dimension" in error_str.lower() or "dim" in error_str.lower():
                logger.error(f"Vector dimension mismatch when storing embeddings: {e}")
                # Extract dimension info from error message if available
                if "expected dim" in error_str and "got" in error_str:
                    raise ValueError(
                        f"Vector dimension mismatch: {error_str}. "
                        f"This usually happens when switching embedding models. "
                        f"The Qdrant collection was created with a different embedding model. "
                        f"Please delete the collection or use 'Rebuild Embeddings' to migrate data."
                    )
                else:
                    raise ValueError(
                        f"Vector dimension mismatch: {error_str}. "
                        f"Please check that the embedding model matches the Qdrant collection configuration."
                    )
            else:
                logger.error(f"Failed to store source embeddings: {e}")
                raise
    
    async def store_source_insight(
        self, 
        source_id: str, 
        insight_type: str, 
        content: str, 
        embedding: List[float], 
        notebook_ids: List[str],
        insight_id: Optional[str] = None
    ) -> None:
        """
        Store source insight in Qdrant
        
        Args:
            source_id: Source ID
            insight_type: Type of insight
            content: Insight content
            embedding: Vector embedding
            notebook_ids: List of notebook IDs this source belongs to
            insight_id: Insight ID from SurrealDB (optional, for deletion purposes)
        """
        await self._ensure_collections()
        client = await self._ensure_client()
        
        point_id = str(uuid.uuid4())
        payload = {
            "source_id": source_id,
            "insight_type": insight_type,
            "notebook_ids": notebook_ids
        }
        
        # Only store content if configured to do so
        if QdrantConfig.store_content_in_payload:
            payload["content"] = content
        
        # Add insight_id to payload if provided (always needed for cleanup)
        if insight_id:
            payload["insight_id"] = insight_id
        
        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=payload
        )
        
        @qdrant_retry
        def _upsert_insight():
            client.upsert(
                collection_name="source_insights",
                points=[point]
            )
        
        try:
            await asyncio.to_thread(_upsert_insight)
            logger.info(f"Stored insight for source {source_id}" + (f" (insight_id: {insight_id})" if insight_id else ""))
        except Exception as e:
            error_str = str(e)
            # Check for dimension mismatch error
            if "dimension" in error_str.lower() or "dim" in error_str.lower():
                logger.error(f"Vector dimension mismatch when storing insight: {e}")
                # Extract dimension info from error message if available
                if "expected dim" in error_str and "got" in error_str:
                    raise ValueError(
                        f"Vector dimension mismatch: {error_str}. "
                        f"This usually happens when switching embedding models. "
                        f"The Qdrant collection was created with a different embedding model. "
                        f"Please delete the collection or use 'Rebuild Embeddings' to migrate data."
                    )
                else:
                    raise ValueError(
                        f"Vector dimension mismatch: {error_str}. "
                        f"Please check that the embedding model matches the Qdrant collection configuration."
                    )
            else:
                logger.error(f"Failed to store source insight: {e}")
                raise
    
    async def delete_source_embeddings(self, source_id: str) -> None:
        """
        Delete all embeddings for a source
        
        Args:
            source_id: Source ID
        """
        await self._ensure_collections()
        client = await self._ensure_client()
        
        @qdrant_retry
        def _delete_embeddings():
            # Delete from source_embeddings collection
            client.delete(
                collection_name="source_embeddings",
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="source_id",
                                match=models.MatchValue(value=source_id)
                            )
                        ]
                    )
                )
            )
            
            # Delete from source_insights collection
            client.delete(
                collection_name="source_insights",
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="source_id",
                                match=models.MatchValue(value=source_id)
                            )
                        ]
                    )
                )
            )
        
        try:
            await asyncio.to_thread(_delete_embeddings)
            logger.info(f"Deleted all embeddings for source {source_id}")
        except Exception as e:
            logger.error(f"Failed to delete source embeddings: {e}")
            raise
    
    async def delete_source_insight(self, insight_id: str) -> None:
        """
        Delete a specific insight from Qdrant by insight_id
        
        Args:
            insight_id: Insight ID from SurrealDB
        """
        await self._ensure_collections()
        client = await self._ensure_client()
        
        @qdrant_retry
        def _delete_insight():
            # Delete from source_insights collection based on insight_id in payload
            client.delete(
                collection_name="source_insights",
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[
                            FieldCondition(
                                key="insight_id",
                                match=models.MatchValue(value=insight_id)
                            )
                        ]
                    )
                )
            )
        
        try:
            await asyncio.to_thread(_delete_insight)
            logger.info(f"Deleted insight {insight_id} from Qdrant")
        except Exception as e:
            logger.error(f"Failed to delete insight {insight_id} from Qdrant: {e}")
            # Don't raise - allow deletion to continue even if Qdrant delete fails
            # This ensures backward compatibility with insights that don't have insight_id stored
    
    async def vector_search(
        self,
        query_vector: List[float],
        limit: int = 10,
        notebook_ids: Optional[List[str]] = None,
        search_sources: bool = True,
        min_score: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Perform vector search across collections
        
        Args:
            query_vector: Query vector
            limit: Maximum number of results
            notebook_ids: Optional list of notebook IDs to filter by
            search_sources: Whether to search source embeddings and insights
            min_score: Minimum similarity score
            
        Returns:
            List of search results with similarity scores
        """
        import time
        search_start_time = time.time()
        
        # 關鍵修復：添加詳細的搜索參數日誌
        logger.info(f"QdrantService.vector_search: Starting search")
        logger.info(f"QdrantService.vector_search: Parameters - limit={limit}, search_sources={search_sources}, min_score={min_score}")
        logger.info(f"QdrantService.vector_search: Query vector dimension: {len(query_vector) if query_vector else 'None'}")
        logger.info(f"QdrantService.vector_search: Notebook IDs filter: {notebook_ids if notebook_ids else 'None (no filter)'}")
        
        await self._ensure_collections()
        client = await self._ensure_client()
        
        # 關鍵修復：驗證客戶端連接
        if client is None:
            logger.error("QdrantService.vector_search: Client is None after _ensure_client()")
            raise RuntimeError("Qdrant client is not initialized")
        
        all_results = []
        
        # Build filter for notebook_ids if provided
        query_filter = None
        if notebook_ids:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="notebook_ids",
                        match=MatchAny(any=notebook_ids)
                    )
                ]
            )
            logger.info(f"QdrantService.vector_search: Applied notebook filter with {len(notebook_ids)} notebook IDs")
        else:
            logger.info(f"QdrantService.vector_search: No notebook filter applied")
        
        # Search source embeddings
        if search_sources:
            try:
                embed_search_start = time.time()
                
                def _search_embeddings():
                    return client.search(
                        collection_name="source_embeddings",
                        query_vector=query_vector,
                        query_filter=query_filter,
                        limit=limit,
                        score_threshold=min_score
                    )
                
                source_results = await asyncio.to_thread(_search_embeddings)
                embed_search_duration = time.time() - embed_search_start
                logger.info(f"QdrantService.vector_search: Source embeddings search completed in {embed_search_duration:.2f}s, found {len(source_results)} results")
                
                # 關鍵修復：記錄結果的相似度範圍
                if source_results:
                    scores = [r.score for r in source_results]
                    logger.info(f"QdrantService.vector_search: Source embeddings similarity scores - min={min(scores):.4f}, max={max(scores):.4f}, avg={sum(scores)/len(scores):.4f}")
                
                # Collect chunk_ids for batch query
                chunk_ids = []
                result_mappings = []  # Store mapping: (result_index, chunk_id or None)
                
                for idx, result in enumerate(source_results):
                    chunk_id = result.payload.get("chunk_id")
                    if chunk_id:
                        chunk_ids.append(chunk_id)
                        result_mappings.append((idx, chunk_id))
                    else:
                        # Backward compatibility: old format with content in payload
                        result_mappings.append((idx, None))
                
                # Batch fetch content from SurrealDB
                chunk_contents = {}
                if chunk_ids:
                    try:
                        from open_notebook.domain.notebook import SourceChunk
                        chunk_contents = await SourceChunk.get_by_ids(chunk_ids)
                        logger.debug(
                            f"QdrantService.vector_search: Retrieved {len(chunk_contents)}/{len(chunk_ids)} chunks from SurrealDB"
                        )
                    except Exception as fetch_error:
                        logger.warning(
                            f"QdrantService.vector_search: Failed to fetch chunks from SurrealDB: {fetch_error}"
                        )
                        # Continue with empty content
                
                # Build results with content from SurrealDB
                for idx, result in enumerate(source_results):
                    try:
                        chunk_id = result.payload.get("chunk_id")
                        content = ""
                        
                        if chunk_id and chunk_id in chunk_contents:
                            # New format: get content from SurrealDB
                            content = chunk_contents[chunk_id].content
                        elif result.payload.get("content"):
                            # Backward compatibility: old format with content in payload
                            content = result.payload.get("content", "")
                            logger.debug(
                                f"QdrantService.vector_search: Using legacy content from payload for result {idx}"
                            )
                        
                        result_dict = {
                            "id": result.payload.get("source_id", ""),
                            "title": result.payload.get("source_id", ""),  # Will be updated with actual title
                            "content": content,
                            "parent_id": result.payload.get("source_id", ""),
                            "similarity": result.score,
                            "type": "source_embedding"
                        }
                        all_results.append(result_dict)
                    except Exception as result_error:
                        logger.warning(f"QdrantService.vector_search: Failed to process source embedding result {idx}: {result_error}")
                        logger.debug(f"QdrantService.vector_search: Result payload keys: {list(result.payload.keys()) if hasattr(result, 'payload') else 'N/A'}")
                
                logger.info(f"QdrantService.vector_search: Successfully processed {len(all_results)} source embedding results")
            except Exception as e:
                logger.error(f"QdrantService.vector_search: Error searching source embeddings: {type(e).__name__}: {e}")
                logger.exception(e)
        
        # Search source insights (Optimistic Read Strategy: no validation during search)
        if search_sources:
            try:
                insight_search_start = time.time()
                
                def _search_insights():
                    return client.search(
                        collection_name="source_insights",
                        query_vector=query_vector,
                        query_filter=query_filter,
                        limit=limit,
                        score_threshold=min_score
                    )
                
                insight_results = await asyncio.to_thread(_search_insights)
                insight_search_duration = time.time() - insight_search_start
                logger.info(
                    f"QdrantService.vector_search: Source insights search completed in {insight_search_duration:.2f}s, "
                    f"found {len(insight_results)} results"
                )
                
                # Optimistic Read: Return results directly without validation
                # Validation will happen when user actually accesses the content
                # Orphaned insights will be cleaned up by background task
                for result in insight_results:
                    try:
                        insight_id = result.payload.get("insight_id")
                        result_id = insight_id if insight_id else str(uuid.uuid4())
                        result_dict = {
                            "id": result_id,
                            "title": f"{result.payload.get('insight_type', 'unknown')} - {result.payload.get('source_id', '')}",
                            "content": result.payload.get("content", "") if QdrantConfig.store_content_in_payload else "",
                            "parent_id": result.payload.get("source_id", ""),
                            "similarity": result.score,
                            "type": "source_insight"
                        }
                        all_results.append(result_dict)
                    except Exception as result_error:
                        logger.warning(f"QdrantService.vector_search: Failed to process insight result: {result_error}")
                
                # Log similarity scores
                if insight_results:
                    scores = [r.score for r in insight_results]
                    logger.info(
                        f"QdrantService.vector_search: Insight similarity scores - "
                        f"min={min(scores):.4f}, max={max(scores):.4f}, avg={sum(scores)/len(scores):.4f}"
                    )
            except Exception as e:
                logger.error(f"QdrantService.vector_search: Error searching source insights: {type(e).__name__}: {e}")
                logger.exception(e)
        
        # Sort by similarity and limit results
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        final_results = all_results[:limit]
        total_duration = time.time() - search_start_time
        
        # 關鍵修復：記錄最終結果的詳細統計
        logger.info(f"QdrantService.vector_search: Completed in {total_duration:.2f}s")
        logger.info(f"QdrantService.vector_search: Final results - {len(final_results)}/{len(all_results)} (limited from {len(all_results)} total)")
        
        if final_results:
            final_scores = [r["similarity"] for r in final_results]
            logger.info(f"QdrantService.vector_search: Final similarity scores - min={min(final_scores):.4f}, max={max(final_scores):.4f}, avg={sum(final_scores)/len(final_scores):.4f}")
            logger.info(f"QdrantService.vector_search: Result types - {len([r for r in final_results if r.get('type') == 'source_embedding'])} embeddings, {len([r for r in final_results if r.get('type') == 'source_insight'])} insights")
        else:
            logger.warning(f"QdrantService.vector_search: WARNING - No results returned! This may indicate:")
            logger.warning(f"QdrantService.vector_search:   - Query vector dimension mismatch")
            logger.warning(f"QdrantService.vector_search:   - min_score threshold too high (current: {min_score})")
            logger.warning(f"QdrantService.vector_search:   - No matching data in collections")
            logger.warning(f"QdrantService.vector_search:   - Notebook filter too restrictive")
        
        return final_results
    
    async def cleanup_orphaned_insights(
        self,
        batch_size: int = 100,
        max_cleanup: Optional[int] = None,
        rate_limit_delay: float = 0.1
    ) -> Dict[str, int]:
        """
        Periodic cleanup task: Remove orphaned insights from Qdrant (exist in Qdrant but not in SurrealDB).
        
        This is a background task that should be run infrequently to avoid impacting search performance.
        Consider using event-driven deletion instead when SurrealDB records are deleted.
        
        Args:
            batch_size: Number of insights to process per batch
            max_cleanup: Maximum number of insights to check (None = no limit)
            rate_limit_delay: Delay between batches in seconds (rate limiting)
            
        Returns:
            Dict with cleanup statistics: {"checked": int, "orphaned": int, "cleaned": int, "failed": int}
        """
        import time
        cleanup_start = time.time()
        logger.info("QdrantService.cleanup_orphaned_insights: Starting periodic cleanup of orphaned insights...")
        
        stats = {
            "checked": 0,
            "orphaned": 0,
            "cleaned": 0,
            "failed": 0
        }
        
        try:
            await self._ensure_collections()
            client = await self._ensure_client()
            
            # Get all insights from Qdrant with rate limiting
            all_insight_ids = []
            offset = None
            
            logger.info("QdrantService.cleanup_orphaned_insights: Scanning Qdrant for insights...")
            while True:
                try:
                    def _scroll_insights():
                        return client.scroll(
                            collection_name="source_insights",
                            limit=batch_size,
                            offset=offset,
                            with_payload=True,
                            with_vectors=False
                        )
                    
                    scroll_result = await asyncio.to_thread(_scroll_insights)
                    
                    points = scroll_result[0] if scroll_result[0] else []
                    if not points:
                        break
                    
                    # Extract insight_ids
                    for point in points:
                        insight_id = point.payload.get("insight_id")
                        if insight_id:
                            all_insight_ids.append(insight_id)
                    
                    stats["checked"] += len(points)
                    logger.info(f"QdrantService.cleanup_orphaned_insights: Scanned {stats['checked']} insights...")
                    
                    # Rate limiting: delay between batches
                    if rate_limit_delay > 0:
                        await asyncio.sleep(rate_limit_delay)
                    
                    # Update offset
                    if len(points) < batch_size:
                        break
                    offset = scroll_result[1]  # Next offset
                    
                    # Stop if max cleanup limit reached
                    if max_cleanup and stats["checked"] >= max_cleanup:
                        logger.info(f"QdrantService.cleanup_orphaned_insights: Reached max_cleanup limit ({max_cleanup})")
                        break
                        
                except Exception as scroll_error:
                    logger.error(f"QdrantService.cleanup_orphaned_insights: Error scrolling insights: {scroll_error}")
                    break
            
            logger.info(f"QdrantService.cleanup_orphaned_insights: Found {len(all_insight_ids)} insights in Qdrant to verify")
            
            if not all_insight_ids:
                logger.info("QdrantService.cleanup_orphaned_insights: No insights found in Qdrant")
                return stats
            
            # 批量驗證 insights 是否存在於 SurrealDB
            from open_notebook.database.repository import repo_query, ensure_record_id
            
            orphaned_ids = []
            
            # 分批驗證
            for batch_start in range(0, len(all_insight_ids), batch_size):
                batch_ids = all_insight_ids[batch_start:batch_start + batch_size]
                
                try:
                    # 構建批量查詢
                    conditions = []
                    params = {}
                    for idx, insight_id in enumerate(batch_ids):
                        param_key = f"id_{idx}"
                        conditions.append(f"id = ${param_key}")
                        params[param_key] = ensure_record_id(insight_id)
                    
                    query = f"SELECT id FROM source_insight WHERE {' OR '.join(conditions)}"
                    batch_result = await repo_query(query, params)
                    
                    # 找出有效的 IDs
                    valid_ids = {str(row.get("id")) for row in batch_result if row.get("id")}
                    
                    # 找出孤立的 IDs
                    for insight_id in batch_ids:
                        if insight_id not in valid_ids:
                            orphaned_ids.append(insight_id)
                    
                    logger.info(
                        f"QdrantService.cleanup_orphaned_insights: Batch {batch_start // batch_size + 1}: "
                        f"{len(valid_ids)} valid, {len(batch_ids) - len(valid_ids)} orphaned"
                    )
                    
                except Exception as batch_error:
                    logger.warning(f"QdrantService.cleanup_orphaned_insights: Batch verification failed: {batch_error}")
                    # 如果批量驗證失敗，跳過這個批次（保守策略）
            
            stats["orphaned"] = len(orphaned_ids)
            logger.info(f"QdrantService.cleanup_orphaned_insights: Found {stats['orphaned']} orphaned insights")
            
            # 批量刪除孤立的 insights
            if orphaned_ids:
                logger.info(f"QdrantService.cleanup_orphaned_insights: Cleaning up {len(orphaned_ids)} orphaned insights...")
                
                for insight_id in orphaned_ids:
                    try:
                        await self.delete_source_insight(insight_id)
                        stats["cleaned"] += 1
                    except Exception as cleanup_error:
                        logger.warning(f"QdrantService.cleanup_orphaned_insights: Failed to delete orphaned insight {insight_id}: {cleanup_error}")
                        stats["failed"] += 1
                
                logger.info(
                    f"QdrantService.cleanup_orphaned_insights: Cleanup completed: "
                    f"{stats['cleaned']} cleaned, {stats['failed']} failed"
                )
            
            cleanup_duration = time.time() - cleanup_start
            logger.info(
                f"QdrantService.cleanup_orphaned_insights: Periodic cleanup completed in {cleanup_duration:.2f}s: "
                f"checked={stats['checked']}, orphaned={stats['orphaned']}, "
                f"cleaned={stats['cleaned']}, failed={stats['failed']}"
            )
            
        except Exception as e:
            logger.error(f"QdrantService.cleanup_orphaned_insights: Error during cleanup: {e}")
            logger.exception(e)
        
        return stats
    
    async def update_notebook_associations(
        self, 
        item_id: str, 
        item_type: str, 
        notebook_ids: List[str]
    ) -> None:
        """
        Update notebook associations for an item
        
        Args:
            item_id: Item ID (source_id)
            item_type: Type of item ("source")
            notebook_ids: New list of notebook IDs
        """
        await self._ensure_collections()
        client = await self._ensure_client()
        
        collections = []
        if item_type == "source":
            collections = ["source_embeddings", "source_insights"]
            key_field = "source_id"
        else:
            raise ValueError(f"Invalid item_type: {item_type}. Only 'source' is supported.")
        
        for collection_name in collections:
            try:
                @qdrant_retry
                def _update_associations():
                    # Find all points for this item
                    scroll_result = client.scroll(
                        collection_name=collection_name,
                        scroll_filter=Filter(
                            must=[
                                FieldCondition(
                                    key=key_field,
                                    match=MatchValue(value=item_id)
                                )
                            ]
                        ),
                        limit=1000  # Adjust based on expected size
                    )
                    
                    if scroll_result[0]:  # If points found
                        point_ids = [point.id for point in scroll_result[0]]
                        client.set_payload(
                            collection_name=collection_name,
                            payload={"notebook_ids": notebook_ids},
                            points=point_ids
                        )
                
                await asyncio.to_thread(_update_associations)
                logger.info(f"Updated notebook associations for {item_type} {item_id} in {collection_name}")
                    
            except Exception as e:
                logger.error(f"Failed to update notebook associations in {collection_name}: {e}")
                raise


# Global instance
qdrant_service = QdrantService()
