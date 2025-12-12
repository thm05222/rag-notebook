"""
Generic ContextBuilder for the Open Notebook project.

This module provides a flexible ContextBuilder class that can handle any parameters
and build context from sources, notebooks, and insights.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from loguru import logger

from open_notebook.domain.notebook import Notebook, Source
from open_notebook.exceptions import DatabaseOperationError, NotFoundError

from .text_utils import token_count


@dataclass
class ContextItem:
    """Represents a single item in the context."""
    
    id: str
    type: Literal["source", "insight", "search_result", "tool_output"]
    content: Dict[str, Any]
    priority: int = 0
    token_count: Optional[int] = None
    
    def __post_init__(self):
        """Calculate token count smartly based on content type."""
        if self.token_count is None:
            text_to_count = ""
            
            if self.type == "search_result":
                # For search results, only count meaningful text fields
                if "content" in self.content:
                    text_to_count += str(self.content.get("content", ""))
                if "title" in self.content:
                    text_to_count += f" {self.content.get('title', '')}"
                # For PageIndex results, extract summary, title, and text
                if "summary" in self.content:
                    text_to_count += f" {self.content.get('summary', '')}"
                if "text" in self.content:
                    text_to_count += f" {self.content.get('text', '')}"
                # If no meaningful fields found, fallback to minimal representation
                if not text_to_count.strip():
                    text_to_count = json.dumps(self.content, ensure_ascii=False)
            else:
                # For other types, use JSON representation but it's already structured
                text_to_count = json.dumps(self.content, ensure_ascii=False)
            
            self.token_count = token_count(text_to_count)


@dataclass
class ContextConfig:
    """Configuration for context building."""

    sources: Optional[Dict[str, str]] = None  # {source_id: inclusion_level}
    include_insights: bool = True
    max_tokens: Optional[int] = None
    priority_weights: Optional[Dict[str, int]] = None  # {type: weight}
    
    def __post_init__(self):
        """Initialize default values."""
        if self.sources is None:
            self.sources = {}
        if self.priority_weights is None:
            self.priority_weights = {"source": 100, "insight": 75}


class ContextBuilder:
    """
    Generic ContextBuilder that can handle any parameters and build context
    from sources, notebooks, and insights.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize ContextBuilder with flexible parameters.

        Supported parameters:
        - source_id: str - Include specific source
        - notebook_id: str - Include notebook content
        - include_insights: bool - Include source insights
        - context_config: ContextConfig - Custom context configuration
        - max_tokens: int - Maximum token limit
        - priority_order: List[str] - Custom priority order
        """
        # Store all parameters for flexibility
        self.params = kwargs

        # Extract commonly used parameters
        self.source_id: Optional[str] = kwargs.get('source_id')
        self.notebook_id: Optional[str] = kwargs.get('notebook_id')
        self.include_insights: bool = kwargs.get('include_insights', True)
        self.max_tokens: Optional[int] = kwargs.get('max_tokens')

        # Context configuration
        context_config_arg: Optional[ContextConfig] = kwargs.get('context_config')
        self.context_config: ContextConfig
        if context_config_arg is None:
            self.context_config = ContextConfig(
                include_insights=self.include_insights,
                max_tokens=self.max_tokens
            )
        else:
            self.context_config = context_config_arg

        # Items storage
        self.items: List[ContextItem] = []
        
        # Concurrency control: Semaphore for limiting concurrent source processing
        # Default to 15, can be overridden via environment variable
        max_concurrent = int(os.getenv("CONTEXT_BUILDER_MAX_CONCURRENT", "15"))
        self._semaphore = asyncio.Semaphore(max_concurrent)

        logger.debug(f"ContextBuilder initialized with params: {list(kwargs.keys())}, max_concurrent={max_concurrent}")
    
    async def build(self) -> Dict[str, Any]:
        """
        Build context based on provided parameters.
        
        Returns:
            Dict containing the built context with metadata
        """
        try:
            logger.info("Starting context building")
            
            # Clear existing items
            self.items = []
            
            # Build context based on parameters
            if self.source_id:
                await self._add_source_context(self.source_id)
            
            if self.notebook_id:
                await self._add_notebook_context(self.notebook_id)
            
            # Process any additional custom parameters
            await self._process_custom_params()
            
            # Apply post-processing
            self.remove_duplicates()
            self.prioritize()
            
            if self.max_tokens:
                self.truncate_to_fit(self.max_tokens)
            
            # Format and return response
            return self._format_response()
            
        except Exception as e:
            logger.error(f"Error building context: {str(e)}")
            raise DatabaseOperationError(f"Failed to build context: {str(e)}")
    
    async def _add_source_context(
        self, 
        source_id: str, 
        inclusion_level: str = "insights"
    ) -> None:
        """
        Add source and its insights to context.
        
        Allows partial failures - logs warnings but doesn't raise exceptions
        to allow processing of other sources to continue.
        
        Args:
            source_id: ID of the source
            inclusion_level: "insights", "full content", or "not in"
        """
        if inclusion_level == "not in":
            return
        
        try:
            # Ensure source ID has table prefix
            full_source_id = (
                source_id if source_id.startswith("source:")
                else f"source:{source_id}"
            )
            
            source = await Source.get(full_source_id)
            if not source:
                logger.warning(f"Source {source_id} not found")
                return
            
            # Determine context size based on inclusion level
            context_size: Literal["short", "long"] = "long" if "full content" in inclusion_level else "short"
            source_context = await source.get_context(context_size=context_size)

            # Add source item
            priority = (self.context_config.priority_weights or {}).get("source", 100)
            item = ContextItem(
                id=source.id or "",
                type="source",
                content=source_context,
                priority=priority
            )
            self.add_item(item)
            
            # Add insights if requested and available
            if self.include_insights and "insights" in inclusion_level:
                try:
                    insights = await source.get_insights()
                    for insight in insights:
                        insight_priority = (self.context_config.priority_weights or {}).get("insight", 75)
                        insight_item = ContextItem(
                            id=insight.id or "",
                            type="insight",
                            content={
                                "id": insight.id,
                                "source_id": source.id,
                                "insight_type": insight.insight_type,
                                "content": insight.content
                            },
                            priority=insight_priority
                        )
                        self.add_item(insight_item)
                except Exception as insight_error:
                    # Log but continue - source context was added successfully
                    logger.warning(
                        f"Failed to add insights for source {source_id}: {insight_error}"
                    )
            
            logger.debug(f"Added source context for {source_id}")
            
        except NotFoundError:
            logger.warning(f"Source {source_id} not found")
        except Exception as e:
            # Changed from raise to warning - allow partial failures
            logger.warning(
                f"Failed to add context for source {source_id}: {e}. "
                f"Continuing with other sources."
            )
            # Don't raise - allow processing to continue
    
    async def _add_notebook_context(self, notebook_id: str) -> None:
        """
        Add notebook content based on context configuration.
        
        Uses concurrent processing with Semaphore to limit parallel requests.
        
        Args:
            notebook_id: ID of the notebook
        """
        try:
            notebook = await Notebook.get(notebook_id)
            if not notebook:
                raise NotFoundError(f"Notebook {notebook_id} not found")
            
            # Wrapper function to safely add source context with semaphore control
            async def safe_add_source(source_id: str, inclusion_level: str = "insights") -> None:
                """Add source context with semaphore-controlled concurrency."""
                async with self._semaphore:
                    try:
                        await self._add_source_context(source_id, inclusion_level)
                    except Exception as e:
                        # Log but don't raise - allow partial failures
                        logger.warning(
                            f"Failed to add context for source {source_id} "
                            f"in notebook {notebook_id}: {e}"
                        )
                        # Re-raise to be caught by gather's return_exceptions
                        raise
            
            # Process sources from context config or get all
            config_sources = self.context_config.sources
            tasks = []
            
            if config_sources:
                # Collect tasks from config
                for source_id, status in config_sources.items():
                    tasks.append(safe_add_source(source_id, status))
            else:
                # Default: get all sources with insights
                sources = await notebook.get_sources()
                for source in sources:
                    if source.id:
                        tasks.append(safe_add_source(source.id, "insights"))
            
            # Execute all tasks concurrently with exception handling
            if tasks:
                import time
                start_time = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Count successes and failures
                success_count = sum(1 for r in results if not isinstance(r, Exception))
                failure_count = len(results) - success_count
                duration = time.time() - start_time
                
                if failure_count > 0:
                    logger.warning(
                        f"Notebook {notebook_id}: Processed {success_count}/{len(tasks)} sources "
                        f"successfully in {duration:.2f}s ({failure_count} failed)"
                    )
                else:
                    logger.info(
                        f"Notebook {notebook_id}: Processed {success_count} sources "
                        f"successfully in {duration:.2f}s"
                    )
            else:
                logger.debug(f"Notebook {notebook_id} has no sources to process")
            
            logger.debug(f"Added notebook context for {notebook_id}")
            
        except Exception as e:
            logger.error(f"Error adding notebook context for {notebook_id}: {str(e)}")
            raise
    
    async def _process_custom_params(self) -> None:
        """Process any additional custom parameters."""
        # Hook for future extensions - can be overridden in subclasses
        # or used to process additional kwargs
        for key, value in self.params.items():
            if key.startswith('custom_'):
                logger.debug(f"Processing custom parameter: {key}={value}")
                # Custom processing logic can be added here
    
    def add_item(self, item: ContextItem) -> None:
        """
        Add a ContextItem to the builder.
        
        Args:
            item: ContextItem to add
        """
        self.items.append(item)
        logger.debug(f"Added item {item.id} with priority {item.priority}")
    
    def add_search_results(
        self, 
        results: List[Dict[str, Any]], 
        search_type: str = "vector",
        max_results: int = 10
    ) -> None:
        """
        Add search results from tools (Vector or PageIndex) to context.
        
        These usually have higher priority than static source content.
        
        Args:
            results: List of search result dictionaries
            search_type: Type of search ("vector" or "pageindex")
            max_results: Maximum number of results to add (default: 10)
        """
        base_priority = 150  # Higher than source (100)
        
        # Limit results to avoid filling entire context window
        limited_results = results[:max_results]
        
        if len(results) > max_results:
            logger.info(
                f"Limiting search results from {len(results)} to {max_results} "
                f"to avoid filling entire context window"
            )
        
        for result in limited_results:
            # Handle PageIndex special structure
            if search_type == "pageindex":
                content = {
                    "type": "pageindex_node",
                    "title": result.get("title"),
                    "content": result.get("content") or result.get("summary"),
                    "text": result.get("text"),
                    "metadata": result.get("metadata", {}),  # Preserve page numbers, sections, etc.
                    "source_id": result.get("source_id"),
                    "source_title": result.get("source_title"),
                }
            else:
                # Vector search results - use as is
                content = result
            
            # Calculate priority based on similarity score if available
            similarity = result.get("similarity", 0.0)
            priority = base_priority + int(similarity * 10)  # Higher similarity = higher priority
            
            item = ContextItem(
                id=f"{search_type}:{result.get('id', 'unknown')}",
                type="search_result",
                content=content,
                priority=priority
            )
            self.add_item(item)
        
        logger.info(f"Added {len(limited_results)} search results from {search_type} search")
    
    def prioritize(self) -> None:
        """Sort items by priority (higher priority first)."""
        self.items.sort(key=lambda x: x.priority, reverse=True)
        logger.debug(f"Prioritized {len(self.items)} items")
    
    def truncate_to_fit(self, max_tokens: int, min_source_ratio: float = 0.2) -> None:
        """
        Remove items if total token count exceeds limit.
        
        Implements a safety mechanism to ensure at least a minimum ratio
        of Source Content is preserved to avoid RAG degradation.
        
        Args:
            max_tokens: Maximum allowed tokens
            min_source_ratio: Minimum ratio of source content to preserve (default: 0.2 = 20%)
        """
        if not max_tokens:
            return
        
        total_tokens = sum(item.token_count or 0 for item in self.items)
        
        if total_tokens <= max_tokens:
            logger.debug(f"Token count {total_tokens} within limit {max_tokens}")
            return
        
        logger.info(f"Truncating from {total_tokens} to {max_tokens} tokens")
        
        # Group items by type for safety mechanism
        items_by_type: Dict[str, List[ContextItem]] = {
            "source": [],
            "insight": [],
            "search_result": [],
            "tool_output": []
        }
        
        for item in self.items:
            items_by_type[item.type].append(item)
        
        # Calculate minimum tokens to preserve for sources
        source_tokens = sum(item.token_count or 0 for item in items_by_type["source"])
        min_source_tokens = int(source_tokens * min_source_ratio)
        
        # Sort all items by priority (lowest first for removal)
        sorted_items = sorted(self.items, key=lambda x: x.priority)
        
        current_tokens = total_tokens
        removed_count = 0
        removed_by_type: Dict[str, int] = {t: 0 for t in items_by_type.keys()}
        preserved_source_tokens = source_tokens
        
        # Remove items from lowest priority, but ensure minimum source content
        for item in sorted_items:
            if current_tokens <= max_tokens:
                break
            
            item_tokens = item.token_count or 0
            
            # Safety check: don't remove if it would violate minimum source ratio
            if item.type == "source":
                if preserved_source_tokens - item_tokens < min_source_tokens:
                    logger.debug(
                        f"Skipping removal of source {item.id} to preserve "
                        f"minimum {min_source_ratio*100:.0f}% source content"
                    )
                    continue
            
            # Safe to remove
            self.items.remove(item)
            current_tokens -= item_tokens
            removed_count += 1
            removed_by_type[item.type] += 1
            
            if item.type == "source":
                preserved_source_tokens -= item_tokens
        
        # Log truncation statistics
        removed_summary = ", ".join([
            f"{count} {t}" for t, count in removed_by_type.items() if count > 0
        ])
        logger.info(
            f"Truncation complete: Removed {removed_count} items ({removed_summary}), "
            f"final token count: {current_tokens}/{max_tokens}. "
            f"Preserved {preserved_source_tokens}/{source_tokens} source tokens "
            f"({preserved_source_tokens/source_tokens*100:.1f}%)"
        )
    
    def remove_duplicates(self) -> None:
        """Remove duplicate items based on ID."""
        seen_ids = set()
        deduplicated_items = []
        
        for item in self.items:
            if item.id not in seen_ids:
                deduplicated_items.append(item)
                seen_ids.add(item.id)
        
        removed_count = len(self.items) - len(deduplicated_items)
        self.items = deduplicated_items
        
        if removed_count > 0:
            logger.debug(f"Removed {removed_count} duplicate items")
    
    def _format_response(self) -> Dict[str, Any]:
        """
        Format the final response.
        
        Returns:
            Formatted context response
        """
        # Group items by type
        sources = []
        insights = []
        search_results = []
        tool_outputs = []
        
        for item in self.items:
            if item.type == "source":
                sources.append(item.content)
            elif item.type == "insight":
                insights.append(item.content)
            elif item.type == "search_result":
                search_results.append(item.content)
            elif item.type == "tool_output":
                tool_outputs.append(item.content)
        
        # Calculate total tokens
        total_tokens = sum(item.token_count or 0 for item in self.items)
        
        # Count items by type
        type_counts = {
            "source": len(sources),
            "insight": len(insights),
            "search_result": len(search_results),
            "tool_output": len(tool_outputs)
        }
        
        response = {
            "sources": sources,
            "insights": insights,
            "search_results": search_results,
            "tool_outputs": tool_outputs,
            "total_tokens": total_tokens,
            "total_items": len(self.items),
            "metadata": {
                "source_count": len(sources),
                "insight_count": len(insights),
                "search_result_count": len(search_results),
                "tool_output_count": len(tool_outputs),
                "type_counts": type_counts,
                "config": {
                    "include_insights": self.include_insights,
                    "max_tokens": self.max_tokens
                }
            }
        }
        
        # Add notebook_id if provided
        if self.notebook_id:
            response["notebook_id"] = self.notebook_id
        
        logger.info(
            f"Built context with {len(self.items)} items ({type_counts}), "
            f"{total_tokens} tokens"
        )
        
        return response


# Convenience functions for common use cases

async def build_notebook_context(
    notebook_id: str,
    context_config: Optional[ContextConfig] = None,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Build context for a notebook.
    
    Args:
        notebook_id: ID of the notebook
        context_config: Optional context configuration
        max_tokens: Optional token limit
    
    Returns:
        Built context
    """
    builder = ContextBuilder(
        notebook_id=notebook_id,
        context_config=context_config,
        max_tokens=max_tokens
    )
    return await builder.build()


async def build_source_context(
    source_id: str,
    include_insights: bool = True,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Build context for a single source.
    
    Args:
        source_id: ID of the source
        include_insights: Whether to include insights
        max_tokens: Optional token limit
    
    Returns:
        Built context
    """
    builder = ContextBuilder(
        source_id=source_id,
        include_insights=include_insights,
        max_tokens=max_tokens
    )
    return await builder.build()


async def build_mixed_context(
    source_ids: Optional[List[str]] = None,
    notebook_id: Optional[str] = None,
    max_tokens: Optional[int] = None
) -> Dict[str, Any]:
    """
    Build context from mixed sources.
    
    Args:
        source_ids: List of source IDs
        notebook_id: Optional notebook ID
        max_tokens: Optional token limit
    
    Returns:
        Built context
    """
    context_config = ContextConfig(max_tokens=max_tokens)
    
    # Configure sources
    if source_ids:
        context_config.sources = {sid: "insights" for sid in source_ids}
    
    builder = ContextBuilder(
        notebook_id=notebook_id,
        context_config=context_config,
        max_tokens=max_tokens
    )
    return await builder.build()