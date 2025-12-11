"""
Generic ContextBuilder for the Open Notebook project.

This module provides a flexible ContextBuilder class that can handle any parameters
and build context from sources, notebooks, and insights.
"""
from __future__ import annotations

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
    type: Literal["source", "insight"]
    content: Dict[str, Any]
    priority: int = 0
    token_count: Optional[int] = None
    
    def __post_init__(self):
        """Calculate token count for the content if not provided."""
        if self.token_count is None:
            content_str = str(self.content)
            self.token_count = token_count(content_str)


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

        logger.debug(f"ContextBuilder initialized with params: {list(kwargs.keys())}")
    
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
            
            logger.debug(f"Added source context for {source_id}")
            
        except NotFoundError:
            logger.warning(f"Source {source_id} not found")
        except Exception as e:
            logger.error(f"Error adding source context for {source_id}: {str(e)}")
            raise
    
    async def _add_notebook_context(self, notebook_id: str) -> None:
        """
        Add notebook content based on context configuration.
        
        Args:
            notebook_id: ID of the notebook
        """
        try:
            notebook = await Notebook.get(notebook_id)
            if not notebook:
                raise NotFoundError(f"Notebook {notebook_id} not found")
            
            # Process sources from context config or get all
            config_sources = self.context_config.sources
            if config_sources:
                for source_id, status in config_sources.items():
                    await self._add_source_context(source_id, status)
            else:
                # Default: get all sources with insights
                sources = await notebook.get_sources()
                for source in sources:
                    if source.id:
                        await self._add_source_context(source.id, "insights")

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
    
    def prioritize(self) -> None:
        """Sort items by priority (higher priority first)."""
        self.items.sort(key=lambda x: x.priority, reverse=True)
        logger.debug(f"Prioritized {len(self.items)} items")
    
    def truncate_to_fit(self, max_tokens: int) -> None:
        """
        Remove items if total token count exceeds limit.
        
        Args:
            max_tokens: Maximum allowed tokens
        """
        if not max_tokens:
            return
        
        total_tokens = sum(item.token_count or 0 for item in self.items)
        
        if total_tokens <= max_tokens:
            logger.debug(f"Token count {total_tokens} within limit {max_tokens}")
            return
        
        logger.info(f"Truncating from {total_tokens} to {max_tokens} tokens")
        
        # Remove items from the end (lowest priority) until under limit
        current_tokens = total_tokens
        removed_count = 0
        
        while current_tokens > max_tokens and self.items:
            removed_item = self.items.pop()
            current_tokens -= (removed_item.token_count or 0)
            removed_count += 1
        
        logger.info(f"Removed {removed_count} items, final token count: {current_tokens}")
    
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
        
        for item in self.items:
            if item.type == "source":
                sources.append(item.content)
            elif item.type == "insight":
                insights.append(item.content)
        
        # Calculate total tokens
        total_tokens = sum(item.token_count or 0 for item in self.items)
        
        response = {
            "sources": sources,
            "insights": insights,
            "total_tokens": total_tokens,
            "total_items": len(self.items),
            "metadata": {
                "source_count": len(sources),
                "insight_count": len(insights),
                "config": {
                    "include_insights": self.include_insights,
                    "max_tokens": self.max_tokens
                }
            }
        }
        
        # Add notebook_id if provided
        if self.notebook_id:
            response["notebook_id"] = self.notebook_id
        
        logger.info(f"Built context with {len(self.items)} items, {total_tokens} tokens")
        
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