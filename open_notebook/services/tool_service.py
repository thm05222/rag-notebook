"""
Tool service abstraction layer for Agentic RAG.
Provides unified interface for all tools (vector search, text search, PageIndex, MCP, etc.)
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from open_notebook.exceptions import ToolExecutionError, ToolNotFoundError


class BaseTool(ABC):
    """Base class for all tools in the Agentic RAG system."""

    def __init__(
        self,
        name: str,
        description: str,
        timeout: float = 30.0,
        retry_count: int = 2,
        enabled: bool = True,
        parameter_model: Optional[Type[BaseModel]] = None,
    ):
        self.name = name
        self.description = description
        self.timeout = timeout
        self.retry_count = retry_count
        self.enabled = enabled
        self.parameter_model = parameter_model
        # Keep backward compatibility: maintain parameters dict for tools without Pydantic models
        self.parameters: Dict[str, Any] = {}

    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool.
        
        Returns:
            Dict with unified format:
            {
                "tool_name": str,
                "success": bool,
                "data": Any,
                "error": Optional[str],
                "execution_time": float,
                "metadata": Dict[str, Any]
            }
        """
        raise NotImplementedError

    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate tool parameters using Pydantic model if available.
        Falls back to original validation logic for backward compatibility.
        """
        if self.parameter_model is None:
            # Backward compatibility: return True if no model defined
            return True
        
        try:
            self.parameter_model.model_validate(kwargs)
            return True
        except ValidationError as e:
            # Log validation errors for debugging
            error_messages = [err["msg"] for err in e.errors()]
            logger.warning(
                f"Parameter validation failed for tool {self.name}: {error_messages}"
            )
            return False
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """
        Get JSON Schema for tool parameters (OpenAI/Anthropic compatible).
        Returns empty dict if no Pydantic model is defined (backward compatibility).
        """
        if self.parameter_model is None:
            # Backward compatibility: return original parameters dict if no model
            return self.parameters
        
        # Generate JSON Schema from Pydantic model
        schema = self.parameter_model.model_json_schema()
        
        # Convert to OpenAI/Anthropic compatible format
        # Remove Pydantic-specific fields like $defs, $schema, title, etc.
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        # Clean up properties: remove any Pydantic-specific metadata and handle nested structures
        def clean_property(value: Any) -> Any:
            """Recursively clean property values."""
            if isinstance(value, dict):
                cleaned = {}
                for k, v in value.items():
                    # Skip Pydantic-specific keys
                    if k in ["title", "$defs", "$schema"]:
                        continue
                    # Recursively clean nested objects
                    cleaned[k] = clean_property(v)
                return cleaned
            elif isinstance(value, list):
                return [clean_property(item) for item in value]
            else:
                return value
        
        cleaned_properties = {}
        for key, value in properties.items():
            cleaned_properties[key] = clean_property(value)
        
        result = {
            "type": "object",
            "properties": cleaned_properties,
        }
        
        # Only include required if there are required fields
        if required:
            result["required"] = required
        
        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary for LangChain tools."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema(),
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "enabled": self.enabled,
        }

    async def execute_with_retry(self, **kwargs) -> Dict[str, Any]:
        """Execute tool with retry logic and error handling."""
        if not self.enabled:
            return {
                "tool_name": self.name,
                "success": False,
                "data": None,
                "error": f"Tool {self.name} is disabled",
                "execution_time": 0.0,
                "metadata": {},
            }

        # Validate parameters and get detailed error message if validation fails
        validation_result = self.validate_parameters(**kwargs)
        if not validation_result:
            # Try to get detailed validation error message
            error_msg = f"Invalid parameters for tool {self.name}"
            if self.parameter_model:
                try:
                    self.parameter_model.model_validate(kwargs)
                except ValidationError as e:
                    error_details = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
                    error_msg = f"Invalid parameters for tool {self.name}: {', '.join(error_details)}"
            
            return {
                "tool_name": self.name,
                "success": False,
                "data": None,
                "error": error_msg,
                "execution_time": 0.0,
                "metadata": {},
            }

        last_error = None
        start_time = time.time()

        for attempt in range(self.retry_count + 1):
            try:
                result = await asyncio.wait_for(
                    self.execute(**kwargs), timeout=self.timeout
                )
                execution_time = time.time() - start_time

                # Ensure result has required fields
                if not isinstance(result, dict):
                    result = {
                        "tool_name": self.name,
                        "success": True,
                        "data": result,
                        "error": None,
                        "execution_time": execution_time,
                        "metadata": {},
                    }

                result["execution_time"] = execution_time
                return result

            except asyncio.TimeoutError:
                last_error = f"Tool {self.name} execution timeout after {self.timeout}s"
                logger.warning(f"{last_error} (attempt {attempt + 1}/{self.retry_count + 1})")

            except Exception as e:
                last_error = f"Tool {self.name} execution failed: {str(e)}"
                logger.warning(f"{last_error} (attempt {attempt + 1}/{self.retry_count + 1})")
                if attempt < self.retry_count:
                    await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff

        execution_time = time.time() - start_time
        return {
            "tool_name": self.name,
            "success": False,
            "data": None,
            "error": last_error or f"Tool {self.name} execution failed",
            "execution_time": execution_time,
            "metadata": {},
        }


class ToolRegistry:
    """Thread-safe registry for all tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._lock = asyncio.Lock()

    async def register(self, tool: BaseTool) -> None:
        """Register a tool (thread-safe)."""
        async with self._lock:
            if tool.name in self._tools:
                logger.warning(f"Tool {tool.name} already registered, overwriting")
            self._tools[tool.name] = tool
            logger.info(f"Registered tool: {tool.name}")

    async def unregister(self, tool_name: str) -> None:
        """Unregister a tool."""
        async with self._lock:
            if tool_name in self._tools:
                del self._tools[tool_name]
                logger.info(f"Unregistered tool: {tool_name}")

    async def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        async with self._lock:
            return self._tools.get(tool_name)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        async with self._lock:
            return [tool.to_dict() for tool in self._tools.values() if tool.enabled]

    async def execute_tool(
        self, tool_name: str, **kwargs
    ) -> Dict[str, Any]:
        """Execute a tool by name."""
        import time
        start_time = time.time()
        
        tool = await self.get_tool(tool_name)
        if not tool:
            raise ToolNotFoundError(f"Tool {tool_name} not found")

        logger.info(f"ToolRegistry.execute_tool: Starting {tool_name} with timeout={tool.timeout}s")
        result = await tool.execute_with_retry(**kwargs)
        duration = time.time() - start_time
        logger.info(f"ToolRegistry.execute_tool: {tool_name} completed in {duration:.2f}s, success={result.get('success', False)}")
        return result


# Global tool registry instance
tool_registry = ToolRegistry()


# Pydantic parameter models for built-in tools
class VectorSearchParameters(BaseModel):
    """Parameters for vector search tool."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, description="Maximum number of results")
    minimum_score: float = Field(default=0.2, ge=0.0, le=1.0, description="Minimum similarity score (0-1)")
    search_sources: bool = Field(default=True, description="Search in sources")
    notebook_ids: Optional[List[str]] = Field(default=None, description="Optional list of notebook IDs to filter by")
    source_ids: Optional[List[str]] = Field(default=None, description="Optional list of source IDs to filter by. Use this to search within specific documents only.")


class TextSearchParameters(BaseModel):
    """Parameters for text search tool."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, description="Maximum number of results")
    search_sources: bool = Field(default=True, description="Search in sources")
    source_ids: Optional[List[str]] = Field(default=None, description="Optional list of source IDs to filter by. Use this to search within specific documents only.")


class CalculationParameters(BaseModel):
    """Parameters for calculation tool."""
    expression: str = Field(..., description="Mathematical expression to evaluate (e.g., '2+2', '100 * 3.14', 'convert 100 USD to EUR')")


class InternetSearchParameters(BaseModel):
    """Parameters for internet search tool."""
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, description="Maximum number of results")


# Built-in tools
class VectorSearchTool(BaseTool):
    """Vector search tool using Qdrant."""

    def __init__(self):
        super().__init__(
            name="vector_search",
            description="Perform semantic similarity search using vector embeddings. "
            "Best for finding conceptually similar content even when exact keywords don't match.",
            timeout=60.0,
            parameter_model=VectorSearchParameters,
        )

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute vector search."""
        from open_notebook.domain.notebook import vector_search

        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 10)
        minimum_score = kwargs.get("minimum_score", 0.2)
        search_sources = kwargs.get("search_sources", True)
        notebook_ids = kwargs.get("notebook_ids")
        source_ids = kwargs.get("source_ids")

        try:
            results = await vector_search(
                keyword=query,
                results=limit,
                source=search_sources,
                minimum_score=minimum_score,
                notebook_ids=notebook_ids,
                source_ids=source_ids,
            )

            return {
                "tool_name": self.name,
                "success": True,
                "data": results,
                "error": None,
                "execution_time": 0.0,  # Will be set by execute_with_retry
                "metadata": {
                    "result_count": len(results),
                    "query": query,
                    "search_type": "vector",
                },
            }
        except Exception as e:
            # 不 raise，而是返回錯誤結果
            error_msg = str(e)
            logger.error(f"{self.name} failed: {error_msg}")
            return {
                "tool_name": self.name,
                "success": False,
                "data": [],  # 空數組，但包含錯誤信息
                "error": error_msg,
                "execution_time": 0.0,
                "metadata": {
                    "error_type": type(e).__name__,
                    "error_message": error_msg,
                    "query": query,
                },
                "error_details": {
                    "reason": "工具執行失敗",
                    "suggestion": "可能需要檢查查詢參數或嘗試其他工具",
                }
            }


class TextSearchTool(BaseTool):
    """Text search tool using SurrealDB full-text search."""

    def __init__(self):
        super().__init__(
            name="text_search",
            description="Perform keyword-based full-text search using BM25 ranking. "
            "Best for finding specific keywords, phrases, or exact matches.",
            timeout=30.0,
            parameter_model=TextSearchParameters,
        )

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute text search."""
        from open_notebook.domain.notebook import text_search

        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 10)
        search_sources = kwargs.get("search_sources", True)
        source_ids = kwargs.get("source_ids")

        try:
            results = await text_search(
                keyword=query,
                results=limit,
                source=search_sources,
                source_ids=source_ids,
            )

            return {
                "tool_name": self.name,
                "success": True,
                "data": results,
                "error": None,
                "execution_time": 0.0,
                "metadata": {
                    "result_count": len(results) if results else 0,
                    "query": query,
                    "search_type": "text",
                },
            }
        except Exception as e:
            # 不 raise，而是返回錯誤結果
            error_msg = str(e)
            logger.error(f"{self.name} failed: {error_msg}")
            return {
                "tool_name": self.name,
                "success": False,
                "data": [],  # 空數組，但包含錯誤信息
                "error": error_msg,
                "execution_time": 0.0,
                "metadata": {
                    "error_type": type(e).__name__,
                    "error_message": error_msg,
                    "query": query,
                },
                "error_details": {
                    "reason": "工具執行失敗",
                    "suggestion": "可能需要檢查查詢參數或嘗試其他工具",
                }
            }


class InternetSearchTool(BaseTool):
    """Internet search tool using DuckDuckGo."""

    def __init__(self):
        super().__init__(
            name="internet_search",
            description="Search the internet for current information, news, or information not available in the local knowledge base. "
            "Use this when you need up-to-date information, external resources, or information beyond the documents in the workspace.",
            timeout=30.0,
            parameter_model=InternetSearchParameters,
        )

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute internet search using DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ToolExecutionError(
                "duckduckgo-search library is not installed. "
                "Please install it with: pip install duckduckgo-search"
            )

        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 10)

        try:
            # Execute search with region set to 'wt-wt' for worldwide results
            # Note: 'region' is now passed to text() method, not DDGS() constructor
            with DDGS() as ddgs:
                results = list(ddgs.text(query, region='wt-wt', max_results=limit))

            # Format results to match other search tools
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": f"internet_search:{hash(result.get('href', ''))}",
                    "title": result.get("title", ""),
                    "content": result.get("body", ""),
                    "url": result.get("href", ""),
                    "source": "internet_search",
                    "similarity": 1.0,  # Internet search doesn't have similarity scores
                })

            return {
                "tool_name": self.name,
                "success": True,
                "data": formatted_results,
                "error": None,
                "execution_time": 0.0,
                "metadata": {
                    "result_count": len(formatted_results),
                    "query": query,
                    "search_type": "internet",
                },
            }
        except Exception as e:
            # 不 raise，而是返回錯誤結果
            error_msg = str(e)
            logger.error(f"{self.name} failed: {error_msg}")
            return {
                "tool_name": self.name,
                "success": False,
                "data": [],
                "error": error_msg,
                "execution_time": 0.0,
                "metadata": {
                    "error_type": type(e).__name__,
                    "error_message": error_msg,
                    "query": query,
                },
                "error_details": {
                    "reason": "工具執行失敗",
                    "suggestion": "可能需要檢查查詢參數或嘗試其他工具",
                }
            }


class MCPToolWrapper(BaseTool):
    """Wrapper for MCP server tools."""

    def __init__(self, tool_info: Dict[str, Any]):
        tool_name = tool_info.get("name", "unknown")
        description = tool_info.get("description", "")
        server_name = tool_info.get("server", "unknown")
        actual_tool_name = tool_info.get("tool_name", tool_name.split("::")[-1] if "::" in tool_name else tool_name)

        super().__init__(
            name=tool_name,
            description=description or f"MCP tool from server {server_name}",
            timeout=60.0,
            # MCP tools don't use Pydantic models - they use dynamic schemas from MCP server
        )
        self.server_name = server_name
        self.actual_tool_name = actual_tool_name
        self.input_schema = tool_info.get("inputSchema", {})
    
    def get_parameters_schema(self) -> Dict[str, Any]:
        """Return MCP input schema directly."""
        return self.input_schema

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute MCP tool."""
        from open_notebook.services.mcp_service import mcp_manager

        if not mcp_manager or not mcp_manager.available:
            return {
                "tool_name": self.name,
                "success": False,
                "data": None,
                "error": "MCP is not available",
                "execution_time": 0.0,
                "metadata": {},
            }

        try:
            result = await mcp_manager.call_tool(
                self.server_name, self.actual_tool_name, kwargs
            )

            return {
                "tool_name": self.name,
                "success": True,
                "data": result,
                "error": None,
                "execution_time": 0.0,
                "metadata": {
                    "server": self.server_name,
                    "mcp_tool": self.actual_tool_name,
                },
            }
        except Exception as e:
            raise ToolExecutionError(f"MCP tool {self.name} failed: {str(e)}") from e