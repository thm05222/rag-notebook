"""
Tool service abstraction layer for Agentic RAG.
Provides unified interface for all tools (vector search, text search, PageIndex, MCP, etc.)
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from loguru import logger

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
    ):
        self.name = name
        self.description = description
        self.timeout = timeout
        self.retry_count = retry_count
        self.enabled = enabled
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
        """Validate tool parameters. Override in subclasses if needed."""
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary for LangChain tools."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
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

        if not self.validate_parameters(**kwargs):
            return {
                "tool_name": self.name,
                "success": False,
                "data": None,
                "error": f"Invalid parameters for tool {self.name}",
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


# Built-in tools
class VectorSearchTool(BaseTool):
    """Vector search tool using Qdrant."""

    def __init__(self):
        super().__init__(
            name="vector_search",
            description="Perform semantic similarity search using vector embeddings. "
            "Best for finding conceptually similar content even when exact keywords don't match.",
            timeout=60.0,
        )
        self.parameters = {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Maximum number of results", "default": 10},
            "minimum_score": {
                "type": "number",
                "description": "Minimum similarity score (0-1)",
                "default": 0.2,
            },
            "search_sources": {
                "type": "boolean",
                "description": "Search in sources",
                "default": True,
            },
            "notebook_ids": {
                "type": "array",
                "description": "Optional list of notebook IDs to filter by",
                "items": {"type": "string"},
            },
        }

    def validate_parameters(self, **kwargs) -> bool:
        """Validate vector search parameters."""
        if "query" not in kwargs or not kwargs["query"]:
            return False
        if "limit" in kwargs and (not isinstance(kwargs["limit"], int) or kwargs["limit"] <= 0):
            return False
        if "minimum_score" in kwargs and (
            not isinstance(kwargs["minimum_score"], (int, float))
            or kwargs["minimum_score"] < 0
            or kwargs["minimum_score"] > 1
        ):
            return False
        return True

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute vector search."""
        from open_notebook.domain.notebook import vector_search

        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 10)
        minimum_score = kwargs.get("minimum_score", 0.2)
        search_sources = kwargs.get("search_sources", True)
        notebook_ids = kwargs.get("notebook_ids")

        try:
            results = await vector_search(
                keyword=query,
                results=limit,
                source=search_sources,
                minimum_score=minimum_score,
                notebook_ids=notebook_ids,
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
        )
        self.parameters = {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Maximum number of results", "default": 10},
            "search_sources": {
                "type": "boolean",
                "description": "Search in sources",
                "default": True,
            },
        }

    def validate_parameters(self, **kwargs) -> bool:
        """Validate text search parameters."""
        if "query" not in kwargs or not kwargs["query"]:
            return False
        if "limit" in kwargs and (not isinstance(kwargs["limit"], int) or kwargs["limit"] <= 0):
            return False
        return True

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute text search."""
        from open_notebook.domain.notebook import text_search

        query = kwargs.get("query", "")
        limit = kwargs.get("limit", 10)
        search_sources = kwargs.get("search_sources", True)

        try:
            results = await text_search(
                keyword=query,
                results=limit,
                source=search_sources,
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


class CalculationTool(BaseTool):
    """Mathematical calculation and unit conversion tool."""

    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations and unit conversions. "
            "Supports basic arithmetic, unit conversions, and simple formulas.",
            timeout=5.0,
        )
        self.parameters = {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2+2', '100 * 3.14', 'convert 100 USD to EUR')",
            },
        }

    def validate_parameters(self, **kwargs) -> bool:
        """Validate calculation parameters."""
        if "expression" not in kwargs or not kwargs["expression"]:
            return False
        return True

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute calculation."""
        import math
        import re

        expression = kwargs.get("expression", "").strip()

        try:
            # Basic safety check - only allow numbers, operators, and common functions
            allowed_chars = set("0123456789+-*/()., ")
            if not all(c in allowed_chars or c.isalpha() for c in expression):
                raise ValueError("Expression contains invalid characters")

            # Handle unit conversions (basic implementation)
            if "convert" in expression.lower() or "to" in expression.lower():
                # Simple unit conversion placeholder
                # In production, use a proper unit conversion library
                result = f"Unit conversion requested: {expression}. "
                result += "Note: Full unit conversion support requires additional library integration."
            else:
                # Safe evaluation of mathematical expressions
                # Remove any function calls for basic safety
                safe_expr = re.sub(r"[a-zA-Z_]+", "", expression)
                result = eval(safe_expr, {"__builtins__": {}}, {"math": math})

            return {
                "tool_name": self.name,
                "success": True,
                "data": {"expression": expression, "result": result},
                "error": None,
                "execution_time": 0.0,
                "metadata": {"calculation_type": "arithmetic"},
            }
        except Exception as e:
            raise ToolExecutionError(f"Calculation failed: {str(e)}") from e


class InternetSearchTool(BaseTool):
    """Internet search tool using DuckDuckGo."""

    def __init__(self):
        super().__init__(
            name="internet_search",
            description="Search the internet for current information, news, or information not available in the local knowledge base. "
            "Use this when you need up-to-date information, external resources, or information beyond the documents in the workspace.",
            timeout=30.0,
        )
        self.parameters = {
            "query": {"type": "string", "description": "Search query"},
            "limit": {"type": "integer", "description": "Maximum number of results", "default": 10},
        }

    def validate_parameters(self, **kwargs) -> bool:
        """Validate internet search parameters."""
        if "query" not in kwargs or not kwargs["query"]:
            return False
        if "limit" in kwargs and (not isinstance(kwargs["limit"], int) or kwargs["limit"] <= 0):
            return False
        return True

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
            # Execute search
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=limit))

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
        except ImportError:
            # 不 raise，而是返回錯誤結果
            error_msg = "duckduckgo-search library is not installed. Please install it with: pip install duckduckgo-search"
            logger.error(f"{self.name} failed: {error_msg}")
            return {
                "tool_name": self.name,
                "success": False,
                "data": [],
                "error": error_msg,
                "execution_time": 0.0,
                "metadata": {
                    "error_type": "ImportError",
                    "error_message": error_msg,
                    "query": query,
                },
                "error_details": {
                    "reason": "缺少必要的依賴庫",
                    "suggestion": "請安裝 duckduckgo-search 庫：pip install duckduckgo-search",
                }
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
        )
        self.server_name = server_name
        self.actual_tool_name = actual_tool_name
        self.input_schema = tool_info.get("inputSchema", {})

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

