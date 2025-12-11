"""
MCP (Model Context Protocol) Client Service
Manages MCP server connections and tool registration
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP library not available. Install with: pip install mcp>=1.19.0")


class MCPServerManager:
    """Manage MCP server connections and tools."""

    def __init__(self, config_path: Optional[str] = None):
        if not MCP_AVAILABLE:
            logger.warning("MCP not available. MCP features will be disabled.")
            self.available = False
            return

        self.available = True
        self.config_path = config_path or os.getenv(
            "MCP_CONFIG_PATH", "./mcp_config.json"
        )
        self.servers: Dict[str, ClientSession] = {}
        self.server_params: Dict[str, StdioServerParameters] = {}
        self.connection_tasks: Dict[str, asyncio.Task] = {}  # Store background tasks for connections
        self.connection_ready: Dict[str, asyncio.Event] = {}  # Events to signal connection ready
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load MCP server configuration from file."""
        if not self.available:
            return

        config_file = Path(self.config_path)
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
                logger.info(f"Loaded MCP config from {self.config_path}")
            except Exception as e:
                logger.error(f"Error loading MCP config: {e}")
                self.config = {"mcpServers": {}}
        else:
            # Create default config file
            self.config = {"mcpServers": {}}
            self._save_config()
            logger.info(f"Created default MCP config at {self.config_path}")

    def _save_config(self) -> None:
        """Save configuration to file."""
        if not self.available:
            return

        try:
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving MCP config: {e}")

    def add_server(self, server_name: str, config: Dict[str, Any]) -> None:
        """Add new MCP server configuration."""
        if not self.available:
            raise RuntimeError("MCP is not available. Please install mcp>=1.19.0")

        if "mcpServers" not in self.config:
            self.config["mcpServers"] = {}

        self.config["mcpServers"][server_name] = config
        self._save_config()
        logger.info(f"Added MCP server: {server_name}")

    def remove_server(self, server_name: str) -> bool:
        """Remove MCP server configuration."""
        if not self.available:
            return False

        if "mcpServers" in self.config:
            if server_name in self.config["mcpServers"]:
                del self.config["mcpServers"][server_name]
                self._save_config()
                # Disconnect if connected
                if server_name in self.servers:
                    asyncio.create_task(self.disconnect_server(server_name))
                logger.info(f"Removed MCP server: {server_name}")
                return True
        return False

    def get_servers(self) -> Dict[str, Any]:
        """Get all configured MCP servers."""
        if not self.available:
            return {}
        return self.config.get("mcpServers", {})

    async def _maintain_connection(self, server_name: str, server_params: StdioServerParameters):
        """Background task to maintain MCP server connection."""
        ready_event = self.connection_ready.get(server_name)
        try:
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self.servers[server_name] = session
                    if ready_event:
                        ready_event.set()  # Signal that connection is ready
                    logger.info(f"Connected to MCP server: {server_name}")
                    
                    # Keep connection alive - wait for cancellation
                    try:
                        while True:
                            await asyncio.sleep(1)
                            # Check if task still exists (connection should be maintained)
                            if server_name not in self.connection_tasks:
                                break
                    except asyncio.CancelledError:
                        logger.info(f"Connection task for {server_name} cancelled")
                        raise
        except Exception as e:
            logger.error(f"Error in connection task for {server_name}: {e}")
            if server_name in self.servers:
                del self.servers[server_name]
        finally:
            if ready_event:
                ready_event.set()  # Ensure event is set even on error
            # Clean up only when task is cancelled/disconnected
            if server_name in self.connection_ready:
                del self.connection_ready[server_name]
            if server_name in self.connection_tasks:
                del self.connection_tasks[server_name]

    async def connect_server(self, server_name: str) -> bool:
        """Connect to MCP server and maintain persistent connection using background task."""
        if not self.available:
            return False

        if server_name in self.servers and server_name in self.connection_tasks:
            logger.info(f"MCP server {server_name} already connected")
            return True

        servers = self.config.get("mcpServers", {})
        if server_name not in servers:
            logger.error(f"MCP server {server_name} not found in config")
            return False

        server_config = servers[server_name]
        try:
            # Build StdioServerParameters
            command = server_config.get("command", "python")
            args = server_config.get("args", [])
            env = server_config.get("env", {})

            server_params = StdioServerParameters(
                command=command, args=args, env=env if env else None
            )

            self.server_params[server_name] = server_params
            
            # Create ready event
            ready_event = asyncio.Event()
            self.connection_ready[server_name] = ready_event
            
            # Start background task to maintain connection
            task = asyncio.create_task(
                self._maintain_connection(server_name, server_params)
            )
            self.connection_tasks[server_name] = task
            
            # Wait for connection to be ready (with timeout)
            try:
                await asyncio.wait_for(ready_event.wait(), timeout=30.0)
                if server_name in self.servers:
                    logger.info(f"MCP server {server_name} connection established")
                    return True
                else:
                    logger.error(f"Failed to establish connection to {server_name}")
                    return False
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for connection to {server_name}")
                if server_name in self.connection_tasks:
                    self.connection_tasks[server_name].cancel()
                return False

        except Exception as e:
            logger.error(f"Error connecting to MCP server {server_name}: {e}")
            # Clean up on error
            if server_name in self.connection_tasks:
                self.connection_tasks[server_name].cancel()
                try:
                    await self.connection_tasks[server_name]
                except asyncio.CancelledError:
                    pass
                del self.connection_tasks[server_name]
            if server_name in self.connection_ready:
                del self.connection_ready[server_name]
            if server_name in self.servers:
                del self.servers[server_name]
            return False

    async def disconnect_server(self, server_name: str) -> None:
        """Disconnect from MCP server and clean up resources."""
        if not self.available:
            return

        try:
            # Cancel background connection task
            if server_name in self.connection_tasks:
                task = self.connection_tasks[server_name]
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                del self.connection_tasks[server_name]
            
            if server_name in self.connection_ready:
                del self.connection_ready[server_name]
            
            if server_name in self.servers:
                del self.servers[server_name]
            
            if server_name in self.server_params:
                del self.server_params[server_name]
                
            logger.info(f"Disconnected from MCP server: {server_name}")
        except Exception as e:
            logger.error(f"Error disconnecting from MCP server {server_name}: {e}")

    async def get_tools(self, server_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get tools from MCP server(s)."""
        if not self.available:
            return []

        all_tools = []
        servers_to_query = [server_name] if server_name else list(self.servers.keys())

        for name in servers_to_query:
            if name not in self.servers:
                # Try to connect
                connected = await self.connect_server(name)
                if not connected:
                    logger.warning(f"Failed to connect to {name}, skipping tools retrieval")
                    continue
            
            # Ensure connection is ready before querying tools
            if name not in self.servers:
                # Wait for connection to be established
                if name in self.connection_ready:
                    try:
                        await asyncio.wait_for(self.connection_ready[name].wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout waiting for {name} connection to be ready")
                        continue

            if name in self.servers:
                try:
                    session = self.servers[name]
                    # Get tools list
                    tools_response = await session.list_tools()
                    for tool in tools_response.tools:
                        all_tools.append({
                            "name": f"mcp::{name}::{tool.name}",
                            "description": tool.description or "",
                            "inputSchema": tool.inputSchema,
                            "server": name,
                            "tool_name": tool.name,
                        })
                except Exception as e:
                    logger.error(f"Error getting tools from {name}: {e}")

        return all_tools

    async def call_tool(
        self, server_name: str, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Call a tool on an MCP server."""
        if not self.available:
            raise RuntimeError("MCP is not available")

        if server_name not in self.servers:
            connected = await self.connect_server(server_name)
            if not connected:
                raise RuntimeError(f"Failed to connect to MCP server: {server_name}")

        try:
            session = self.servers[server_name]
            result = await session.call_tool(tool_name, arguments or {})
            return result
        except Exception as e:
            logger.error(f"Error calling tool {tool_name} on {server_name}: {e}")
            raise


# Global instance
mcp_manager = MCPServerManager() if MCP_AVAILABLE else None

