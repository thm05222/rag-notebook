"""
MCP (Model Context Protocol) API Routes
"""

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from loguru import logger

from api.models import MCPCallToolRequest, MCPServerConfig, MCPServersConfig, MCPToolResponse
from open_notebook.services.mcp_service import MCP_AVAILABLE, mcp_manager

router = APIRouter()


@router.get("/mcp/servers")
async def get_mcp_servers() -> Dict[str, Any]:
    """Get all configured MCP servers."""
    if not MCP_AVAILABLE or not mcp_manager:
        return {"mcpServers": {}}
    
    try:
        servers = mcp_manager.get_servers()
        return {"mcpServers": servers}
    except Exception as e:
        logger.error(f"Error getting MCP servers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mcp/servers")
async def add_mcp_server(server_name: str, config: MCPServerConfig) -> Dict[str, str]:
    """Add a new MCP server."""
    if not MCP_AVAILABLE or not mcp_manager:
        raise HTTPException(status_code=400, detail="MCP is not available. Please install mcp>=1.19.0")
    
    try:
        mcp_manager.add_server(
            server_name,
            {
                "command": config.command,
                "args": config.args,
                "env": config.env,
            }
        )
        return {"status": "success", "message": f"Added MCP server: {server_name}"}
    except Exception as e:
        logger.error(f"Error adding MCP server: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mcp/servers/bulk")
async def add_mcp_servers_bulk(servers_config: MCPServersConfig) -> Dict[str, str]:
    """Bulk add MCP servers (using config file format)."""
    if not MCP_AVAILABLE or not mcp_manager:
        raise HTTPException(status_code=400, detail="MCP is not available. Please install mcp>=1.19.0")
    
    try:
        for server_name, config in servers_config.mcpServers.items():
            mcp_manager.add_server(
                server_name,
                {
                    "command": config.command,
                    "args": config.args,
                    "env": config.env,
                }
            )
        return {
            "status": "success",
            "message": f"Added {len(servers_config.mcpServers)} MCP servers"
        }
    except Exception as e:
        logger.error(f"Error adding MCP servers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/mcp/servers/{server_name}")
async def remove_mcp_server(server_name: str) -> Dict[str, str]:
    """Remove an MCP server."""
    if not MCP_AVAILABLE or not mcp_manager:
        raise HTTPException(status_code=400, detail="MCP is not available")
    
    try:
        removed = mcp_manager.remove_server(server_name)
        if removed:
            return {"status": "success", "message": f"Removed MCP server: {server_name}"}
        else:
            raise HTTPException(status_code=404, detail=f"MCP server not found: {server_name}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing MCP server: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mcp/servers/status")
async def get_mcp_servers_status() -> Dict[str, Any]:
    """Get connection status of all MCP servers."""
    if not MCP_AVAILABLE or not mcp_manager:
        raise HTTPException(status_code=400, detail="MCP is not available")
    
    try:
        # Get list of all configured servers
        servers = mcp_manager.get_servers()
        connected_servers = []
        
        # Check which servers are actually connected by checking both servers dict and connection_tasks
        for server_name in servers.keys():
            # Server is connected if it exists in self.servers AND has an active connection task
            if server_name in mcp_manager.servers and server_name in mcp_manager.connection_tasks:
                # Verify the connection task is still running
                task = mcp_manager.connection_tasks.get(server_name)
                if task and not task.done():
                    connected_servers.append(server_name)
                    logger.debug(f"Server {server_name} is connected (has active task)")
                else:
                    logger.debug(f"Server {server_name} has task but it's done: {task.done() if task else 'no task'}")
            else:
                logger.debug(f"Server {server_name} is NOT connected (servers={server_name in mcp_manager.servers}, tasks={server_name in mcp_manager.connection_tasks})")
        
        logger.info(f"MCP server status: connected={connected_servers}, total={len(servers)}, all_servers={list(servers.keys())}")
        return {
            "connected": connected_servers,
            "total": len(servers),
        }
    except Exception as e:
        logger.error(f"Error getting MCP server status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mcp/servers/{server_name}/connect")
async def connect_mcp_server(server_name: str) -> Dict[str, Any]:
    """Connect to an MCP server and automatically list its tools."""
    if not MCP_AVAILABLE or not mcp_manager:
        raise HTTPException(status_code=400, detail="MCP is not available")
    
    try:
        connected = await mcp_manager.connect_server(server_name)
        if not connected:
            raise HTTPException(status_code=500, detail=f"Failed to connect to MCP server: {server_name}")
        
        # Automatically get tools after successful connection
        tools = await mcp_manager.get_tools(server_name)
        
        return {
            "status": "success",
            "message": f"Connected to MCP server: {server_name}",
            "tools_count": len(tools)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error connecting to MCP server: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mcp/servers/{server_name}/register-tools")
async def register_mcp_tools(server_name: str) -> Dict[str, Any]:
    """Connect to MCP server and register its tools to tool_registry."""
    if not MCP_AVAILABLE or not mcp_manager:
        raise HTTPException(status_code=400, detail="MCP is not available")
    
    try:
        from open_notebook.services.tool_service import MCPToolWrapper, tool_registry
        
        # Connect to server first
        connected = await mcp_manager.connect_server(server_name)
        if not connected:
            raise HTTPException(status_code=500, detail=f"Failed to connect to MCP server: {server_name}")
        
        # Get tools from the server
        tools = await mcp_manager.get_tools(server_name)
        
        # Register each tool to tool_registry
        registered_count = 0
        registered_tools = []
        for tool_info in tools:
            try:
                tool_name = tool_info.get("name", "unknown")
                tool_description = tool_info.get("description", "")
                mcp_wrapper = MCPToolWrapper(tool_info)
                await tool_registry.register(mcp_wrapper)
                registered_count += 1
                registered_tools.append(tool_name)
                logger.info(f"Registered MCP tool: {mcp_wrapper.name} (original: {tool_name}, server: {server_name}, description: {tool_description[:100] if tool_description else 'None'}...)")
            except Exception as e:
                logger.warning(f"Failed to register tool {tool_info.get('name')}: {e}")
        
        return {
            "status": "success",
            "message": f"Registered {registered_count} tools from MCP server: {server_name}",
            "tools_registered": registered_count,
            "total_tools": len(tools),
            "registered_tool_names": registered_tools
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering MCP tools: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mcp/tools")
async def get_mcp_tools(server_name: str = None) -> List[MCPToolResponse]:
    """Get tools from MCP server(s)."""
    if not MCP_AVAILABLE or not mcp_manager:
        return []
    
    try:
        tools = await mcp_manager.get_tools(server_name)
        return [MCPToolResponse(**tool) for tool in tools]
    except Exception as e:
        logger.error(f"Error getting MCP tools: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mcp/tools/call")
async def call_mcp_tool(request: MCPCallToolRequest) -> Dict[str, Any]:
    """Call an MCP tool."""
    if not MCP_AVAILABLE or not mcp_manager:
        raise HTTPException(status_code=400, detail="MCP is not available")
    
    try:
        result = await mcp_manager.call_tool(
            request.server_name,
            request.tool_name,
            request.arguments
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Error calling MCP tool: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

