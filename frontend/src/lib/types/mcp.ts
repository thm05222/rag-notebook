export interface MCPServerConfig {
  command: string
  args: string[]
  env?: Record<string, string>
}

export interface MCPServer {
  name: string
  config: MCPServerConfig
}

export interface MCPServersResponse {
  mcpServers: Record<string, MCPServerConfig>
}

export interface MCPTool {
  name: string
  description: string
  inputSchema: Record<string, unknown>
  server: string
}

export interface RegisterToolsResponse {
  status: string
  message: string
  tools_registered: number
  total_tools: number
  registered_tool_names: string[]
}

export interface MCPCallToolRequest {
  server_name: string
  tool_name: string
  arguments?: Record<string, unknown>
}

export interface MCPCallToolResponse {
  status: string
  result: unknown
}

