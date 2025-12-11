import apiClient from './client'
import {
  MCPServerConfig,
  MCPServersResponse,
  MCPTool,
  RegisterToolsResponse,
  MCPCallToolRequest,
  MCPCallToolResponse,
} from '@/lib/types/mcp'

export const mcpApi = {
  getServers: async () => {
    const response = await apiClient.get<MCPServersResponse>('/mcp/servers')
    return response.data
  },

  getServersStatus: async () => {
    const response = await apiClient.get<{ connected: string[]; total: number }>('/mcp/servers/status')
    return response.data
  },

  addServer: async (serverName: string, config: MCPServerConfig) => {
    const response = await apiClient.post<{ status: string; message: string }>(
      `/mcp/servers?server_name=${encodeURIComponent(serverName)}`,
      config
    )
    return response.data
  },

  addServersBulk: async (serversConfig: MCPServersResponse) => {
    const response = await apiClient.post<{ status: string; message: string }>(
      '/mcp/servers/bulk',
      serversConfig
    )
    return response.data
  },

  deleteServer: async (serverName: string) => {
    const response = await apiClient.delete<{ status: string; message: string }>(
      `/mcp/servers/${encodeURIComponent(serverName)}`
    )
    return response.data
  },

  connectServer: async (serverName: string) => {
    const response = await apiClient.post<{ status: string; message: string }>(
      `/mcp/servers/${encodeURIComponent(serverName)}/connect`
    )
    return response.data
  },

  registerTools: async (serverName: string) => {
    const response = await apiClient.post<RegisterToolsResponse>(
      `/mcp/servers/${encodeURIComponent(serverName)}/register-tools`
    )
    return response.data
  },

  getTools: async (serverName?: string) => {
    const params = serverName ? { server_name: serverName } : {}
    const response = await apiClient.get<MCPTool[]>('/mcp/tools', { params })
    return response.data
  },

  callTool: async (request: MCPCallToolRequest) => {
    const response = await apiClient.post<MCPCallToolResponse>(
      '/mcp/tools/call',
      request
    )
    return response.data
  },
}

