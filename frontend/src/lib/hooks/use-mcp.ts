import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { mcpApi } from '@/lib/api/mcp'
import { useToast } from '@/lib/hooks/use-toast'
import { MCPServerConfig, MCPServersResponse, MCPCallToolRequest } from '@/lib/types/mcp'

export const MCP_QUERY_KEYS = {
  servers: ['mcp', 'servers'] as const,
  tools: (serverName?: string) => ['mcp', 'tools', serverName] as const,
}

export function useMCPServers() {
  return useQuery({
    queryKey: MCP_QUERY_KEYS.servers,
    queryFn: () => mcpApi.getServers(),
  })
}

export function useMCPServersStatus() {
  return useQuery({
    queryKey: ['mcp', 'servers', 'status'],
    queryFn: () => mcpApi.getServersStatus(),
    // No automatic refresh - only update on connect/disconnect/manual refresh
  })
}

export function useMCPTools(serverName?: string) {
  return useQuery({
    queryKey: MCP_QUERY_KEYS.tools(serverName),
    queryFn: () => mcpApi.getTools(serverName),
    enabled: true,
  })
}

export function useAddMCPServer() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: ({ serverName, config }: { serverName: string; config: MCPServerConfig }) =>
      mcpApi.addServer(serverName, config),
    onSuccess: async () => {
      queryClient.invalidateQueries({ queryKey: MCP_QUERY_KEYS.servers })
      // Refresh status to ensure newly added server shows as disconnected
      await queryClient.invalidateQueries({ queryKey: ['mcp', 'servers', 'status'] })
      await queryClient.refetchQueries({ queryKey: ['mcp', 'servers', 'status'] })
      toast({
        title: 'Success',
        description: 'MCP server added successfully',
      })
    },
    onError: (error: unknown) => {
      const errorMessage =
        (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        'Failed to add MCP server'
      toast({
        title: 'Error',
        description: errorMessage,
        variant: 'destructive',
      })
    },
  })
}

export function useAddMCPServersBulk() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: (serversConfig: MCPServersResponse) => mcpApi.addServersBulk(serversConfig),
    onSuccess: async (data) => {
      queryClient.invalidateQueries({ queryKey: MCP_QUERY_KEYS.servers })
      // Refresh status to ensure newly added servers show as disconnected
      await queryClient.invalidateQueries({ queryKey: ['mcp', 'servers', 'status'] })
      await queryClient.refetchQueries({ queryKey: ['mcp', 'servers', 'status'] })
      toast({
        title: 'Success',
        description: data.message || 'MCP servers added successfully',
      })
    },
    onError: (error: unknown) => {
      const errorMessage =
        (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        'Failed to add MCP servers'
      toast({
        title: 'Error',
        description: errorMessage,
        variant: 'destructive',
      })
    },
  })
}

export function useDeleteMCPServer() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: (serverName: string) => mcpApi.deleteServer(serverName),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: MCP_QUERY_KEYS.servers })
      queryClient.invalidateQueries({ queryKey: ['mcp', 'tools'] })
      queryClient.invalidateQueries({ queryKey: ['mcp', 'servers', 'status'] })
      toast({
        title: 'Success',
        description: 'MCP server deleted successfully',
      })
    },
    onError: (error: unknown) => {
      const errorMessage =
        (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        'Failed to delete MCP server'
      toast({
        title: 'Error',
        description: errorMessage,
        variant: 'destructive',
      })
    },
  })
}

export function useConnectMCPServer() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: (serverName: string) => mcpApi.connectServer(serverName),
    onSuccess: async (_, serverName) => {
      // Invalidate both specific server tools and all tools query
      queryClient.invalidateQueries({ queryKey: MCP_QUERY_KEYS.tools(serverName) })
      queryClient.invalidateQueries({ queryKey: ['mcp', 'tools'] })
      // Invalidate and immediately refetch server status to update connection indicators
      await queryClient.invalidateQueries({ queryKey: ['mcp', 'servers', 'status'] })
      await queryClient.refetchQueries({ queryKey: ['mcp', 'servers', 'status'] })
      toast({
        title: 'Success',
        description: `Connected to MCP server: ${serverName}`,
      })
    },
    onError: (error: unknown, serverName: string) => {
      const errorMessage =
        (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        `Failed to connect to MCP server: ${serverName}`
      toast({
        title: 'Error',
        description: errorMessage,
        variant: 'destructive',
      })
    },
  })
}

export function useRegisterMCPTools() {
  const queryClient = useQueryClient()
  const { toast } = useToast()

  return useMutation({
    mutationFn: (serverName: string) => mcpApi.registerTools(serverName),
    onSuccess: (data, serverName) => {
      queryClient.invalidateQueries({ queryKey: MCP_QUERY_KEYS.tools(serverName) })
      queryClient.invalidateQueries({ queryKey: MCP_QUERY_KEYS.servers })
      queryClient.invalidateQueries({ queryKey: ['mcp', 'tools'] })
      toast({
        title: 'Success',
        description: data.message || `Registered ${data.tools_registered} tools from ${serverName}`,
      })
    },
    onError: (error: unknown, serverName: string) => {
      const errorMessage =
        (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        `Failed to register tools from MCP server: ${serverName}`
      toast({
        title: 'Error',
        description: errorMessage,
        variant: 'destructive',
      })
    },
  })
}

export function useCallMCPTool() {
  const { toast } = useToast()

  return useMutation({
    mutationFn: (request: MCPCallToolRequest) => mcpApi.callTool(request),
    onError: (error: unknown) => {
      const errorMessage =
        (error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        'Failed to call MCP tool'
      toast({
        title: 'Error',
        description: errorMessage,
        variant: 'destructive',
      })
    },
  })
}

