'use client'

import { useMemo } from 'react'
import { MCPServerCard } from './MCPServerCard'
import { useMCPServers, useMCPServersStatus } from '@/lib/hooks/use-mcp'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'

export function MCPServerList() {
  const { data: serversData, isLoading, error } = useMCPServers()
  const { data: statusData } = useMCPServersStatus()

  // Create a set of connected server names based on actual connection status
  const connectedServers = useMemo(() => {
    if (!statusData || !statusData.connected || !Array.isArray(statusData.connected)) {
      return new Set<string>()
    }
    // Ensure we only mark servers that are explicitly in the connected array
    return new Set(statusData.connected)
  }, [statusData])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
        <p className="text-sm text-destructive">
          Failed to load MCP servers. Please try again.
        </p>
      </div>
    )
  }

  if (!serversData || !serversData.mcpServers || Object.keys(serversData.mcpServers).length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-12 text-center">
        <p className="text-sm text-muted-foreground">
          No MCP servers configured. Add a server to get started.
        </p>
      </div>
    )
  }

  const servers = Object.entries(serversData.mcpServers)

  return (
    <div className="space-y-4">
      {servers.map(([serverName, config]) => {
        const isConnected = connectedServers.has(serverName)
        return (
          <MCPServerCard
            key={serverName}
            serverName={serverName}
            config={config}
            isConnected={isConnected}
          />
        )
      })}
    </div>
  )
}

