'use client'

import { useState } from 'react'
import { AppShell } from '@/components/layout/AppShell'
import { MCPServerList } from './components/MCPServerList'
import { AddMCPServerDialog } from './components/AddMCPServerDialog'
import { useMCPServers, useMCPTools, useMCPServersStatus } from '@/lib/hooks/use-mcp'
import { Button } from '@/components/ui/button'
import { RefreshCw, Plus } from 'lucide-react'

export default function MCPPage() {
  const [dialogOpen, setDialogOpen] = useState(false)
  const { refetch: refetchServers } = useMCPServers()
  const { refetch: refetchTools } = useMCPTools()
  const { refetch: refetchStatus } = useMCPServersStatus()

  const handleRefresh = () => {
    refetchServers()
    refetchTools()
    refetchStatus()
  }

  return (
    <AppShell>
      <div className="flex-1 overflow-y-auto">
        <div className="p-6">
          <div className="max-w-6xl mx-auto space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold">MCP Servers</h1>
                <p className="text-muted-foreground mt-2">
                  Manage Model Context Protocol (MCP) servers and their tools. Configure servers to
                  extend the system with additional capabilities.
                </p>
              </div>
              <div className="flex gap-2">
                <Button variant="outline" size="sm" onClick={handleRefresh}>
                  <RefreshCw className="h-4 w-4" />
                </Button>
                <Button size="sm" onClick={() => setDialogOpen(true)}>
                  <Plus className="h-4 w-4 mr-2" />
                  Add MCP Server
                </Button>
              </div>
            </div>

            <MCPServerList />

            <AddMCPServerDialog open={dialogOpen} onOpenChange={setDialogOpen} />
          </div>
        </div>
      </div>
    </AppShell>
  )
}

