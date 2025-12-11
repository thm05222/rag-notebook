'use client'

import { useState } from 'react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import {
  Plug,
  Trash2,
  Link as LinkIcon,
  Wrench,
  ChevronDown,
  ChevronUp,
} from 'lucide-react'
import { MCPServerConfig, MCPTool } from '@/lib/types/mcp'
import {
  useConnectMCPServer,
  useDeleteMCPServer,
  useRegisterMCPTools,
  useMCPTools,
} from '@/lib/hooks/use-mcp'
import { Separator } from '@/components/ui/separator'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'

interface MCPServerCardProps {
  serverName: string
  config: MCPServerConfig
  isConnected?: boolean
}

export function MCPServerCard({ serverName, config, isConnected }: MCPServerCardProps) {
  const [showDetails, setShowDetails] = useState(false)
  const [showTools, setShowTools] = useState(false)

  const connectMutation = useConnectMCPServer()
  const deleteMutation = useDeleteMCPServer()
  const registerToolsMutation = useRegisterMCPTools()
  const { data: tools, isLoading: toolsLoading, refetch: refetchTools } = useMCPTools(
    serverName
  )

  const handleConnect = async () => {
    await connectMutation.mutateAsync(serverName)
    // Wait a bit for the connection to be established, then fetch tools
    await new Promise((resolve) => setTimeout(resolve, 300))
    await refetchTools()
  }
  
  // Use prop directly - status is managed by parent component via useMCPServersStatus
  // Explicitly convert to boolean to avoid any truthy/falsy issues
  const displayConnected = Boolean(isConnected)

  const handleRegisterTools = async () => {
    await registerToolsMutation.mutateAsync(serverName)
    refetchTools()
  }

  const handleDelete = async () => {
    await deleteMutation.mutateAsync(serverName)
  }

  const serverTools = tools?.filter((tool) => tool.server === serverName) || []

  return (
    <Card>
      <CardHeader>
        <div className="flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <CardTitle className="text-lg">{serverName}</CardTitle>
              {displayConnected ? (
                <Badge className="bg-green-500 hover:bg-green-600">
                  <Plug className="h-3 w-3 mr-1" />
                  Connected
                </Badge>
              ) : (
                <Badge variant="outline" className="text-muted-foreground">
                  Not Connected
                </Badge>
              )}
            </div>
            <CardDescription className="mt-1">
              {config.command} {config.args.join(' ')}
            </CardDescription>
          </div>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowDetails(!showDetails)}
          >
            {showDetails ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </Button>
        </div>
      </CardHeader>

      {showDetails && (
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <div>
              <span className="text-sm font-medium text-muted-foreground">Command:</span>
              <p className="text-sm font-mono bg-muted p-2 rounded mt-1">{config.command}</p>
            </div>
            {config.args.length > 0 && (
              <div>
                <span className="text-sm font-medium text-muted-foreground">Arguments:</span>
                <p className="text-sm font-mono bg-muted p-2 rounded mt-1">
                  {JSON.stringify(config.args, null, 2)}
                </p>
              </div>
            )}
            {config.env && Object.keys(config.env).length > 0 && (
              <div>
                <span className="text-sm font-medium text-muted-foreground">
                  Environment Variables:
                </span>
                <p className="text-sm font-mono bg-muted p-2 rounded mt-1">
                  {JSON.stringify(config.env, null, 2)}
                </p>
              </div>
            )}
          </div>

          <Separator />

          <div className="flex flex-wrap gap-2">
            {!displayConnected ? (
              <Button
                variant="default"
                size="sm"
                onClick={handleConnect}
                disabled={connectMutation.isPending}
              >
                {connectMutation.isPending ? (
                  <LoadingSpinner size="sm" className="mr-2" />
                ) : (
                  <LinkIcon className="h-4 w-4 mr-2" />
                )}
                Connect
              </Button>
            ) : (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleRegisterTools}
                  disabled={registerToolsMutation.isPending}
                >
                  {registerToolsMutation.isPending ? (
                    <LoadingSpinner size="sm" className="mr-2" />
                  ) : (
                    <Wrench className="h-4 w-4 mr-2" />
                  )}
                  Register Tools
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    setShowTools(!showTools)
                    refetchTools()
                  }}
                >
                  {showTools ? 'Hide' : 'Show'} Tools ({serverTools.length})
                </Button>
              </>
            )}

            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="destructive" size="sm" disabled={deleteMutation.isPending}>
                  <Trash2 className="h-4 w-4 mr-2" />
                  Delete
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete MCP Server</AlertDialogTitle>
                  <AlertDialogDescription>
                    Are you sure you want to delete the MCP server &quot;{serverName}&quot;? This
                    action cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={handleDelete} className="bg-destructive">
                    Delete
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>

          {showTools && displayConnected && (
            <div className="space-y-2">
              <Separator />
              <div>
                <span className="text-sm font-medium">Available Tools:</span>
                {toolsLoading ? (
                  <div className="mt-2 flex justify-center">
                    <LoadingSpinner size="sm" />
                  </div>
                ) : serverTools.length > 0 ? (
                  <div className="mt-2 space-y-2">
                    {serverTools.map((tool: MCPTool) => (
                      <div
                        key={tool.name}
                        className="rounded-lg border p-3 bg-muted/50"
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <p className="text-sm font-medium">{tool.name}</p>
                            {tool.description && (
                              <p className="text-xs text-muted-foreground mt-1">
                                {tool.description}
                              </p>
                            )}
                          </div>
                          <Badge variant="secondary" className="ml-2">
                            {tool.server}
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground mt-2">
                    No tools available from this server.
                  </p>
                )}
              </div>
            </div>
          )}
        </CardContent>
      )}
    </Card>
  )
}

