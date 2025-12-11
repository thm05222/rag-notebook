'use client'

import { useEffect, useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useAddMCPServer, useAddMCPServersBulk } from '@/lib/hooks/use-mcp'
import { MCPServerConfig, MCPServersResponse } from '@/lib/types/mcp'
import { FileCode, FormInput } from 'lucide-react'

const addMCPServerSchema = z.object({
  serverName: z.string().min(1, 'Server name is required'),
  command: z.string().min(1, 'Command is required'),
  args: z.string().optional(),
  env: z.string().optional(),
})

type AddMCPServerFormData = z.infer<typeof addMCPServerSchema>

interface AddMCPServerDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

const EXAMPLE_JSON = `{
  "mcpServers": {
    "yahoo-finance": {
      "command": "uvx",
      "args": ["mcp-yahoo-finance"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/app"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "your-token-here"
      }
    }
  }
}`

export function AddMCPServerDialog({ open, onOpenChange }: AddMCPServerDialogProps) {
  const [activeTab, setActiveTab] = useState<'json' | 'form'>('json')
  const [jsonInput, setJsonInput] = useState<string>('')
  const [jsonError, setJsonError] = useState<string>('')
  const [argsError, setArgsError] = useState<string>('')
  const [envError, setEnvError] = useState<string>('')

  const addMCPServer = useAddMCPServer()
  const addMCPServersBulk = useAddMCPServersBulk()

  const {
    register,
    handleSubmit,
    formState: { errors, isValid },
    reset,
    watch,
  } = useForm<AddMCPServerFormData>({
    resolver: zodResolver(addMCPServerSchema),
    mode: 'onChange',
    defaultValues: {
      serverName: '',
      command: '',
      args: '[]',
      env: '{}',
    },
  })

  const argsValue = watch('args')
  const envValue = watch('env')

  // Validate JSON input
  useEffect(() => {
    if (activeTab === 'json' && jsonInput.trim()) {
      try {
        const parsed = JSON.parse(jsonInput.trim())
        
        // Check if it's the correct format
        if (typeof parsed !== 'object' || Array.isArray(parsed)) {
          setJsonError('Root must be a JSON object')
          return
        }

        if (!parsed.mcpServers || typeof parsed.mcpServers !== 'object') {
          setJsonError('Must contain "mcpServers" object')
          return
        }

          // Validate each server config
        for (const [serverName, config] of Object.entries(parsed.mcpServers)) {
          if (typeof config !== 'object' || Array.isArray(config) || config === null) {
            setJsonError(`Server "${serverName}": config must be an object`)
            return
          }

          const serverConfig = config as Record<string, unknown>
          if (!serverConfig.command || typeof serverConfig.command !== 'string') {
            setJsonError(`Server "${serverName}": "command" is required and must be a string`)
            return
          }

          if (serverConfig.args !== undefined && !Array.isArray(serverConfig.args)) {
            setJsonError(`Server "${serverName}": "args" must be an array`)
            return
          }

          if (serverConfig.env !== undefined && (typeof serverConfig.env !== 'object' || Array.isArray(serverConfig.env))) {
            setJsonError(`Server "${serverName}": "env" must be an object`)
            return
          }
        }

        setJsonError('')
      } catch (e) {
        setJsonError(`Invalid JSON: ${e instanceof Error ? e.message : 'Parse error'}`)
      }
    } else if (activeTab === 'json') {
      setJsonError('')
    }
  }, [jsonInput, activeTab])

  useEffect(() => {
    // Validate args JSON
    if (argsValue && argsValue.trim()) {
      try {
        const parsed = JSON.parse(argsValue)
        if (!Array.isArray(parsed)) {
          setArgsError('Arguments must be a JSON array')
        } else {
          setArgsError('')
        }
      } catch {
        setArgsError('Invalid JSON format for arguments')
      }
    } else {
      setArgsError('')
    }
  }, [argsValue])

  useEffect(() => {
    // Validate env JSON
    if (envValue && envValue.trim()) {
      try {
        const parsed = JSON.parse(envValue)
        if (typeof parsed !== 'object' || Array.isArray(parsed)) {
          setEnvError('Environment variables must be a JSON object')
        } else {
          setEnvError('')
        }
      } catch {
        setEnvError('Invalid JSON format for environment variables')
      }
    } else {
      setEnvError('')
    }
  }, [envValue])

  const closeDialog = () => {
    onOpenChange(false)
    setJsonInput('')
    setJsonError('')
    setArgsError('')
    setEnvError('')
  }

  const handleJsonSubmit = async () => {
    if (jsonError || !jsonInput.trim()) {
      return
    }

    try {
      const parsed = JSON.parse(jsonInput.trim()) as MCPServersResponse
      await addMCPServersBulk.mutateAsync(parsed)
      closeDialog()
    } catch (e) {
      // Error handling is done in the hook
      console.error('Failed to add servers:', e)
    }
  }

  const handleFormSubmit = async (data: AddMCPServerFormData) => {
    let args: string[] = []
    let env: Record<string, string> | undefined = undefined

    // Parse args
    if (data.args && data.args.trim()) {
      try {
        const parsed = JSON.parse(data.args)
        if (Array.isArray(parsed)) {
          args = parsed
        } else {
          setArgsError('Arguments must be a JSON array')
          return
        }
      } catch {
        setArgsError('Invalid JSON format for arguments')
        return
      }
    }

    // Parse env
    if (data.env && data.env.trim()) {
      try {
        const parsed = JSON.parse(data.env)
        if (typeof parsed === 'object' && !Array.isArray(parsed)) {
          env = parsed
        } else {
          setEnvError('Environment variables must be a JSON object')
          return
        }
      } catch {
        setEnvError('Invalid JSON format for environment variables')
        return
      }
    }

    const config: MCPServerConfig = {
      command: data.command,
      args,
      env,
    }

    await addMCPServer.mutateAsync({
      serverName: data.serverName,
      config,
    })

    closeDialog()
    reset()
  }

  useEffect(() => {
    if (!open) {
      reset()
      setJsonInput('')
      setJsonError('')
      setArgsError('')
      setEnvError('')
      setActiveTab('json')
    }
  }, [open, reset])

  const formIsValid = isValid && !argsError && !envError
  const jsonIsValid = !jsonError && jsonInput.trim().length > 0
  const isLoading = addMCPServer.isPending || addMCPServersBulk.isPending

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-[700px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Add MCP Server</DialogTitle>
          <DialogDescription>
            Configure MCP (Model Context Protocol) servers. Paste JSON configuration (recommended)
            or use the form to add a single server.
          </DialogDescription>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as 'json' | 'form')}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="json" className="gap-2">
              <FileCode className="h-4 w-4" />
              JSON Config
            </TabsTrigger>
            <TabsTrigger value="form" className="gap-2">
              <FormInput className="h-4 w-4" />
              Form
            </TabsTrigger>
          </TabsList>

          <TabsContent value="json" className="space-y-4 mt-4">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="json-config">Configuration (JSON)</Label>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => setJsonInput(EXAMPLE_JSON)}
                  className="h-auto py-1 px-2 text-xs"
                >
                  Load Example
                </Button>
              </div>
              <Textarea
                id="json-config"
                value={jsonInput}
                onChange={(e) => setJsonInput(e.target.value)}
                placeholder={EXAMPLE_JSON}
                rows={12}
                className="font-mono text-sm"
              />
              {jsonError && <p className="text-sm text-destructive">{jsonError}</p>}
              <p className="text-xs text-muted-foreground">
                Paste your MCP server configuration in JSON format. You can add multiple servers at
                once. Compatible with Claude Desktop and Cursor MCP configuration format.
              </p>
            </div>

            <DialogFooter className="gap-2 sm:gap-0">
              <Button type="button" variant="outline" onClick={closeDialog}>
                Cancel
              </Button>
              <Button
                type="button"
                onClick={handleJsonSubmit}
                disabled={!jsonIsValid || isLoading}
              >
                {isLoading ? 'Adding…' : 'Add Servers'}
              </Button>
            </DialogFooter>
          </TabsContent>

          <TabsContent value="form" className="space-y-4 mt-4">
            <form onSubmit={handleSubmit(handleFormSubmit)} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="server-name">
                  Server Name <span className="text-destructive">*</span>
                </Label>
                <Input
                  id="server-name"
                  {...register('serverName')}
                  placeholder="e.g., filesystem, github, database"
                  autoFocus
                />
                {errors.serverName && (
                  <p className="text-sm text-destructive">{errors.serverName.message}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="command">
                  Command <span className="text-destructive">*</span>
                </Label>
                <Input
                  id="command"
                  {...register('command')}
                  placeholder="e.g., python, node, npx, uvx"
                />
                {errors.command && (
                  <p className="text-sm text-destructive">{errors.command.message}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="args">Arguments (JSON Array)</Label>
                <Textarea
                  id="args"
                  {...register('args')}
                  placeholder='["-m", "mcp_server", "--option", "value"]'
                  rows={3}
                  className="font-mono text-sm"
                />
                {argsError && <p className="text-sm text-destructive">{argsError}</p>}
                <p className="text-xs text-muted-foreground">
                  JSON array of command-line arguments. Default: []
                </p>
              </div>

              <div className="space-y-2">
                <Label htmlFor="env">Environment Variables (JSON Object)</Label>
                <Textarea
                  id="env"
                  {...register('env')}
                  placeholder='{"API_KEY": "your-key", "DEBUG": "true"}'
                  rows={4}
                  className="font-mono text-sm"
                />
                {envError && <p className="text-sm text-destructive">{envError}</p>}
                <p className="text-xs text-muted-foreground">
                  JSON object with environment variable key-value pairs. Optional.
                </p>
              </div>

              <DialogFooter className="gap-2 sm:gap-0">
                <Button type="button" variant="outline" onClick={closeDialog}>
                  Cancel
                </Button>
                <Button type="submit" disabled={!formIsValid || isLoading}>
                  {isLoading ? 'Adding…' : 'Add Server'}
                </Button>
              </DialogFooter>
            </form>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}
