'use client'

import { useState, useEffect } from 'react'
import { useParams } from 'next/navigation'
import { PanelGroup, Panel, PanelResizeHandle } from 'react-resizable-panels'
import { AppShell } from '@/components/layout/AppShell'
import { NotebookHeader } from '../components/NotebookHeader'
import { SourcesColumn } from '../components/SourcesColumn'
import { ChatColumn } from '../components/ChatColumn'
import { useNotebook } from '@/lib/hooks/use-notebooks'
import { useSources } from '@/lib/hooks/use-sources'
import { useSettings } from '@/lib/hooks/use-settings'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'

export type ContextMode = 'off' | 'insights' | 'full'

export interface ContextSelections {
  sources: Record<string, ContextMode>
}

export default function NotebookPage() {
  const params = useParams()

  // Ensure the notebook ID is properly decoded from URL
  const notebookId = decodeURIComponent(params.id as string)

  const { data: notebook, isLoading: notebookLoading } = useNotebook(notebookId)
  const { data: sources, isLoading: sourcesLoading, refetch: refetchSources } = useSources(notebookId)
  const { data: settings } = useSettings()

  // Context selection state
  const [contextSelections, setContextSelections] = useState<ContextSelections>({
    sources: {}
  })

  // Panel sizes state
  const [panelSizes, setPanelSizes] = useState<{ sources: number; chat: number }>({
    sources: 33,
    chat: 67
  })

  // Load saved panel sizes from localStorage on mount
  useEffect(() => {
    if (typeof window === 'undefined') return
    try {
      const saved = localStorage.getItem(`notebook-panel-sizes-${notebookId}`)
      if (saved) {
        const parsed = JSON.parse(saved)
        setPanelSizes({
          sources: parsed.sources || 33,
          chat: parsed.chat || 67
        })
      }
    } catch (error) {
      console.error('Failed to load panel sizes:', error)
    }
  }, [notebookId])

  // Save panel sizes to localStorage
  const handlePanelResize = (sizes: number[]) => {
    if (typeof window === 'undefined') return
    try {
      const newSizes = {
        sources: sizes[0] || 33,
        chat: sizes[1] || 67
      }
      setPanelSizes(newSizes)
      localStorage.setItem(
        `notebook-panel-sizes-${notebookId}`,
        JSON.stringify(newSizes)
      )
    } catch (error) {
      console.error('Failed to save panel sizes:', error)
    }
  }

  // Initialize default selections when sources load
  useEffect(() => {
    if (sources && sources.length > 0) {
      setContextSelections(prev => {
        const newSourceSelections = { ...prev.sources }
        // Get default context mode from settings, fallback to 'full'
        const defaultMode = (settings?.default_context_mode as ContextMode) || 'full'
        
        sources.forEach(source => {
          // Only set default if not already set
          if (!(source.id in newSourceSelections)) {
            let mode: ContextMode = defaultMode
            
            // If default is 'insights' but source has no insights, fallback to 'full'
            if (defaultMode === 'insights' && source.insights_count === 0) {
              mode = 'full'
            }
            
            newSourceSelections[source.id] = mode
          }
        })
        return { ...prev, sources: newSourceSelections }
      })
    }
  }, [sources, settings])

  // Handler to update context selection
  const handleContextModeChange = (itemId: string, mode: ContextMode) => {
    setContextSelections(prev => ({
      ...prev,
      sources: {
        ...prev.sources,
        [itemId]: mode
      }
    }))
  }

  if (notebookLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner size="lg" />
      </div>
    )
  }

  if (!notebook) {
    return (
      <AppShell>
        <div className="p-6">
          <h1 className="text-2xl font-bold mb-4">Notebook Not Found</h1>
          <p className="text-muted-foreground">The requested notebook could not be found.</p>
        </div>
      </AppShell>
    )
  }

  return (
    <AppShell>
      <div className="flex flex-col flex-1 min-h-0">
        <div className="flex-shrink-0 p-6 pb-0">
          <NotebookHeader notebook={notebook} />
        </div>

        <div className="flex-1 p-6 pt-6 overflow-hidden">
          {/* Mobile: Stack vertically */}
          <div className="lg:hidden flex flex-col gap-6 h-full min-h-0">
            <div className="flex flex-col h-full min-h-0 overflow-hidden">
              <SourcesColumn
                sources={sources}
                isLoading={sourcesLoading}
                notebookId={notebookId}
                notebookName={notebook?.name}
                onRefresh={refetchSources}
                contextSelections={contextSelections.sources}
                onContextModeChange={(sourceId, mode) => handleContextModeChange(sourceId, mode)}
              />
            </div>
            <div className="flex flex-col h-full min-h-0 overflow-hidden">
              <ChatColumn
                notebookId={notebookId}
                contextSelections={contextSelections}
              />
            </div>
          </div>

          {/* Desktop: Resizable panels */}
          <div className="hidden lg:block h-full min-h-0">
            <PanelGroup
              direction="horizontal"
              className="h-full"
              onLayout={handlePanelResize}
            >
              <Panel
                id="sources-panel"
                defaultSize={panelSizes.sources}
                minSize={25}
                maxSize={75}
                className="flex flex-col h-full min-h-0 overflow-hidden"
              >
                <SourcesColumn
                  sources={sources}
                  isLoading={sourcesLoading}
                  notebookId={notebookId}
                  notebookName={notebook?.name}
                  onRefresh={refetchSources}
                  contextSelections={contextSelections.sources}
                  onContextModeChange={(sourceId, mode) => handleContextModeChange(sourceId, mode)}
                />
              </Panel>

              <PanelResizeHandle className="w-2 bg-transparent hover:bg-border transition-colors cursor-col-resize flex-shrink-0 relative group">
                <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-0.5 bg-border group-hover:bg-primary/50 transition-colors" />
              </PanelResizeHandle>

              <Panel
                id="chat-panel"
                defaultSize={panelSizes.chat}
                minSize={25}
                maxSize={75}
                className="flex flex-col h-full min-h-0 overflow-hidden"
              >
                <ChatColumn
                  notebookId={notebookId}
                  contextSelections={contextSelections}
                />
              </Panel>
            </PanelGroup>
          </div>
        </div>
      </div>
    </AppShell>
  )
}
