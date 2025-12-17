'use client'

import { useState, useRef, useEffect, useMemo, useCallback, memo, useDeferredValue } from 'react'
import { useQueries } from '@tanstack/react-query'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog'
import { Bot, User, Send, Loader2, FileText, Lightbulb, Clock, ChevronDown } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import {
  SourceChatMessage,
  SourceChatContextIndicator,
  BaseChatSession,
  AgentThinkingStep
} from '@/lib/types/api'
import { ModelSelector } from './ModelSelector'
import { ContextIndicator } from '@/components/common/ContextIndicator'
import { SessionManager } from '@/components/source/SessionManager'
import { MessageActions } from '@/components/source/MessageActions'
import { ThinkingProcessDisplay } from '@/components/source/ThinkingProcessDisplay'
import { AgentDecisionDisplay } from '@/components/source/AgentDecisionDisplay'
import { convertReferencesToCompactMarkdown, createCompactReferenceLinkComponent, parseSourceReferences } from '@/lib/utils/source-references'
import { useModalManager } from '@/lib/hooks/use-modal-manager'
import { useSources } from '@/lib/hooks/use-sources'
import { sourcesApi } from '@/lib/api/sources'
import { insightsApi } from '@/lib/api/insights'
import { QUERY_KEYS } from '@/lib/api/query-client'
import { toast } from 'sonner'

interface NotebookContextStats {
  sourcesInsights: number
  sourcesFull: number
  tokenCount?: number
  charCount?: number
}

interface ChatPanelProps {
  messages: SourceChatMessage[]
  isStreaming: boolean
  contextIndicators: SourceChatContextIndicator | null
  onSendMessage: (message: string, modelOverride?: string) => void
  modelOverride?: string
  onModelChange?: (model?: string) => void
  // Session management props
  sessions?: BaseChatSession[]
  currentSessionId?: string | null
  onCreateSession?: (title: string) => void
  onSelectSession?: (sessionId: string) => void
  onDeleteSession?: (sessionId: string) => void
  onUpdateSession?: (sessionId: string, title: string) => void
  loadingSessions?: boolean
  // Generic props for reusability
  title?: string
  contextType?: 'source' | 'notebook'
  // Notebook context stats (for notebook chat)
  notebookContextStats?: NotebookContextStats
  notebookId?: string
  // Thinking process props (for streaming)
  thinkingSteps?: AgentThinkingStep[]
  isThinking?: boolean
}

export function ChatPanel({
  messages,
  isStreaming,
  contextIndicators,
  onSendMessage,
  modelOverride,
  onModelChange,
  sessions = [],
  currentSessionId,
  onCreateSession,
  onSelectSession,
  onDeleteSession,
  onUpdateSession,
  loadingSessions = false,
  title = 'Chat with Source',
  contextType = 'source',
  notebookContextStats,
  notebookId,
  thinkingSteps = [],
  isThinking = false
}: ChatPanelProps) {
  const [input, setInput] = useState('')
  const [sessionManagerOpen, setSessionManagerOpen] = useState(false)
  const [isNearBottom, setIsNearBottom] = useState(true)
  const [hasNewMessages, setHasNewMessages] = useState(false)
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { openModal } = useModalManager()

  // Scroll handler to detect user position
  const handleScroll = useCallback(() => {
    const scrollArea = scrollAreaRef.current
    if (!scrollArea) return
    
    // Get the actual scrollable viewport element from ScrollArea
    const viewport = scrollArea.querySelector('[data-radix-scroll-area-viewport]') as HTMLElement | null
    if (!viewport) return
    
    const threshold = 100
    const isAtBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight < threshold
    setIsNearBottom(isAtBottom)
    if (isAtBottom) setHasNewMessages(false)
  }, [])

  // Scroll to bottom helper
  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    setHasNewMessages(false)
    setIsNearBottom(true)
  }, [])

  // Fetch sources for notebook chat
  const { data: notebookSources = [] } = useSources(notebookId || undefined)

  // Parse all references from AI messages
  const allReferences = useMemo(() => {
    const references = new Set<{ type: 'source' | 'source_insight', id: string }>()
    messages.forEach(message => {
      if (message.type === 'ai') {
        const parsed = parseSourceReferences(message.content)
        parsed.forEach(ref => {
          references.add({ type: ref.type, id: ref.id })
        })
      }
    })
    return Array.from(references)
  }, [messages])

  // Extract unique source IDs and insight IDs
  const sourceIds = useMemo(() => {
    const ids = new Set<string>()
    allReferences.forEach(ref => {
      if (ref.type === 'source') {
        ids.add(ref.id)
      }
    })
    // Also include sources from contextIndicators
    if (contextIndicators?.sources) {
      contextIndicators.sources.forEach(id => {
        // Remove 'source:' prefix if present
        const cleanId = id.replace(/^source:/, '')
        ids.add(cleanId)
      })
    }
    return Array.from(ids)
  }, [allReferences, contextIndicators])

  const insightIds = useMemo(() => {
    const ids = new Set<string>()
    allReferences.forEach(ref => {
      if (ref.type === 'source_insight') {
        ids.add(ref.id)
      }
    })
    // Also include insights from contextIndicators
    if (contextIndicators?.insights) {
      contextIndicators.insights.forEach(id => {
        // Remove 'source_insight:' prefix if present
        const cleanId = id.replace(/^source_insight:/, '')
        ids.add(cleanId)
      })
    }
    return Array.from(ids)
  }, [allReferences, contextIndicators])

  // Fetch individual sources for source chat (not notebook chat) using useQueries
  const sourceQueries = useQueries({
    queries: sourceIds.map(id => ({
      queryKey: QUERY_KEYS.source(id),
      queryFn: () => sourcesApi.get(id),
      enabled: !!id,
      staleTime: 30 * 1000,
    }))
  })
  
  // Fetch insights to get their source_ids using useQueries
  const insightQueries = useQueries({
    queries: insightIds.map(id => {
      const insightIdWithPrefix = id.includes(':') ? id : `source_insight:${id}`
      return {
        queryKey: ['insights', insightIdWithPrefix],
        queryFn: () => insightsApi.get(insightIdWithPrefix),
        enabled: !!id,
        staleTime: 30 * 1000,
      }
    })
  })

  // Extract source IDs from insights that we need to fetch
  const insightSourceIds = useMemo(() => {
    const ids = new Set<string>()
    insightQueries.forEach(query => {
      if (query.data?.source_id) {
        const sourceId = query.data.source_id.replace(/^source:/, '')
        // Only add if not already in sourceIds
        if (!sourceIds.includes(sourceId)) {
          ids.add(sourceId)
        }
      }
    })
    return Array.from(ids)
  }, [insightQueries, sourceIds])

  // Fetch additional sources for insights using useQueries
  const insightSourceQueries = useQueries({
    queries: insightSourceIds.map(id => ({
      queryKey: QUERY_KEYS.source(id),
      queryFn: () => sourcesApi.get(id),
      enabled: !!id,
      staleTime: 30 * 1000,
    }))
  })

  // Build titleMap
  const titleMap = useMemo(() => {
    const map = new Map<string, string>()

    // For notebook chat: use notebookSources
    if (notebookId && notebookSources.length > 0) {
      notebookSources.forEach(source => {
        if (source.id && source.title) {
          // Handle both formats: with or without 'source:' prefix
          const cleanId = source.id.replace(/^source:/, '')
          map.set(`source:${cleanId}`, source.title)
        }
      })
    }

    // For source chat: use individual source queries
    sourceQueries.forEach((query, index) => {
      const sourceId = sourceIds[index]
      if (sourceId && query.data?.title) {
        map.set(`source:${sourceId}`, query.data.title)
      }
    })

    // Add insight source queries to the map as well
    insightSourceQueries.forEach((query, index) => {
      const sourceId = insightSourceIds[index]
      if (sourceId && query.data?.title) {
        map.set(`source:${sourceId}`, query.data.title)
      }
    })

    // For insights: get source_id from insight, then get source title
    insightQueries.forEach((insightQuery, index) => {
      const insightId = insightIds[index]
      // Use the original insightId without prefix for the map key
      const insightKey = insightId.includes(':') ? insightId : `source_insight:${insightId}`
      
      if (insightQuery.data?.source_id && insightId) {
        const sourceId = insightQuery.data.source_id.replace(/^source:/, '')
        // Find the source title from sourceQueries, insightSourceQueries, or notebookSources
        let sourceTitle: string | null = null

        // Check in notebook sources first
        if (notebookId && notebookSources.length > 0) {
          const source = notebookSources.find(s => {
            const cleanId = (s.id || '').replace(/^source:/, '')
            return cleanId === sourceId
          })
          sourceTitle = source?.title || null
        }

        // If not found, check in source queries
        if (!sourceTitle) {
          const sourceQuery = sourceQueries.find((_, idx) => {
            const id = sourceIds[idx]
            return id === sourceId
          })
          sourceTitle = sourceQuery?.data?.title || null
        }

        // If still not found, check in insight source queries
        if (!sourceTitle) {
          const sourceQuery = insightSourceQueries.find((_, idx) => {
            const id = insightSourceIds[idx]
            return id === sourceId
          })
          sourceTitle = sourceQuery?.data?.title || null
        }

        // Use title if found
        if (sourceTitle) {
          map.set(insightKey, sourceTitle)
        }
      }
    })

    return map
  }, [notebookId, notebookSources, sourceQueries, sourceIds, insightQueries, insightIds, insightSourceQueries, insightSourceIds])

  const handleReferenceClick = (type: string, id: string) => {
    const modalType = type === 'source_insight' ? 'insight' : type as 'source' | 'insight'

    try {
      openModal(modalType, id)
      // Note: The modal system uses URL parameters and doesn't throw errors for missing items.
      // The modal component itself will handle displaying "not found" states.
      // This try-catch is here for future enhancements or unexpected errors.
    } catch {
      const typeLabel = type === 'source_insight' ? 'insight' : type
      toast.error(`This ${typeLabel} could not be found`)
    }
  }

  // Smart auto-scroll: only scroll if user is near bottom
  useEffect(() => {
    if (isNearBottom) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    } else if (messages.length > 0) {
      // User scrolled up, show "new messages" indicator
      setHasNewMessages(true)
    }
  }, [messages, isNearBottom])

  // Attach scroll listener to ScrollArea viewport
  useEffect(() => {
    const scrollArea = scrollAreaRef.current
    if (!scrollArea) return

    const viewport = scrollArea.querySelector('[data-radix-scroll-area-viewport]') as HTMLElement | null
    if (!viewport) return

    viewport.addEventListener('scroll', handleScroll, { passive: true })
    return () => viewport.removeEventListener('scroll', handleScroll)
  }, [handleScroll])

  const handleSend = () => {
    if (input.trim() && !isStreaming) {
      onSendMessage(input.trim(), modelOverride)
      setInput('')
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Detect platform for correct modifier key
    const isMac = typeof navigator !== 'undefined' && navigator.userAgent.toUpperCase().indexOf('MAC') >= 0
    const isModifierPressed = isMac ? e.metaKey : e.ctrlKey

    if (e.key === 'Enter' && isModifierPressed) {
      e.preventDefault()
      handleSend()
    }
  }

  // Detect platform for placeholder text
  const isMac = typeof navigator !== 'undefined' && navigator.userAgent.toUpperCase().indexOf('MAC') >= 0
  const keyHint = isMac ? '⌘+Enter' : 'Ctrl+Enter'

  return (
    <>
    <Card className="flex flex-col h-full flex-1 overflow-hidden">
      <CardHeader className="pb-3 flex-shrink-0">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Bot className="h-5 w-5" />
            {title}
          </CardTitle>
          {onSelectSession && onCreateSession && onDeleteSession && (
            <Dialog open={sessionManagerOpen} onOpenChange={setSessionManagerOpen}>
              <Button
                variant="ghost"
                size="sm"
                className="gap-2"
                onClick={() => setSessionManagerOpen(true)}
                disabled={loadingSessions}
              >
                <Clock className="h-4 w-4" />
                <span className="text-xs">Sessions</span>
              </Button>
              <DialogContent className="sm:max-w-[420px] p-0 overflow-hidden">
                <DialogTitle className="sr-only">Chat Sessions</DialogTitle>
                <SessionManager
                  sessions={sessions}
                  currentSessionId={currentSessionId ?? null}
                  onCreateSession={(title) => onCreateSession?.(title)}
                  onSelectSession={(sessionId) => {
                    onSelectSession(sessionId)
                    setSessionManagerOpen(false)
                  }}
                  onUpdateSession={(sessionId, title) => onUpdateSession?.(sessionId, title)}
                  onDeleteSession={(sessionId) => onDeleteSession?.(sessionId)}
                  loadingSessions={loadingSessions}
                />
              </DialogContent>
            </Dialog>
          )}
        </div>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col min-h-0 p-0 relative">
        <ScrollArea className="flex-1 min-h-0 px-4" ref={scrollAreaRef}>
          <div className="space-y-4 py-4">
            {messages.length === 0 ? (
              <div className="text-center text-muted-foreground py-8">
                <Bot className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p className="text-sm">
                  Start a conversation about this {contextType}
                </p>
                <p className="text-xs mt-2">Ask questions to understand the content better</p>
              </div>
            ) : (
              messages.map((message, index) => {
                const isLastMessage = index === messages.length - 1
                const isStreamingLastMessage = isLastMessage && isStreaming && message.type === 'ai'
                const showLiveThinking = isStreamingLastMessage && thinkingSteps.length > 0

                return (
                  <div
                    key={message.id}
                    className={`flex gap-3 ${
                      message.type === 'human' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    {message.type === 'ai' && (
                      <div className="flex-shrink-0">
                        <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                          <Bot className="h-4 w-4" />
                        </div>
                      </div>
                    )}
                    <div className="flex flex-col gap-2 max-w-[80%]">
                      {/* 決策過程顯示（在消息泡泡上方） */}
                      {message.type === 'ai' && message.thinking_process && (
                        <AgentDecisionDisplay thinkingProcess={message.thinking_process} />
                      )}
                      
                      {/* 即時思考過程顯示（僅在串流中顯示） */}
                      {showLiveThinking && (
                        <ThinkingProcessDisplay 
                          thinkingProcess={{
                            steps: thinkingSteps,
                            total_iterations: 0,
                            total_tool_calls: thinkingSteps.filter(s => s.step_type === 'tool_call').length,
                            search_count: thinkingSteps.filter(s => s.step_type === 'search').length,
                            reasoning_trace: []
                          }}
                        />
                      )}
                      
                      <div
                        className={`rounded-lg px-4 py-2 ${
                          message.type === 'human'
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-muted'
                        }`}
                      >
                        {message.type === 'ai' ? (
                          <AIMessageContent
                            content={message.content}
                            onReferenceClick={handleReferenceClick}
                            titleMap={titleMap}
                          />
                        ) : (
                          <p className="text-sm break-words overflow-wrap-anywhere">{message.content}</p>
                        )}
                      </div>
                      {message.type === 'ai' && (
                        <>
                          {/* 最終思考過程（完成後顯示，不與即時思考重複） */}
                          {message.thinking_process && !showLiveThinking && (
                            <ThinkingProcessDisplay thinkingProcess={message.thinking_process} />
                          )}
                          <MessageActions
                            content={message.content}
                            notebookId={notebookId}
                          />
                        </>
                      )}
                    </div>
                    {message.type === 'human' && (
                      <div className="flex-shrink-0">
                        <div className="h-8 w-8 rounded-full bg-primary flex items-center justify-center">
                          <User className="h-4 w-4 text-primary-foreground" />
                        </div>
                      </div>
                    )}
                  </div>
                )
              })
            )}
            {isStreaming && (
              <div className="flex gap-3 justify-start">
                <div className="flex-shrink-0">
                  <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                    <Bot className="h-4 w-4" />
                  </div>
                </div>
                <div className="rounded-lg px-4 py-2 bg-muted">
                  <Loader2 className="h-4 w-4 animate-spin" />
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        {/* New Messages Button - shows when user scrolls up */}
        {hasNewMessages && (
          <div className="absolute bottom-[200px] left-1/2 -translate-x-1/2 z-10">
            <Button
              variant="secondary"
              size="sm"
              className="shadow-lg gap-1 animate-in fade-in slide-in-from-bottom-2"
              onClick={scrollToBottom}
            >
              <ChevronDown className="h-4 w-4" />
              New messages
            </Button>
          </div>
        )}

        {/* Context Indicators */}
        {contextIndicators && (
          <div className="border-t px-4 py-2">
            <div className="flex flex-wrap gap-2 text-xs">
              {contextIndicators.sources?.length > 0 && (
                <Badge variant="outline" className="gap-1">
                  <FileText className="h-3 w-3" />
                  {contextIndicators.sources.length} source{contextIndicators.sources.length > 1 ? 's' : ''}
                </Badge>
              )}
              {contextIndicators.insights?.length > 0 && (
                <Badge variant="outline" className="gap-1">
                  <Lightbulb className="h-3 w-3" />
                  {contextIndicators.insights.length} insight{contextIndicators.insights.length > 1 ? 's' : ''}
                </Badge>
              )}
            </div>
          </div>
        )}

        {/* Notebook Context Indicator */}
        {notebookContextStats && (
          <ContextIndicator
            sourcesInsights={notebookContextStats.sourcesInsights}
            sourcesFull={notebookContextStats.sourcesFull}
            tokenCount={notebookContextStats.tokenCount}
            charCount={notebookContextStats.charCount}
          />
        )}

        {/* Input Area */}
        <div className="flex-shrink-0 p-4 space-y-3 border-t">
          {/* Model selector */}
          {onModelChange && (
            <div className="flex items-center justify-between">
              <span className="text-xs text-muted-foreground">Model</span>
              <ModelSelector
                currentModel={modelOverride}
                onModelChange={onModelChange}
                disabled={isStreaming}
              />
            </div>
          )}

          <div className="flex gap-2 items-end">
            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={`Ask a question about this ${contextType}... (${keyHint} to send)`}
              disabled={isStreaming}
              className="flex-1 min-h-[40px] max-h-[100px] resize-none py-2 px-3"
              rows={1}
            />
            <Button
              onClick={handleSend}
              disabled={!input.trim() || isStreaming}
              size="icon"
              className="h-[40px] w-[40px] flex-shrink-0"
            >
              {isStreaming ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Send className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>

    </>
  )
}

// Helper component to render AI messages with clickable references
// Memoized to reduce re-renders during streaming
interface AIMessageContentProps {
  content: string
  onReferenceClick: (type: string, id: string) => void
  titleMap?: Map<string, string> | Record<string, string>
}

const AIMessageContent = memo(function AIMessageContent({
  content,
  onReferenceClick,
  titleMap
}: AIMessageContentProps) {
  // Use deferred value to reduce render frequency during fast streaming updates
  const deferredContent = useDeferredValue(content)
  
  // Memoize the markdown conversion for performance
  const markdownWithCompactRefs = useMemo(
    () => convertReferencesToCompactMarkdown(deferredContent, titleMap),
    [deferredContent, titleMap]
  )

  // Memoize the link component to prevent recreation on each render
  const LinkComponent = useMemo(
    () => createCompactReferenceLinkComponent(onReferenceClick),
    [onReferenceClick]
  )

  return (
    <div className="prose prose-sm prose-neutral dark:prose-invert max-w-none break-words prose-headings:font-semibold prose-a:text-blue-600 prose-a:break-all prose-code:bg-muted prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-p:mb-4 prose-p:leading-7 prose-li:mb-2">
      <ReactMarkdown
        components={{
          a: LinkComponent,
          p: ({ children }) => <p className="mb-4">{children}</p>,
          h1: ({ children }) => <h1 className="mb-4 mt-6">{children}</h1>,
          h2: ({ children }) => <h2 className="mb-3 mt-5">{children}</h2>,
          h3: ({ children }) => <h3 className="mb-3 mt-4">{children}</h3>,
          h4: ({ children }) => <h4 className="mb-2 mt-4">{children}</h4>,
          h5: ({ children }) => <h5 className="mb-2 mt-3">{children}</h5>,
          h6: ({ children }) => <h6 className="mb-2 mt-3">{children}</h6>,
          li: ({ children }) => <li className="mb-1">{children}</li>,
          ul: ({ children }) => <ul className="mb-4 space-y-1">{children}</ul>,
          ol: ({ children }) => <ol className="mb-4 space-y-1">{children}</ol>,
        }}
      >
        {markdownWithCompactRefs}
      </ReactMarkdown>
    </div>
  )
})
