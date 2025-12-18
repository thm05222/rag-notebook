'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { toast } from 'sonner'
import { chatApi } from '@/lib/api/chat'
import { QUERY_KEYS } from '@/lib/api/query-client'
import {
  NotebookChatMessage,
  CreateNotebookChatSessionRequest,
  UpdateNotebookChatSessionRequest,
  SourceListResponse,
  ChatStreamEvent,
  AgentThinkingStep
} from '@/lib/types/api'
import { ContextSelections } from '@/app/(dashboard)/notebooks/[id]/page'

interface UseNotebookChatParams {
  notebookId: string
  sources: SourceListResponse[]
  contextSelections: ContextSelections
}

// 輔助函數：合併消息列表並去重
const mergeMessages = (local: NotebookChatMessage[], remote: NotebookChatMessage[]): NotebookChatMessage[] => {
  const messageMap = new Map<string, NotebookChatMessage>();
  
  // 先放原本的（保留前端可能的臨時狀態）
  local.forEach(m => {
    const key = m.id || `${m.type}-${m.content?.substring(0, 50)}`; // 優先用 ID，沒有 ID 用內容當 Key
    messageMap.set(key, m);
  });
  
  // 再放後端的（以服務器為準，但要確保不覆蓋正在生成的流）
  remote.forEach(m => {
    const key = m.id || `${m.type}-${m.content?.substring(0, 50)}`;
    messageMap.set(key, m);
  });
  
  return Array.from(messageMap.values());
};

export function useNotebookChat({ notebookId, sources, contextSelections }: UseNotebookChatParams) {
  const queryClient = useQueryClient()
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<NotebookChatMessage[]>([])
  const [isSending, setIsSending] = useState(false)
  const [tokenCount, setTokenCount] = useState<number>(0)
  const [charCount, setCharCount] = useState<number>(0)
  const [thinkingSteps, setThinkingSteps] = useState<AgentThinkingStep[]>([])
  const [isThinking, setIsThinking] = useState(false)
  const [isSwitchingSession, setIsSwitchingSession] = useState(false)
  
  // 使用 useRef 追蹤上一次的 sessionId，避免不必要的清空
  const prevSessionIdRef = useRef<string | null>(null)

  // Fetch sessions for this notebook
  const {
    data: sessions = [],
    isLoading: loadingSessions,
    refetch: refetchSessions
  } = useQuery({
    queryKey: QUERY_KEYS.notebookChatSessions(notebookId),
    queryFn: () => chatApi.listSessions(notebookId),
    enabled: !!notebookId
  })

  // Fetch current session with messages
  const {
    data: currentSession,
    refetch: refetchCurrentSession,
    isLoading: isSessionLoading,
    isFetching: isSessionFetching
  } = useQuery({
    queryKey: QUERY_KEYS.notebookChatSession(currentSessionId!),
    queryFn: () => chatApi.getSession(currentSessionId!),
    enabled: !!notebookId && !!currentSessionId
  })

  // Update messages when current session changes
  useEffect(() => {
    // 1. 如果正在流式傳輸中 (Sending)，絕對不要用後端數據覆蓋前端，也不要清空消息
    // 這是最高優先級的檢查，必須放在最前面
    if (isSending) {
      return
    }

    // 2. 檢查 session ID 是否真的改變了
    // 但只在非流式傳輸時才清空，避免清空正在生成的流
    const sessionIdChanged = prevSessionIdRef.current !== currentSessionId
    if (sessionIdChanged) {
      prevSessionIdRef.current = currentSessionId
      // Session ID 改變時，立即清空消息（switchSession 已經做了，這裡是雙重保險）
      // 但只在非流式傳輸時清空
      if (messages.length > 0) {
        setMessages([])
      }
    }

    // 3. 如果 currentSession 尚未加載，或者 ID 不匹配（舊數據殘留），等待
    if (!currentSession || (currentSessionId && currentSession.id !== currentSessionId)) {
      return
    }

    // 4. 數據同步核心邏輯：只要後端有數據，且 ID 對得上，就更新
    if (currentSession.messages && currentSession.messages.length > 0) {
      setMessages((prev) => {
        const backendMsgs = currentSession.messages || []
        
        // 性能優化：如果長度與內容都一樣，就不觸發 re-render
        if (prev.length === backendMsgs.length) {
          const lastPrev = prev[prev.length - 1]
          const lastBackend = backendMsgs[backendMsgs.length - 1]
          if (lastPrev?.id === lastBackend?.id && lastPrev?.content === lastBackend?.content) {
            return prev
          }
        }
        
        // 否則更新為後端數據
        return backendMsgs
      })
    } else if (currentSession && (!currentSession.messages || currentSession.messages.length === 0)) {
      // 處理後端返回空陣列的情況（例如新對話或已清空的會話）
      // 但只在非流式傳輸時清空，避免清空正在生成的流
      if (messages.length > 0) {
        setMessages([])
      }
    }

    // 清除切換狀態
    if (!isSessionLoading && !isSessionFetching) {
      setIsSwitchingSession(false)
    }
  }, [currentSession, currentSessionId, isSending, isSessionLoading, isSessionFetching])

  // Auto-select most recent session when sessions are loaded
  useEffect(() => {
    if (sessions.length > 0 && !currentSessionId) {
      // Sessions are sorted by created date desc from API
      const mostRecentSession = sessions[0]
      setCurrentSessionId(mostRecentSession.id)
    }
  }, [sessions, currentSessionId])

  // Create session mutation
  const createSessionMutation = useMutation({
    mutationFn: (data: CreateNotebookChatSessionRequest) =>
      chatApi.createSession(data),
    onSuccess: (newSession) => {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.notebookChatSessions(notebookId)
      })
      setCurrentSessionId(newSession.id)
      toast.success('Chat session created')
    },
    onError: () => {
      toast.error('Failed to create chat session')
    }
  })

  // Update session mutation
  const updateSessionMutation = useMutation({
    mutationFn: ({ sessionId, data }: {
      sessionId: string
      data: UpdateNotebookChatSessionRequest
    }) => chatApi.updateSession(sessionId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.notebookChatSessions(notebookId)
      })
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.notebookChatSession(currentSessionId!)
      })
      toast.success('Session updated')
    },
    onError: () => {
      toast.error('Failed to update session')
    }
  })

  // Delete session mutation
  const deleteSessionMutation = useMutation({
    mutationFn: (sessionId: string) =>
      chatApi.deleteSession(sessionId),
    onSuccess: (_, deletedId) => {
      queryClient.invalidateQueries({
        queryKey: QUERY_KEYS.notebookChatSessions(notebookId)
      })
      if (currentSessionId === deletedId) {
        setCurrentSessionId(null)
        setMessages([])
      }
      toast.success('Session deleted')
    },
    onError: () => {
      toast.error('Failed to delete session')
    }
  })

  // Build context from sources based on user selections
  const buildContext = useCallback(async () => {
    // Build context_config mapping IDs to selection modes
    const context_config: { sources: Record<string, string> } = {
      sources: {}
    }

    // Map source selections
    sources.forEach(source => {
      const mode = contextSelections.sources[source.id]
      if (mode === 'insights') {
        context_config.sources[source.id] = 'insights'
      } else if (mode === 'full') {
        context_config.sources[source.id] = 'full content'
      } else {
        context_config.sources[source.id] = 'not in'
      }
    })

    // Call API to build context with actual content
    const response = await chatApi.buildContext({
      notebook_id: notebookId,
      context_config
    })

    // Store token and char counts
    setTokenCount(response.token_count)
    setCharCount(response.char_count)

    return response.context
  }, [notebookId, sources, contextSelections])

  // Helper function to convert stream event to thinking step
  const convertToThinkingStep = useCallback((event: ChatStreamEvent): AgentThinkingStep | null => {
    if (event.type !== 'thought' || !event.content) {
      return null
    }

    let stepType: AgentThinkingStep['step_type'] = 'decision'
    if (event.stage === 'executing') {
      stepType = 'tool_call'
    } else if (event.stage === 'planning') {
      stepType = 'decision'
    } else if (event.stage === 'synthesizing') {
      stepType = 'synthesis'
    }

    return {
      step_type: stepType,
      timestamp: event.timestamp || Date.now() / 1000,
      content: event.content,
      metadata: event.metadata || {}
    }
  }, [])

  // Send message (streaming)
  const sendMessage = useCallback(async (message: string, modelOverride?: string) => {
    let sessionId = currentSessionId

    // Auto-create session if none exists
    if (!sessionId) {
      try {
        const defaultTitle = message.length > 30
          ? `${message.substring(0, 30)}...`
          : message
        const newSession = await chatApi.createSession({
          notebook_id: notebookId,
          title: defaultTitle
        })
        sessionId = newSession.id
        setCurrentSessionId(sessionId)
        queryClient.invalidateQueries({
          queryKey: QUERY_KEYS.notebookChatSessions(notebookId)
        })
      } catch {
        toast.error('Failed to create chat session')
        return
      }
    }

    // Reset thinking state
    setThinkingSteps([])
    setIsThinking(false)

    // Add user message optimistically
    const userMessage: NotebookChatMessage = {
      id: `temp-${Date.now()}`,
      type: 'human',
      content: message,
      timestamp: new Date().toISOString()
    }
    setMessages(prev => [...prev, userMessage])
    setIsSending(true)

    try {
      // Build context and send message
      const context = await buildContext()
      const response = await chatApi.sendMessage({
        session_id: sessionId,
        message,
        context,
        model_override: modelOverride ?? (currentSession?.model_override ?? undefined),
        notebook_id: notebookId  // Include notebook_id for auto-creating session if not found
      })

      if (!response) {
        throw new Error('No response body received')
      }

      const reader = response.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      let aiMessage: NotebookChatMessage | null = null
      const startTime = Date.now() / 1000

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')

        // Keep the last incomplete line in buffer
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const jsonStr = line.slice(6).trim()
              if (!jsonStr || jsonStr === '[DONE]') continue

              const data: ChatStreamEvent = JSON.parse(jsonStr)

              if (data.type === 'thought') {
                setIsThinking(true)
                const step = convertToThinkingStep(data)
                if (step) {
                  // Adjust timestamp to be relative to start time
                  step.timestamp = startTime + (Date.now() / 1000 - startTime)
                  setThinkingSteps(prev => [...prev, step])
                }
              } else if (data.type === 'content') {
                setIsThinking(false) // Stop showing thinking when content starts
                // Create AI message on first content chunk
                if (!aiMessage) {
                  aiMessage = {
                    id: `ai-${Date.now()}`,
                    type: 'ai',
                    content: data.content || '',
                    timestamp: new Date().toISOString()
                  }
                  setMessages(prev => [...prev, aiMessage!])
                } else {
                  // Append content to existing message
                  aiMessage.content += data.content || ''
                  setMessages(prev =>
                    prev.map(msg => msg.id === aiMessage!.id
                      ? { ...msg, content: aiMessage!.content }
                      : msg
                    )
                  )
                }
              } else if (data.type === 'error') {
                throw new Error(data.message || 'Stream error occurred')
              } else if (data.type === 'complete') {
                // Stop thinking indicator
                setIsThinking(false)
                // If session was auto-created, update currentSessionId and immediately fetch session data
                if (data.session_id && data.session_id !== sessionId) {
                  console.log(`Session auto-created: ${sessionId} -> ${data.session_id}`)
                  setCurrentSessionId(data.session_id)
                  // Invalidate sessions list to include the new session
                  queryClient.invalidateQueries({
                    queryKey: QUERY_KEYS.notebookChatSessions(notebookId)
                  })
                  // Immediately fetch the new session data using the new sessionId
                  // This ensures the conversation history is properly saved and displayed
                  try {
                    const newSessionData = await chatApi.getSession(data.session_id)
                    // Update the query cache with the new session data
                    queryClient.setQueryData(
                      QUERY_KEYS.notebookChatSession(data.session_id),
                      newSessionData
                    )
                    // 使用合併策略，而不是完全替換
                    if (newSessionData.messages) {
                      setMessages(prev => {
                        // 使用 mergeMessages 函數合併消息
                        return mergeMessages(prev, newSessionData.messages)
                      })
                    }
                  } catch (error) {
                    console.error('Failed to fetch new session data:', error)
                  }
                }
                // Final thinking process will be loaded via refetchCurrentSession
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e, 'Line:', line)
              // Don't throw - continue processing other lines
            }
          }
        }
      }

      // Ensure streaming is stopped
      setIsSending(false)
      setIsThinking(false)

      // 延遲 refetch，確保後端已保存消息
      setTimeout(async () => {
        try {
          const updatedSession = await refetchCurrentSession()
          if (updatedSession.data?.messages) {
            setMessages(prev => {
              // 如果後端消息數量明顯比較多，或者長度一樣但內容不同，使用後端的比較安全
              if (updatedSession.data.messages.length >= prev.length) {
                // 使用合併策略，確保不丟失任何消息
                return mergeMessages(prev, updatedSession.data.messages)
              }
              return prev; // 後端資料還沒跟上，保留前端的
            })
          }
        } catch (error) {
          console.error('Failed to sync session:', error)
          // 不拋出錯誤，保留當前消息
        }
      }, 500) // 延遲 500ms 確保後端已保存，可以考慮 retry 機制
    } catch (error) {
      console.error('Error sending message:', error)
      
      // Extract detailed error message
      let errorMessage = 'Failed to send message'
      let isModelConfigError = false
      
      if (error instanceof Error) {
        errorMessage = error.message || errorMessage
        
        // Check if it's a model configuration error
        if (errorMessage.toLowerCase().includes('model') ||
            errorMessage.toLowerCase().includes('configuration') ||
            errorMessage.toLowerCase().includes('no longer exists')) {
          isModelConfigError = true
          errorMessage = errorMessage.replace(
            /Please configure.*?Models\./i,
            'Please go to Settings > Models to configure your default models.'
          )
        }
      }
      
      // Show error toast with appropriate styling
      if (isModelConfigError) {
        toast.error(errorMessage, {
          duration: 8000, // Longer duration for important messages
        })
      } else {
        toast.error(errorMessage)
      }
      
      // Remove optimistic message on error
      setMessages(prev => prev.filter(msg => !msg.id.startsWith('temp-')))
      setThinkingSteps([])
      setIsThinking(false)
    } finally {
      setIsSending(false)
    }
  }, [
    notebookId,
    currentSessionId,
    currentSession,
    buildContext,
    refetchCurrentSession,
    queryClient,
    convertToThinkingStep
  ])

  // Switch session
  const switchSession = useCallback((sessionId: string) => {
    setIsSwitchingSession(true); // 設置切換狀態
    setCurrentSessionId(sessionId);
    setMessages([]); // 1. 立即清空，防止殘影
    // 2. 觸發 React Query 加載新 ID
    // useEffect 會在 currentSession 加載完成後自動更新消息並清除 isSwitchingSession
  }, [])

  // Create session
  const createSession = useCallback((title?: string) => {
    return createSessionMutation.mutate({
      notebook_id: notebookId,
      title
    })
  }, [createSessionMutation, notebookId])

  // Update session
  const updateSession = useCallback((sessionId: string, data: UpdateNotebookChatSessionRequest) => {
    return updateSessionMutation.mutate({
      sessionId,
      data
    })
  }, [updateSessionMutation])

  // Delete session
  const deleteSession = useCallback((sessionId: string) => {
    return deleteSessionMutation.mutate(sessionId)
  }, [deleteSessionMutation])

  // Update token/char counts when context selections change
  useEffect(() => {
    const updateContextCounts = async () => {
      try {
        await buildContext()
      } catch (error) {
        console.error('Error updating context counts:', error)
      }
    }
    updateContextCounts()
  }, [buildContext])

  return {
    // State
    sessions,
    currentSession: currentSession || sessions.find(s => s.id === currentSessionId),
    currentSessionId,
    messages,
    isSending,
    loadingSessions,
    tokenCount,
    charCount,
    thinkingSteps,
    isThinking,
    isSwitchingSession,

    // Actions
    createSession,
    updateSession,
    deleteSession,
    switchSession,
    sendMessage,
    refetchSessions
  }
}
