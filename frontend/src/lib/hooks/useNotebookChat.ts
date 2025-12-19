'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { flushSync } from 'react-dom'
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

// 輔助函數：合併消息列表並去重 (修復版)
// 邏輯：以後端消息為準，但保留前端剛產生的消息(如果後端還沒跟上的話)
const mergeMessages = (local: NotebookChatMessage[], remote: NotebookChatMessage[]): NotebookChatMessage[] => {
  if (!remote || remote.length === 0) return local;
  
  // 如果後端消息比前端多(或相等)，通常代表後端是最新的(包含歷史紀錄)，直接使用後端
  // 這避免了 temp-id 和 real-id 同時存在造成的重複問題
  if (remote.length >= local.length) {
    return remote;
  }

  // 如果前端比後端多，可能是剛發送的消息後端還沒存入
  // 這時我們保留後端的所有消息，並把前端多出來的(最新的)消息補在後面
  const remoteIds = new Set(remote.map(m => m.id));
  const uniqueLocal = local.filter(m => !remoteIds.has(m.id));
  
  // 簡單的去重策略：只取後端沒有的，且看起來是新的(在後端最後一條消息之後)
  // 這裡做個簡化：直接回傳 local，因為 local 通常包含了完整的樂觀更新狀態
  return local;
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
  
  const prevSessionIdRef = useRef<string | null>(null)

  // Fetch sessions
  const {
    data: sessions = [],
    isLoading: loadingSessions,
    refetch: refetchSessions
  } = useQuery({
    queryKey: QUERY_KEYS.notebookChatSessions(notebookId),
    queryFn: () => chatApi.listSessions(notebookId),
    enabled: !!notebookId
  })

  // Fetch current session
  const {
    data: currentSession,
    refetch: refetchCurrentSession,
    isLoading: isSessionLoading,
    isFetching: isSessionFetching
  } = useQuery({
    queryKey: QUERY_KEYS.notebookChatSession(currentSessionId!),
    queryFn: () => chatApi.getSession(currentSessionId!),
    enabled: !!notebookId && !!currentSessionId,
    // 保持舊數據直到新數據加載完成，可以減少閃爍，但要小心 ID 判斷
    placeholderData: (previousData) => previousData 
  })

  // 追蹤是否剛完成串流（用於防止 session ID 變更時清空消息）
  const justFinishedStreamingRef = useRef(false)

  // Update messages when current session changes
  useEffect(() => {
    // 1. 串流中絕對不更新
    if (isSending) return

    // 2. 處理 Session 切換
    const sessionIdChanged = prevSessionIdRef.current !== currentSessionId
    if (sessionIdChanged) {
      prevSessionIdRef.current = currentSessionId
      
      // [關鍵修復] 如果是剛完成串流導致的 session ID 變更，不要清空消息
      // 這種情況發生在自動創建 session 時，complete 事件帶回新的 session_id
      if (justFinishedStreamingRef.current) {
        console.log('[useNotebookChat] Session ID changed after streaming, preserving messages')
        justFinishedStreamingRef.current = false
        // 不清空消息，讓後端同步來更新
      } else if (messages.length > 0) {
        // 只有在手動切換 session 時才清空消息
        console.log('[useNotebookChat] Manual session switch, clearing messages')
        setMessages([])
      }
    }

    // 3. 數據有效性檢查
    if (!currentSession || (currentSessionId && currentSession.id !== currentSessionId)) {
      return
    }

    // 4. 數據同步核心邏輯
    const backendMsgs = currentSession.messages || []
    
    setMessages(prev => {
      // [CRITICAL FIX]: 如果本地訊息比後端多，說明後端還沒同步完成
      // 此時強制保留本地狀態，防止訊息「閃退」
      if (prev.length > backendMsgs.length) {
        console.log('[useNotebookChat] Local messages > backend, preserving local:', prev.length, 'vs', backendMsgs.length)
        return prev
      }

      // 如果本地有消息但後端為空，也保留本地（後端可能還沒同步）
      if (prev.length > 0 && backendMsgs.length === 0) {
        console.log('[useNotebookChat] Backend empty but local has messages, preserving local')
        return prev
      }

      // 性能優化：完全相同則不更新
      if (prev.length === backendMsgs.length && prev.length > 0) {
        const lastPrev = prev[prev.length - 1]
        const lastBackend = backendMsgs[backendMsgs.length - 1]
        if (lastPrev?.id === lastBackend?.id && lastPrev?.content === lastBackend?.content) {
          return prev
        }
      }

      // 只有當後端數據 >= 本地數據時，才信任後端並更新
      console.log('[useNotebookChat] Syncing with backend messages:', backendMsgs.length)
      return backendMsgs
    })

    // 清除切換狀態
    if (!isSessionLoading && !isSessionFetching) {
      setIsSwitchingSession(false)
    }
  }, [currentSession, currentSessionId, isSending, isSessionLoading, isSessionFetching])

  // Auto-select most recent session
  useEffect(() => {
    if (sessions.length > 0 && !currentSessionId) {
      const mostRecentSession = sessions[0]
      setCurrentSessionId(mostRecentSession.id)
    }
  }, [sessions, currentSessionId])

  // Mutations (Create/Update/Delete) ... 保持不變 ...
  const createSessionMutation = useMutation({
    mutationFn: (data: CreateNotebookChatSessionRequest) => chatApi.createSession(data),
    onSuccess: (newSession) => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.notebookChatSessions(notebookId) })
      setCurrentSessionId(newSession.id)
      toast.success('Chat session created')
    },
    onError: () => toast.error('Failed to create chat session')
  })

  const updateSessionMutation = useMutation({
    mutationFn: ({ sessionId, data }: { sessionId: string; data: UpdateNotebookChatSessionRequest }) => 
      chatApi.updateSession(sessionId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.notebookChatSessions(notebookId) })
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.notebookChatSession(currentSessionId!) })
      toast.success('Session updated')
    },
    onError: () => toast.error('Failed to update session')
  })

  const deleteSessionMutation = useMutation({
    mutationFn: (sessionId: string) => chatApi.deleteSession(sessionId),
    onSuccess: (_, deletedId) => {
      queryClient.invalidateQueries({ queryKey: QUERY_KEYS.notebookChatSessions(notebookId) })
      if (currentSessionId === deletedId) {
        setCurrentSessionId(null)
        setMessages([])
      }
      toast.success('Session deleted')
    },
    onError: () => toast.error('Failed to delete session')
  })

  // Build Context ... 保持不變 ...
  const buildContext = useCallback(async () => {
    const context_config: { sources: Record<string, string> } = { sources: {} }
    sources.forEach(source => {
      const mode = contextSelections.sources[source.id]
      if (mode === 'insights') context_config.sources[source.id] = 'insights'
      else if (mode === 'full') context_config.sources[source.id] = 'full content'
      else context_config.sources[source.id] = 'not in'
    })
    const response = await chatApi.buildContext({ notebook_id: notebookId, context_config })
    setTokenCount(response.token_count)
    setCharCount(response.char_count)
    return response.context
  }, [notebookId, sources, contextSelections])

  // Helper: Convert Stream Event ... 保持不變 ...
  const convertToThinkingStep = useCallback((event: ChatStreamEvent): AgentThinkingStep | null => {
    if (event.type !== 'thought' || !event.content) return null
    let stepType: AgentThinkingStep['step_type'] = 'decision'
    if (event.stage === 'executing') stepType = 'tool_call'
    else if (event.stage === 'planning') stepType = 'decision'
    else if (event.stage === 'synthesizing') stepType = 'synthesis'
    return {
      step_type: stepType,
      timestamp: event.timestamp || Date.now() / 1000,
      content: event.content,
      metadata: event.metadata || {}
    }
  }, [])

  // Send Message
  const sendMessage = useCallback(async (message: string, modelOverride?: string) => {
    let sessionId = currentSessionId

    // Auto-create session logic
    if (!sessionId) {
      try {
        const defaultTitle = message.length > 30 ? `${message.substring(0, 30)}...` : message
        const newSession = await chatApi.createSession({ notebook_id: notebookId, title: defaultTitle })
        sessionId = newSession.id
        setCurrentSessionId(sessionId)
        queryClient.invalidateQueries({ queryKey: QUERY_KEYS.notebookChatSessions(notebookId) })
      } catch {
        toast.error('Failed to create chat session')
        return
      }
    }

    setThinkingSteps([])
    setIsThinking(false)

    // Optimistic Update
    const userMessage: NotebookChatMessage = {
      id: `temp-${Date.now()}`,
      type: 'human',
      content: message,
      timestamp: new Date().toISOString()
    }
    setMessages(prev => [...prev, userMessage])
    setIsSending(true)

    try {
      const context = await buildContext()
      const response = await chatApi.sendMessage({
        session_id: sessionId,
        message,
        context,
        model_override: modelOverride ?? (currentSession?.model_override ?? undefined),
        notebook_id: notebookId
      })

      if (!response) throw new Error('No response body received')

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
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const jsonStr = line.slice(6).trim()
              if (!jsonStr || jsonStr === '[DONE]') continue
              const data: ChatStreamEvent = JSON.parse(jsonStr)
              
              // Debug: Log received events
              console.log('[SSE] Received event:', data.type, data)

              if (data.type === 'thought') {
                console.log('[SSE] Processing thought event:', data.content?.substring(0, 50))
                const step = convertToThinkingStep(data)
                if (step) {
                  step.timestamp = startTime + (Date.now() / 1000 - startTime)
                  // [關鍵修復] 使用 flushSync 強制 React 立即同步更新狀態
                  // 這樣 thinkingSteps 會在每次收到 thought 事件時立即更新 UI
                  // 而不是被 React 批次處理延遲到 stream 結束後
                  flushSync(() => {
                    setIsThinking(true)
                    setThinkingSteps(prev => {
                      console.log('[SSE] Adding thinking step (sync), new count:', prev.length + 1)
                      return [...prev, step]
                    })
                  })
                }
              } else if (data.type === 'content') {
                setIsThinking(false)
                if (!aiMessage) {
                  aiMessage = {
                    id: `ai-${Date.now()}`,
                    type: 'ai',
                    content: data.content || '',
                    timestamp: new Date().toISOString()
                  }
                  setMessages(prev => [...prev, aiMessage!])
                } else {
                  aiMessage.content += data.content || ''
                  setMessages(prev => prev.map(msg => msg.id === aiMessage!.id ? { ...msg, content: aiMessage!.content } : msg))
                }
              } else if (data.type === 'error') {
                throw new Error(data.message || 'Stream error occurred')
              } else if (data.type === 'complete') {
                setIsThinking(false)
                // [關鍵修復]：標記剛完成串流，防止 session ID 變更時清空消息
                justFinishedStreamingRef.current = true
                
                // [新增] 強制將 final_answer 加入前端狀態
                const finalAnswer = data.final_answer
                const thinkingProcess = data.thinking_process
                if (finalAnswer) {
                  setMessages(prev => {
                    // 檢查是否已經有這條訊息 (避免重複)
                    const lastMsg = prev[prev.length - 1];
                    if (lastMsg && lastMsg.type === 'ai' && lastMsg.content === finalAnswer) {
                      console.log('[useNotebookChat] Final answer already exists, skipping')
                      // 即使內容相同，也要更新 thinking_process
                      if (thinkingProcess) {
                        return prev.map(msg => 
                          msg.id === lastMsg.id 
                            ? { ...msg, thinking_process: thinkingProcess } 
                            : msg
                        );
                      }
                      return prev;
                    }
                    
                    // 如果最後一條是正在生成的 AI 訊息，更新它
                    if (aiMessage) {
                      console.log('[useNotebookChat] Updating existing AI message with final answer')
                      return prev.map(msg => 
                        msg.id === aiMessage!.id 
                          ? { ...msg, content: finalAnswer, thinking_process: thinkingProcess } 
                          : msg
                      );
                    }
                    
                    // 否則，新增一條
                    console.log('[useNotebookChat] Adding new AI message with final answer')
                    return [...prev, {
                      id: `ai-complete-${Date.now()}`,
                      type: 'ai' as const,
                      content: finalAnswer,
                      timestamp: new Date().toISOString(),
                      thinking_process: thinkingProcess
                    }];
                  });
                } else if (thinkingProcess) {
                  // 即使沒有 final_answer，也要更新最後一條 AI 訊息的 thinking_process
                  setMessages(prev => {
                    const lastMsg = prev[prev.length - 1];
                    if (lastMsg && lastMsg.type === 'ai') {
                      console.log('[useNotebookChat] Updating last AI message with thinking_process')
                      return prev.map(msg => 
                        msg.id === lastMsg.id 
                          ? { ...msg, thinking_process: thinkingProcess } 
                          : msg
                      );
                    }
                    return prev;
                  });
                }
                
                // 只更新 ID，不覆蓋訊息
                // 因為後端可能還沒寫入完成，這裡去 fetch 往往會拿到空列表，導致畫面洗白
                if (data.session_id && data.session_id !== sessionId) {
                  console.log('[useNotebookChat] Session ID changed in complete event:', sessionId, '->', data.session_id)
                  setCurrentSessionId(data.session_id)
                  queryClient.invalidateQueries({ queryKey: QUERY_KEYS.notebookChatSessions(notebookId) })
                }
              }
            } catch (e) { console.error('Error parsing SSE data:', e) }
          }
        }
      }

      // Stream Finished
      console.log('[useNotebookChat] Stream finished, messages count:', messages.length)
      justFinishedStreamingRef.current = true
      setIsSending(false)
      setIsThinking(false)

      // 延遲同步後端數據
      // [關鍵]：這時 useEffect 會觸發，但我們前面加了長度檢查，所以不會被舊資料覆蓋
      // 延長到 1.5秒 比較保險，給後端足夠時間寫入 DB
      setTimeout(async () => {
        try {
          await refetchCurrentSession()
          // refetch 完成後，currentSession 更新，useEffect 會再次執行
          // 這次後端資料應該已經包含新消息，長度會 >= 本地長度，於是會正確更新成後端資料(替換掉 temp ID)
          // 重置 flag，允許正常的 session 切換行為
          justFinishedStreamingRef.current = false
        } catch (error) {
          console.error('Failed to sync session:', error)
          justFinishedStreamingRef.current = false
        }
      }, 1500)

    } catch (error) {
      console.error('Error sending message:', error)
      let errorMessage = 'Failed to send message'
      if (error instanceof Error) errorMessage = error.message
      toast.error(errorMessage)
      
      // Rollback
      setMessages(prev => prev.filter(msg => !msg.id.startsWith('temp-')))
      setThinkingSteps([])
      setIsThinking(false)
      setIsSending(false)
    }
  }, [notebookId, currentSessionId, currentSession, buildContext, refetchCurrentSession, queryClient, convertToThinkingStep])

  // Switch Session
  const switchSession = useCallback((sessionId: string) => {
    setIsSwitchingSession(true)
    setCurrentSessionId(sessionId)
    setMessages([]) // 立即清空，防止殘影，這是對的
  }, [])

  // Action Wrappers ... 保持不變 ...
  const createSession = useCallback((title?: string) => createSessionMutation.mutate({ notebook_id: notebookId, title }), [createSessionMutation, notebookId])
  const updateSession = useCallback((sessionId: string, data: UpdateNotebookChatSessionRequest) => updateSessionMutation.mutate({ sessionId, data }), [updateSessionMutation])
  const deleteSession = useCallback((sessionId: string) => deleteSessionMutation.mutate(sessionId), [deleteSessionMutation])

  useEffect(() => {
    const updateContextCounts = async () => {
      try { await buildContext() } catch (error) { console.error('Error updating context counts:', error) }
    }
    updateContextCounts()
  }, [buildContext])

  return {
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
    createSession,
    updateSession,
    deleteSession,
    switchSession,
    sendMessage,
    refetchSessions
  }
}
