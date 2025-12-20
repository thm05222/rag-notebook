export interface NotebookResponse {
  id: string
  name: string
  description: string
  archived: boolean
  created: string
  updated: string
  source_count: number
}

export interface SourceListResponse {
  id: string
  title: string | null
  topics?: string[]                  // Make optional to match Python API
  asset: {
    file_path?: string
    url?: string
  } | null
  embedded: boolean
  embedded_chunks: number            // ADD: From Python API
  insights_count: number
  created: string
  updated: string
  file_available?: boolean
  // ADD: Async processing fields from Python API
  command_id?: string
  status?: string
  processing_info?: Record<string, unknown>
  // Processing status and error tracking
  processing_status?: string
  error_message?: string
  // PageIndex fields
  has_pageindex?: boolean
  pageindex_built_at?: string | null
}

export interface SourceDetailResponse extends SourceListResponse {
  full_text: string
  notebooks?: string[]  // List of notebook IDs this source is linked to
  // PageIndex fields are inherited from SourceListResponse
}

export type SourceResponse = SourceDetailResponse

export interface SourceStatusResponse {
  status?: string
  message: string
  processing_info?: Record<string, unknown>
  command_id?: string
}

export interface SettingsResponse {
  default_content_processing_engine_doc?: string
  default_content_processing_engine_url?: string
  default_embedding_option?: string
  auto_delete_files?: string
  youtube_preferred_languages?: string[]
  default_context_mode?: string
}

export interface CreateNotebookRequest {
  name: string
  description?: string
}

export interface UpdateNotebookRequest {
  name?: string
  description?: string
  archived?: boolean
}

export interface CreateSourceRequest {
  // Backward compatibility: support old single notebook_id
  notebook_id?: string
  // New multi-notebook support
  notebooks?: string[]
  // Required fields
  type: 'link' | 'upload' | 'text'
  url?: string
  file_path?: string
  content?: string
  title?: string
  transformations?: string[]
  embed?: boolean
  build_pageindex?: boolean
  delete_source?: boolean
  // New async processing support
  async_processing?: boolean
}

export interface UpdateSourceRequest {
  title?: string
  type?: 'link' | 'upload' | 'text'
  url?: string
  content?: string
}

export interface APIError {
  detail: string
}

// Source Chat Types
// Base session interface with common fields
export interface BaseChatSession {
  id: string
  title: string
  created: string
  updated: string
  message_count?: number
  model_override?: string | null
}

export interface SourceChatSession extends BaseChatSession {
  source_id: string
  model_override?: string
}

export interface TokenUsage {
  input_tokens?: number
  output_tokens?: number
  total_tokens?: number
}

export interface AgentThinkingStep {
  step_type: 'decision' | 'tool_call' | 'search' | 'evaluation' | 'refinement' | 'synthesis'
  timestamp: number
  content: string
  metadata?: Record<string, unknown> & {
    token_usage?: TokenUsage
  }
}

export interface AgentThinkingProcess {
  steps: AgentThinkingStep[]
  total_iterations: number
  total_tool_calls: number
  search_count: number
  evaluation_scores?: Record<string, number>
  reasoning_trace: string[]
}

export interface SourceChatMessage {
  id: string
  type: 'human' | 'ai'
  content: string
  timestamp?: string
  thinking_process?: AgentThinkingProcess
}

export interface SourceChatContextIndicator {
  sources: string[]
  insights: string[]
}

export interface SourceChatSessionWithMessages extends SourceChatSession {
  messages: SourceChatMessage[]
  context_indicators?: SourceChatContextIndicator
}

export interface CreateSourceChatSessionRequest {
  source_id: string
  title?: string
  model_override?: string
}

export interface UpdateSourceChatSessionRequest {
  title?: string
  model_override?: string
}

export interface SendMessageRequest {
  message: string
  model_override?: string
}

export interface SourceChatStreamEvent {
  type: 'user_message' | 'ai_message' | 'context_indicators' | 'complete' | 'error'
  content?: string
  data?: unknown
  message?: string
  timestamp?: string
}

// Notebook Chat Types
export interface NotebookChatSession extends BaseChatSession {
  notebook_id: string
}

export interface NotebookChatMessage {
  id: string
  type: 'human' | 'ai'
  content: string
  timestamp?: string
  thinking_process?: AgentThinkingProcess
}

export interface NotebookChatSessionWithMessages extends NotebookChatSession {
  messages: NotebookChatMessage[]
}

export interface CreateNotebookChatSessionRequest {
  notebook_id: string
  title?: string
  model_override?: string
}

export interface UpdateNotebookChatSessionRequest {
  title?: string
  model_override?: string | null
}

export interface SendNotebookChatMessageRequest {
  session_id: string
  message: string
  context: {
    sources: Array<Record<string, unknown>>
  }
  model_override?: string
  notebook_id?: string  // Optional: for auto-creating session if not found
}

export interface BuildContextRequest {
  notebook_id: string
  context_config: {
    sources: Record<string, string>
  }
}

export interface BuildContextResponse {
  context: {
    sources: Array<Record<string, unknown>>
  }
  token_count: number
  char_count: number
}

export interface ChatStreamEvent {
  type: 'thought' | 'content' | 'error' | 'complete'
  stage?: 'planning' | 'executing' | 'synthesizing'
  content?: string
  metadata?: Record<string, unknown>
  message?: string
  timestamp?: number
  final_answer?: string
  session_id?: string  // Optional: new session ID if session was auto-created
  thinking_process?: AgentThinkingProcess  // Optional: thinking process from backend
}
