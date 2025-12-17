export interface Model {
  id: string
  name: string
  provider: string
  type: 'language' | 'embedding'
  created: string
  updated: string
}

export interface CreateModelRequest {
  name: string
  provider: string
  type: 'language' | 'embedding'
}

export interface ModelDefaults {
  default_chat_model?: string | null
  default_transformation_model?: string | null
  large_context_model?: string | null
  default_embedding_model?: string | null
  default_tools_model?: string | null
  role_default_models?: {
    orchestrator?: string | null
    executor?: string | null
    refiner?: string | null
  } | null
}

export interface ProviderAvailability {
  available: string[]
  unavailable: string[]
  supported_types: Record<string, string[]>
}