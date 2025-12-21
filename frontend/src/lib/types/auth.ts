export interface AuthState {
  isAuthenticated: boolean
  token: string | null
  isLoading: boolean
  error: string | null
}

export interface LoginCredentials {
  username: string
  password: string
  captcha_token?: string
}

export interface LoginRequest {
  username: string
  password: string
  captcha_token: string
}

export interface LoginResponse {
  access_token: string
  token_type: string
  expires_in: number
}

export interface AuthStatusResponse {
  auth_enabled: boolean
  auth_mode: 'jwt' | 'legacy' | 'none'
  turnstile_enabled: boolean
  message: string
}

export interface JWTPayload {
  sub: string
  exp: number
  iat?: number
}
