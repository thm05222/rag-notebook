import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { getApiUrl } from '@/lib/config'
import type { AuthStatusResponse, LoginResponse } from '@/lib/types/auth'

interface AuthState {
  isAuthenticated: boolean
  token: string | null
  isLoading: boolean
  error: string | null
  lastAuthCheck: number | null
  isCheckingAuth: boolean
  hasHydrated: boolean
  authRequired: boolean | null
  authMode: 'jwt' | 'legacy' | 'none'
  turnstileEnabled: boolean
  turnstileSiteKey: string
  setHasHydrated: (state: boolean) => void
  checkAuthRequired: () => Promise<boolean>
  login: (username: string, password: string, captchaToken?: string) => Promise<boolean>
  logout: () => void
  checkAuth: () => Promise<boolean>
  clearError: () => void
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      isAuthenticated: false,
      token: null,
      isLoading: false,
      error: null,
      lastAuthCheck: null,
      isCheckingAuth: false,
      hasHydrated: false,
      authRequired: null,
      authMode: 'none',
      turnstileEnabled: false,
      turnstileSiteKey: '',

      setHasHydrated: (state: boolean) => {
        set({ hasHydrated: state })
      },

      clearError: () => {
        set({ error: null })
      },

      checkAuthRequired: async () => {
        try {
          const apiUrl = await getApiUrl()
          // Use relative path if apiUrl is empty (Tailscale) or localhost:5055 (Next.js rewrites will proxy)
          const authStatusUrl = (apiUrl === '' || apiUrl === 'http://localhost:5055') 
            ? '/api/auth/status' 
            : `${apiUrl}/api/auth/status`
          const response = await fetch(authStatusUrl, {
            cache: 'no-store',
          })

          if (!response.ok) {
            throw new Error(`Auth status check failed: ${response.status}`)
          }

          const data: AuthStatusResponse = await response.json()
          const required = data.auth_enabled || false
          
          set({ 
            authRequired: required,
            authMode: data.auth_mode || 'none',
            turnstileEnabled: data.turnstile_enabled || false,
            turnstileSiteKey: data.turnstile_site_key || '',
          })

          // If auth is not required, mark as authenticated
          if (!required) {
            set({ isAuthenticated: true, token: 'not-required' })
          }

          return required
        } catch (error) {
          console.error('Failed to check auth status:', error)

          // If it's a network error, set a more helpful error message
          if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
            set({
              error: 'Unable to connect to server. Please check if the API is running.',
              authRequired: null  // Don't assume auth is required if we can't connect
            })
          } else {
            // For other errors, default to requiring auth to be safe
            set({ authRequired: true })
          }

          // Re-throw the error so the UI can handle it
          throw error
        }
      },

      login: async (username: string, password: string, captchaToken?: string) => {
        set({ isLoading: true, error: null })
        
        const { authMode } = get()
        
        try {
          const apiUrl = await getApiUrl()
          
          if (authMode === 'jwt') {
            // JWT authentication - call /api/auth/login
            const loginUrl = (apiUrl === '' || apiUrl === 'http://localhost:5055') 
              ? '/api/auth/login' 
              : `${apiUrl}/api/auth/login`
            
            const response = await fetch(loginUrl, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                username,
                password,
                captcha_token: captchaToken || '',
              }),
            })
            
            if (response.ok) {
              const data: LoginResponse = await response.json()
              set({ 
                isAuthenticated: true, 
                token: data.access_token, 
                isLoading: false,
                lastAuthCheck: Date.now(),
                error: null
              })
              return true
            } else {
              let errorMessage = 'Authentication failed'
              
              try {
                const errorData = await response.json()
                if (errorData.detail) {
                  // Map specific error messages to user-friendly ones
                  if (errorData.detail.includes('Captcha')) {
                    errorMessage = 'Captcha verification failed. Please try again.'
                  } else if (errorData.detail.includes('username or password')) {
                    errorMessage = 'Invalid username or password.'
                  } else if (errorData.detail.includes('not configured')) {
                    errorMessage = 'Authentication service is not properly configured.'
                  } else {
                    errorMessage = errorData.detail
                  }
                }
              } catch {
                // Use status code based messages
                if (response.status === 401) {
                  errorMessage = 'Invalid username or password.'
                } else if (response.status === 400) {
                  errorMessage = 'Invalid request. Please check your input.'
                } else if (response.status >= 500) {
                  errorMessage = 'Server error. Please try again later.'
                }
              }
              
              set({ 
                error: errorMessage,
                isLoading: false,
                isAuthenticated: false,
                token: null
              })
              return false
            }
          } else {
            // Legacy password authentication - test with notebooks endpoint
            const notebooksUrl = (apiUrl === '' || apiUrl === 'http://localhost:5055') 
              ? '/api/notebooks' 
              : `${apiUrl}/api/notebooks`

            const response = await fetch(notebooksUrl, {
              method: 'GET',
              headers: {
                'Authorization': `Bearer ${password}`,
                'Content-Type': 'application/json'
              }
            })
            
            if (response.ok) {
              set({ 
                isAuthenticated: true, 
                token: password, 
                isLoading: false,
                lastAuthCheck: Date.now(),
                error: null
              })
              return true
            } else {
              let errorMessage = 'Authentication failed'
              if (response.status === 401) {
                errorMessage = 'Invalid password. Please try again.'
              } else if (response.status === 403) {
                errorMessage = 'Access denied. Please check your credentials.'
              } else if (response.status >= 500) {
                errorMessage = 'Server error. Please try again later.'
              } else {
                errorMessage = `Authentication failed (${response.status})`
              }
              
              set({ 
                error: errorMessage,
                isLoading: false,
                isAuthenticated: false,
                token: null
              })
              return false
            }
          }
        } catch (error) {
          console.error('Network error during auth:', error)
          let errorMessage = 'Authentication failed'
          
          if (error instanceof TypeError && error.message.includes('Failed to fetch')) {
            errorMessage = 'Unable to connect to server. Please check if the API is running.'
          } else if (error instanceof Error) {
            errorMessage = `Network error: ${error.message}`
          } else {
            errorMessage = 'An unexpected error occurred during authentication'
          }
          
          set({ 
            error: errorMessage,
            isLoading: false,
            isAuthenticated: false,
            token: null
          })
          return false
        }
      },
      
      logout: () => {
        set({ 
          isAuthenticated: false, 
          token: null, 
          error: null,
          lastAuthCheck: null,
        })
      },
      
      checkAuth: async () => {
        const state = get()
        const { token, lastAuthCheck, isCheckingAuth, isAuthenticated, authMode } = state

        // If already checking, return current auth state
        if (isCheckingAuth) {
          return isAuthenticated
        }

        // If no token, not authenticated
        if (!token) {
          return false
        }

        // If we checked recently (within 30 seconds) and are authenticated, skip
        const now = Date.now()
        if (isAuthenticated && lastAuthCheck && (now - lastAuthCheck) < 30000) {
          return true
        }

        set({ isCheckingAuth: true })

        try {
          const apiUrl = await getApiUrl()
          const notebooksUrl = (apiUrl === '' || apiUrl === 'http://localhost:5055') 
            ? '/api/notebooks' 
            : `${apiUrl}/api/notebooks`

          const response = await fetch(notebooksUrl, {
            method: 'GET',
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json'
            }
          })
          
          if (response.ok) {
            set({ 
              isAuthenticated: true, 
              lastAuthCheck: now,
              isCheckingAuth: false 
            })
            return true
          } else {
            set({
              isAuthenticated: false,
              token: null,
              lastAuthCheck: null,
              isCheckingAuth: false
            })
            return false
          }
        } catch (error) {
          console.error('checkAuth error:', error)
          set({ 
            isAuthenticated: false, 
            token: null,
            lastAuthCheck: null,
            isCheckingAuth: false 
          })
          return false
        }
      }
    }),
    {
      name: 'auth-storage',
      partialize: (state) => ({
        token: state.token,
        isAuthenticated: state.isAuthenticated,
        authMode: state.authMode,
      }),
      onRehydrateStorage: () => (state) => {
        state?.setHasHydrated(true)
      }
    }
  )
)
