'use client'

import { useState, useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'
import Turnstile, { BoundTurnstileObject } from 'react-turnstile'
import { useAuthStore } from '@/lib/stores/auth-store'
import { getConfig } from '@/lib/config'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { AlertCircle, Loader2 } from 'lucide-react'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'

export function LoginForm() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [captchaToken, setCaptchaToken] = useState<string | null>(null)
  const [captchaError, setCaptchaError] = useState<string | null>(null)
  const turnstileRef = useRef<BoundTurnstileObject | null>(null)
  
  const { 
    login, 
    isLoading, 
    error, 
    authRequired, 
    authMode,
    turnstileEnabled,
    checkAuthRequired, 
    hasHydrated, 
    isAuthenticated,
    clearError
  } = useAuthStore()
  
  const [isCheckingAuth, setIsCheckingAuth] = useState(true)
  const [configInfo, setConfigInfo] = useState<{ apiUrl: string; version: string; buildTime: string } | null>(null)
  const router = useRouter()

  // Get Turnstile site key from environment
  const turnstileSiteKey = process.env.NEXT_PUBLIC_TURNSTILE_SITE_KEY || ''

  // Load config info for debugging
  useEffect(() => {
    getConfig().then(cfg => {
      setConfigInfo({
        apiUrl: cfg.apiUrl,
        version: cfg.version,
        buildTime: cfg.buildTime,
      })
    }).catch(err => {
      console.error('Failed to load config:', err)
    })
  }, [])

  // Check if authentication is required on mount
  useEffect(() => {
    if (!hasHydrated) {
      return
    }

    const checkAuth = async () => {
      try {
        const required = await checkAuthRequired()

        // If auth is not required, redirect to notebooks
        if (!required) {
          router.push('/notebooks')
        }
      } catch (error) {
        console.error('Error checking auth requirement:', error)
        // On error, assume auth is required to be safe
      } finally {
        setIsCheckingAuth(false)
      }
    }

    // If we already know auth status, use it
    if (authRequired !== null) {
      if (!authRequired && isAuthenticated) {
        router.push('/notebooks')
      } else {
        setIsCheckingAuth(false)
      }
    } else {
      void checkAuth()
    }
  }, [hasHydrated, authRequired, checkAuthRequired, router, isAuthenticated])

  // Clear error when form changes
  useEffect(() => {
    if (error) {
      clearError()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [username, password])

  // Show loading while checking if auth is required
  if (!hasHydrated || isCheckingAuth) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <LoadingSpinner />
      </div>
    )
  }

  // If we still don't know if auth is required (connection error), show error
  if (authRequired === null) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background p-4">
        <Card className="w-full max-w-md">
          <CardHeader className="text-center">
            <CardTitle>Connection Error</CardTitle>
            <CardDescription>
              Unable to connect to the API server
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-start gap-2 text-red-600 text-sm">
                <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                <div className="flex-1">
                  {error || 'Unable to connect to server. Please check if the API is running.'}
                </div>
              </div>

              {configInfo && (
                <div className="space-y-2 text-xs text-muted-foreground border-t pt-3">
                  <div className="font-medium">Diagnostic Information:</div>
                  <div className="space-y-1 font-mono">
                    <div>Version: {configInfo.version}</div>
                    <div>Built: {new Date(configInfo.buildTime).toLocaleString()}</div>
                    <div className="break-all">API URL: {configInfo.apiUrl}</div>
                    <div className="break-all">Frontend: {typeof window !== 'undefined' ? window.location.href : 'N/A'}</div>
                  </div>
                  <div className="text-xs pt-2">
                    Check browser console for detailed logs (look for 🔧 [Config] messages)
                  </div>
                </div>
              )}

              <Button
                onClick={() => window.location.reload()}
                className="w-full"
              >
                Retry Connection
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    // Validate form
    if (authMode === 'jwt') {
      if (!username.trim() || !password.trim()) {
        return
      }
      
      // Check captcha if required
      if (turnstileEnabled && !captchaToken) {
        setCaptchaError('Please complete the captcha verification')
        return
      }
    } else {
      // Legacy mode only needs password
      if (!password.trim()) {
        return
      }
    }
    
    setCaptchaError(null)
    
    try {
      const success = await login(
        authMode === 'jwt' ? username : 'admin',
        password,
        captchaToken || undefined
      )
      
      if (success) {
        router.push('/notebooks')
      } else {
        // Reset captcha on failure
        if (turnstileRef.current) {
          turnstileRef.current.reset()
          setCaptchaToken(null)
        }
      }
    } catch (error) {
      console.error('Unhandled error during login:', error)
      // Reset captcha on error
      if (turnstileRef.current) {
        turnstileRef.current.reset()
        setCaptchaToken(null)
      }
    }
  }

  const handleCaptchaVerify = (token: string, boundTurnstile: BoundTurnstileObject) => {
    setCaptchaToken(token)
    setCaptchaError(null)
    turnstileRef.current = boundTurnstile
  }

  const handleCaptchaError = (_error: Error | undefined, boundTurnstile?: BoundTurnstileObject) => {
    setCaptchaToken(null)
    setCaptchaError('Captcha verification failed. Please try again.')
    if (boundTurnstile) {
      turnstileRef.current = boundTurnstile
    }
  }

  const handleCaptchaExpire = (_token: string, boundTurnstile: BoundTurnstileObject) => {
    setCaptchaToken(null)
    turnstileRef.current = boundTurnstile
  }

  // Determine if submit button should be disabled
  const isSubmitDisabled = () => {
    if (isLoading) return true
    
    if (authMode === 'jwt') {
      if (!username.trim() || !password.trim()) return true
      if (turnstileEnabled && !captchaToken) return true
    } else {
      if (!password.trim()) return true
    }
    
    return false
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-background p-4">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <CardTitle>RAG Notebook</CardTitle>
          <CardDescription>
            {authMode === 'jwt' 
              ? 'Sign in to access the application'
              : 'Enter your password to access the application'
            }
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Username field - only for JWT mode */}
            {authMode === 'jwt' && (
              <div className="space-y-2">
                <Label htmlFor="username">Username</Label>
                <Input
                  id="username"
                  type="text"
                  placeholder="Enter username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  disabled={isLoading}
                  autoComplete="username"
                />
              </div>
            )}

            {/* Password field */}
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                placeholder="Enter password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                disabled={isLoading}
                autoComplete="current-password"
              />
            </div>

            {/* Turnstile captcha - only if enabled */}
            {authMode === 'jwt' && turnstileEnabled && turnstileSiteKey && (
              <div className="space-y-2">
                <div className="flex justify-center">
                  <Turnstile
                    sitekey={turnstileSiteKey}
                    onVerify={handleCaptchaVerify}
                    onError={handleCaptchaError}
                    onExpire={handleCaptchaExpire}
                    theme="auto"
                  />
                </div>
                {captchaError && (
                  <div className="flex items-center justify-center gap-2 text-red-600 text-sm">
                    <AlertCircle className="h-4 w-4" />
                    {captchaError}
                  </div>
                )}
              </div>
            )}

            {/* Error message */}
            {error && (
              <div className="flex items-center gap-2 text-red-600 text-sm">
                <AlertCircle className="h-4 w-4" />
                {error}
              </div>
            )}

            {/* Submit button */}
            <Button
              type="submit"
              className="w-full"
              disabled={isSubmitDisabled()}
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Signing in...
                </>
              ) : (
                'Sign In'
              )}
            </Button>

            {/* Version info */}
            {configInfo && (
              <div className="text-xs text-center text-muted-foreground pt-2 border-t">
                <div>Version {configInfo.version}</div>
                <div className="font-mono text-[10px]">{configInfo.apiUrl}</div>
              </div>
            )}
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
