import { NextResponse } from 'next/server'
import type { NextRequest } from 'next/server'

// Paths that don't require authentication
const publicPaths = [
  '/login',
  '/_next',
  '/favicon.ico',
  '/api/auth/status',
  '/api/auth/login',
  '/api/config',
]

// Check if a path is public
function isPublicPath(pathname: string): boolean {
  return publicPaths.some(path => pathname.startsWith(path))
}

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl

  // Redirect root to notebooks
  if (pathname === '/') {
    return NextResponse.redirect(new URL('/notebooks', request.url))
  }

  // Allow public paths without auth check
  if (isPublicPath(pathname)) {
    return NextResponse.next()
  }

  // Check for authentication token
  // The token can be in:
  // 1. Cookie named 'auth-token' (for SSR)
  // 2. The auth-storage localStorage item (but we can't access that in middleware)
  // 
  // Since localStorage is not accessible in middleware, we rely on:
  // - Client-side protection via ConnectionGuard component
  // - API-level protection via backend middleware
  //
  // The middleware here provides a basic check for cookie-based auth
  // For full protection, the client-side code checks localStorage
  
  const authCookie = request.cookies.get('auth-token')
  
  // If we have an auth cookie, let the request through
  // The actual token validation happens on the backend
  if (authCookie?.value) {
    return NextResponse.next()
  }

  // For non-API routes, we let them through and rely on client-side protection
  // This is because:
  // 1. The auth token is stored in localStorage (not accessible in middleware)
  // 2. The client-side AuthGuard/ConnectionGuard handles redirection
  // 3. API requests are protected by the backend middleware
  //
  // We could redirect to login here, but it would cause issues with:
  // - Initial page load when token is in localStorage
  // - Hydration mismatches between server and client
  
  return NextResponse.next()
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes - handled by backend)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     */
    '/((?!api|_next/static|_next/image|favicon.ico).*)',
  ],
}
