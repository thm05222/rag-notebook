import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Enable standalone output for optimized Docker deployment
  output: "standalone",

  // 啟用壓縮以提升傳輸效能
  // Next.js 15 默認啟用 gzip 壓縮，但明確設置以確保啟用
  compress: true,

  // 生產環境優化
  poweredByHeader: false, // 移除 X-Powered-By 標頭以提升安全性

  // 實驗性功能：設置代理超時時間（毫秒）
  // 這可以防止 Next.js rewrites 在長時間運行的請求中提前關閉連接
  // 默認值通常是 30-60 秒，我們設置為 5 分鐘（300000ms）以支持 Agentic RAG 的長時間執行
  experimental: {
    proxyTimeout: 300000, // 5 分鐘（300 秒）
  },

  // API Rewrites: Proxy /api/* requests to FastAPI backend
  // This simplifies reverse proxy configuration - users only need to proxy to port 8502
  // Next.js handles internal routing to the API backend on port 5055
  async rewrites() {
    // INTERNAL_API_URL: Where Next.js server-side should proxy API requests
    // Default: http://localhost:5055 (single-container deployment)
    // Override for multi-container: INTERNAL_API_URL=http://api-service:5055
    const internalApiUrl = process.env.INTERNAL_API_URL || 'http://localhost:5055'

    console.log(`[Next.js Rewrites] Proxying /api/* to ${internalApiUrl}/api/*`)
    console.log(`[Next.js Rewrites] Proxy timeout: 300000ms (5 minutes)`)

    return [
      {
        source: '/api/:path*',
        destination: `${internalApiUrl}/api/:path*`,
      },
    ]
  },

  // 設置 HTTP 標頭以改善快取和效能
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
        ],
      },
      {
        // 靜態資源快取（圖片、字體等）
        source: '/:path*\\.(jpg|jpeg|png|gif|svg|ico|webp|woff|woff2|ttf|eot)',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
      {
        // Next.js 靜態資源快取
        source: '/_next/static/:path*',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=31536000, immutable',
          },
        ],
      },
    ]
  },
};

export default nextConfig;
