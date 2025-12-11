# 連線速度慢 - Server 端問題診斷與優化指南

## 問題描述
客戶反映連線到 `100.77.36.85:8188` 網速非常慢。

## 可能的 Server 端問題

### 1. **Next.js 配置問題** ✅ 已優化
- **問題**：未明確啟用壓縮，可能導致傳輸資料量大
- **解決方案**：已在 `next.config.ts` 中啟用 `compress: true`
- **影響**：可減少 60-80% 的傳輸資料量

### 2. **缺少靜態資源快取** ✅ 已優化
- **問題**：靜態資源（JS、CSS、圖片）沒有適當的快取標頭
- **解決方案**：已添加 `Cache-Control` 標頭，設置長期快取
- **影響**：減少重複請求，提升後續載入速度

### 3. **API 代理延遲**
- **問題**：Next.js 通過 rewrites 代理 API 請求到後端（5055 端口），可能增加延遲
- **檢查方法**：
  ```bash
  # 在 server 上測試內部 API 響應時間
  curl -w "@-" -o /dev/null -s "http://localhost:5055/api/health" <<'EOF'
       time_namelookup:  %{time_namelookup}\n
          time_connect:  %{time_connect}\n
       time_appconnect:  %{time_appconnect}\n
      time_pretransfer:  %{time_pretransfer}\n
         time_redirect:  %{time_redirect}\n
    time_starttransfer:  %{time_starttransfer}\n
                       ----------\n
            time_total:  %{time_total}\n
  EOF
  ```

### 4. **Docker 資源限制**
- **當前配置**：
  - 記憶體限制：4GB
  - CPU 限制：2 核心
  - 檔案描述符：65536
- **檢查方法**：
  ```bash
  # 檢查容器資源使用情況
  docker stats <container_name>
  
  # 檢查是否有 OOM (Out of Memory) 事件
  dmesg | grep -i "out of memory"
  docker logs <container_name> | grep -i "memory\|oom"
  ```

### 5. **網路層問題**
- **檢查方法**：
  ```bash
  # 檢查網路延遲
  ping 100.77.36.85
  
  # 檢查端口是否正常監聽
  netstat -tuln | grep 8188
  ss -tuln | grep 8188
  
  # 檢查防火牆規則
  iptables -L -n | grep 8188
  ```

### 6. **缺少反向代理（Nginx）**
- **問題**：直接暴露 Next.js 服務，沒有使用 Nginx 等反向代理
- **影響**：
  - 無法使用更高效的壓縮算法（如 Brotli）
  - 無法集中管理 SSL/TLS
  - 無法進行負載均衡
- **建議**：考慮在 Next.js 前添加 Nginx 反向代理

## 立即診斷步驟

### 步驟 1：檢查服務狀態
```bash
# 進入容器
docker exec -it <container_name> bash

# 檢查 Next.js 進程
ps aux | grep node

# 檢查端口監聽
netstat -tuln | grep 8188
```

### 步驟 2：測試內部響應速度
```bash
# 在 server 上測試本地連接
time curl -s http://localhost:8188 > /dev/null

# 測試 API 響應
time curl -s http://localhost:5055/api/health
```

### 步驟 3：檢查資源使用
```bash
# 監控容器資源
docker stats <container_name> --no-stream

# 檢查 Node.js 記憶體使用
docker exec <container_name> node -e "console.log(process.memoryUsage())"
```

### 步驟 4：檢查日誌
```bash
# 檢查 Next.js 日誌
docker logs <container_name> --tail 100 | grep -i "frontend\|next"

# 檢查是否有錯誤
docker logs <container_name> --tail 100 | grep -i "error\|warn\|slow"
```

### 步驟 5：網路診斷
```bash
# 從客戶端測試
curl -v -w "\n\nTime: %{time_total}s\n" http://100.77.36.85:8188

# 檢查 DNS 解析時間
dig 100.77.36.85

# 追蹤路由
traceroute 100.77.36.85
```

## 已實施的優化

### 1. Next.js 配置優化
- ✅ 啟用 `compress: true` 確保 gzip 壓縮
- ✅ 移除 `X-Powered-By` 標頭
- ✅ 添加靜態資源快取標頭
- ✅ 設置安全標頭

### 2. 需要重新建置
```bash
# 重新建置 frontend
cd frontend
npm run build

# 重啟容器
docker-compose restart open_notebook
```

## 進一步優化建議

### 1. 添加 Nginx 反向代理（建議）
創建 `nginx.conf`：
```nginx
server {
    listen 80;
    server_name 100.77.36.85;

    # 啟用 gzip 和 brotli 壓縮
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/json application/xml+rss;

    # 靜態資源快取
    location /_next/static/ {
        proxy_pass http://localhost:8188;
        proxy_cache_valid 200 365d;
        add_header Cache-Control "public, max-age=31536000, immutable";
    }

    location / {
        proxy_pass http://localhost:8188;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 2. 監控和日誌
- 設置應用效能監控（APM）
- 記錄慢請求日誌
- 監控 API 響應時間

### 3. CDN 整合
- 將靜態資源放到 CDN
- 使用 CDN 快取 HTML 頁面

### 4. 資料庫優化
- 檢查 SurrealDB 和 Qdrant 查詢效能
- 優化索引
- 檢查連線池設置

## 常見問題排查

### Q: 為什麼客戶端慢但 server 本地測試快？
A: 可能是網路問題，檢查：
- 防火牆規則
- 網路頻寬限制
- 路由問題

### Q: 首次載入慢但後續快？
A: 可能是：
- 首次需要編譯/渲染
- 缺少快取
- 資料庫查詢慢（首次）

### Q: 所有請求都慢？
A: 可能是：
- Server 資源不足
- 資料庫連線問題
- API 後端處理慢

## 聯絡資訊
如有問題，請檢查：
1. Docker 容器日誌
2. Next.js 日誌
3. 系統資源使用情況
4. 網路連線狀態

