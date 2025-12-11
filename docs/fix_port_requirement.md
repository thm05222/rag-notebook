# 修復需要加端口號才能訪問的問題

## 問題描述
訪問 `q.java-geological.ts.net` 時需要加端口號：`http://q.java-geological.ts.net:8188`

正常情況下應該可以直接訪問：`https://q.java-geological.ts.net`（不需要端口號）

## 原因
Tailscale Serve 配置使用了簡化語法 `tailscale serve --bg 8188`，這不會自動映射到標準 HTTP/HTTPS 端口（80/443）。

## 解決方案

### 方法 1：使用修復腳本（推薦）
```bash
sudo ./scripts/fix-tailscale-connection.sh
```

### 方法 2：手動修復（推薦：只使用 HTTPS）
```bash
# 1. 重置現有配置
sudo tailscale serve reset

# 2. 配置 HTTPS 映射（端口 443）- 推薦，更安全
# 直接使用端口號，Tailscale 會自動映射到根路徑，避免 404 錯誤
# Tailscale 會自動處理 HTTPS 證書
sudo tailscale serve --bg --https 443 8188

# 3. 驗證配置
sudo tailscale serve status
```

### 方法 2b：如果需要同時支持 HTTP 和 HTTPS
```bash
# 1. 重置現有配置
sudo tailscale serve reset

# 2. 配置 HTTP 映射（端口 80）- 可選
sudo tailscale serve --bg --http 80 8188

# 3. 配置 HTTPS 映射（端口 443）- 推薦
sudo tailscale serve --bg --https 443 8188

# 4. 驗證配置
sudo tailscale serve status
```

### 方法 3：重新運行設置腳本
```bash
sudo ./scripts/setup-tailscale-serve.sh
```

## 驗證

修復後，應該可以：
- ✅ 使用 `https://q.java-geological.ts.net` 訪問（不需要端口號，推薦）
- ✅ 如果配置了 HTTP，也可以使用 `http://q.java-geological.ts.net` 訪問（不需要端口號）

## 命令說明

### 舊的錯誤語法
```bash
tailscale serve --bg 8188
```
這只會將服務映射到非標準端口，需要加端口號才能訪問。

### 新的正確語法（Tailscale v1.90+）

**推薦：只使用 HTTPS（更安全）**
```bash
tailscale serve --bg --https 443 8188
```
這會將服務映射到 HTTPS 標準端口 443，可以直接用 `https://域名` 訪問。

**如果需要同時支持 HTTP 和 HTTPS：**
```bash
tailscale serve --bg --http 80 8188
tailscale serve --bg --https 443 8188
```

**重要說明**：
- **只配置 HTTPS 就夠了**：Tailscale Serve 會自動處理 HTTPS 證書，不需要額外配置
- 直接使用端口號（如 `8188`）而不是完整 URL（如 `http://localhost:8188`）
- 這樣可以避免路徑映射問題，防止出現 404 錯誤
- Tailscale 會自動將端口號映射到根路徑 `/`
- HTTP（端口 80）是可選的，通常只需要 HTTPS（端口 443）

**注意**：Tailscale Serve 的 CLI 語法在 v1.90+ 版本中已改變，需要使用 `--http` 和 `--https` 參數來指定端口。

## 注意事項

1. **需要 root 權限**：配置 Tailscale Serve 需要 sudo 權限
2. **Serve 功能必須啟用**：在 Tailscale Admin Console 中確認 Serve 功能已啟用
3. **本地服務必須運行**：確保 `http://localhost:8188` 可以訪問
4. **DNS 緩存**：修復後可能需要清除 DNS 緩存或等待幾分鐘

## 清除 DNS 緩存

### Linux
```bash
sudo systemd-resolve --flush-caches
# 或
sudo resolvectl flush-caches
```

### Mac
```bash
sudo dscacheutil -flushcache
sudo killall -HUP mDNSResponder
```

### Windows
```cmd
ipconfig /flushdns
```

## 檢查配置狀態

```bash
# 查看 Serve 狀態
sudo tailscale serve status

# 應該看到類似：
# / http://localhost:8188
# / https://localhost:8188
```

## 如果仍然無法訪問

1. 檢查本地服務：
   ```bash
   curl http://localhost:8188
   ```

2. 檢查 Tailscale 連接：
   ```bash
   tailscale status
   ```

3. 檢查防火牆：
   ```bash
   sudo ufw status
   ```

4. 查看詳細診斷：
   ```bash
   ./scripts/diagnose-tailscale-connection.sh
   ```

