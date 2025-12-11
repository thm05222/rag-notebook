# Tailscale 連線問題排查指南

## 問題：無法連線到 q.java-geological.ts.net

### 快速診斷

執行診斷腳本：
```bash
./scripts/diagnose-tailscale-connection.sh
```

## 常見問題和解決方案

### 1. **Tailscale Serve 配置被重置或未配置** ⚠️ 最常見

**症狀**：
- 域名無法訪問
- 返回 502 或連接超時

**解決方案**：
```bash
# 檢查當前 Serve 狀態
sudo tailscale serve status

# 如果未配置或配置錯誤，重新配置
sudo tailscale serve reset
sudo ./scripts/setup-tailscale-serve.sh
```

### 2. **DNS 緩存問題** ⚠️ 很常見

**症狀**：
- 本地可以訪問，但遠端無法訪問
- DNS 解析返回舊的 IP 或無法解析

**解決方案**：

**Linux:**
```bash
# 清除 systemd-resolved 緩存
sudo systemd-resolve --flush-caches
# 或
sudo resolvectl flush-caches

# 清除 nscd 緩存（如果使用）
sudo systemctl restart nscd
```

**Mac:**
```bash
sudo dscacheutil -flushcache
sudo killall -HUP mDNSResponder
```

**Windows:**
```cmd
ipconfig /flushdns
```

**在 Tailscale 節點上（server 端）:**
```bash
# 重啟 Tailscale 服務以刷新 DNS
sudo systemctl restart tailscaled
```

### 3. **本地服務未運行**

**檢查方法**：
```bash
# 檢查端口是否監聽
netstat -tuln | grep 8188
# 或
ss -tuln | grep 8188

# 檢查 Docker 容器
docker ps | grep open_notebook
```

**解決方案**：
```bash
# 重啟 Docker 服務
docker-compose restart open_notebook
# 或
docker-compose up -d
```

### 4. **Tailscale Serve 功能未啟用**

**檢查方法**：
1. 登入 Tailscale Admin Console: https://login.tailscale.com/admin
2. 檢查節點設置
3. 確認 "Serve" 功能已啟用

**解決方案**：
- 在 Admin Console 中啟用 Serve 功能
- 確認節點有足夠的權限

### 5. **節點名稱或域名不匹配**

**檢查方法**：
```bash
# 查看當前節點信息
tailscale status --self

# 查看 MagicDNS 域名
tailscale status | grep -E "\.ts\.net"
```

**解決方案**：
- 確認節點名稱正確
- 如果節點名稱改變，MagicDNS 域名也會改變
- 使用 `tailscale status --self` 查看當前域名

### 6. **防火牆或網路問題**

**檢查方法**：
```bash
# 檢查防火牆規則
sudo ufw status
sudo iptables -L -n | grep 8188

# 測試本地連接
curl -v http://localhost:8188
```

**解決方案**：
```bash
# 如果使用 ufw
sudo ufw allow 8188/tcp

# 檢查 iptables 規則
sudo iptables -L -n -v
```

### 7. **Tailscale 連接狀態問題**

**檢查方法**：
```bash
tailscale status
tailscale ping <other-node>
```

**解決方案**：
```bash
# 重新連接 Tailscale
sudo tailscale down
sudo tailscale up
```

## 完整重置流程

如果以上方法都無法解決，嘗試完整重置：

```bash
# 1. 停止 Tailscale Serve
sudo tailscale serve reset

# 2. 重啟 Tailscale 服務
sudo systemctl restart tailscaled

# 3. 確認 Tailscale 連接
tailscale status

# 4. 確認本地服務運行
curl http://localhost:8188

# 5. 重新配置 Tailscale Serve
sudo ./scripts/setup-tailscale-serve.sh

# 6. 驗證配置
sudo tailscale serve status

# 7. 測試連接
curl https://q.java-geological.ts.net
```

## 驗證步驟

### 步驟 1: 檢查 Tailscale 狀態
```bash
tailscale status
```
應該看到：
- ✅ 節點已連接
- ✅ 顯示 MagicDNS 域名（如 `q.java-geological.ts.net`）

### 步驟 2: 檢查 Serve 配置
```bash
sudo tailscale serve status
```
應該看到：
- ✅ 端口 8188 已配置
- ✅ 映射到 `http://localhost:8188`

### 步驟 3: 檢查本地服務
```bash
curl -I http://localhost:8188
```
應該返回：
- ✅ HTTP 200 或 302 響應

### 步驟 4: 測試 Tailscale 域名
```bash
curl -I https://q.java-geological.ts.net
```
應該返回：
- ✅ HTTP 200 或 302 響應

## 常見錯誤訊息

### "connection refused"
- **原因**：本地服務未運行或端口未監聽
- **解決**：檢查 Docker 容器和服務狀態

### "502 Bad Gateway"
- **原因**：Tailscale Serve 配置了，但本地服務無法訪問
- **解決**：確認 `http://localhost:8188` 可訪問

### "DNS resolution failed"
- **原因**：DNS 緩存問題或 MagicDNS 未啟用
- **解決**：清除 DNS 緩存，確認 MagicDNS 已啟用

### "timeout"
- **原因**：網路問題或防火牆阻擋
- **解決**：檢查網路連接和防火牆規則

## 預防措施

1. **定期檢查 Serve 狀態**：
   ```bash
   sudo tailscale serve status
   ```

2. **監控服務狀態**：
   ```bash
   ./scripts/check-services.sh
   ```

3. **設置自動重啟**（可選）：
   - 使用 systemd timer 定期檢查
   - 或使用監控工具（如 monit）

## 聯絡支援

如果問題持續存在，請提供：
1. 診斷腳本輸出：`./scripts/diagnose-tailscale-connection.sh`
2. Tailscale 狀態：`tailscale status`
3. Serve 狀態：`sudo tailscale serve status`
4. 本地服務測試：`curl -v http://localhost:8188`
5. 錯誤訊息截圖或日誌



