#!/bin/bash
# Nginx + UFW + SSL 部署腳本
# 為 rag-notebook 配置反向代理和 HTTPS

set -e  # 遇到錯誤時停止

echo "=========================================="
echo "  rag-notebook Nginx + SSL 部署腳本"
echo "=========================================="

# 確認域名
DOMAIN="${1:-rag.polaris-x.com}"
echo "域名: $DOMAIN"
echo ""

# ========================================
# 第一步：UFW 防火牆配置
# ========================================
echo "[1/5] 配置 UFW 防火牆..."

# 設定預設策略
sudo ufw default deny incoming
sudo ufw default allow outgoing

# 開放公網必要端口
sudo ufw allow 22/tcp comment 'SSH'
sudo ufw allow 80/tcp comment 'HTTP'
sudo ufw allow 443/tcp comment 'HTTPS'

# 開放 Tailscale 全部訪問
if ip addr show tailscale0 &>/dev/null; then
    sudo ufw allow in on tailscale0 comment 'Tailscale'
    echo "✓ Tailscale 介面已配置"
else
    echo "⚠ 未找到 Tailscale 介面 (tailscale0)"
fi

# 啟用防火牆（非互動模式）
echo "y" | sudo ufw enable

echo "✓ UFW 防火牆配置完成"
sudo ufw status numbered
echo ""

# ========================================
# 第二步：安裝 Nginx
# ========================================
echo "[2/5] 安裝 Nginx..."

sudo apt update
sudo apt install nginx -y

echo "✓ Nginx 安裝完成"
echo ""

# ========================================
# 第三步：創建 Nginx 配置
# ========================================
echo "[3/5] 創建 Nginx 反向代理配置..."

# 創建配置文件
sudo tee /etc/nginx/sites-available/rag > /dev/null << 'NGINX_CONFIG'
server {
    listen 80;
    server_name DOMAIN_PLACEHOLDER;

    # 前端轉發 (port 8188)
    location / {
        proxy_pass http://localhost:8188;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket 支持
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # 後端 API 轉發 (port 5055)
    location /api/ {
        proxy_pass http://localhost:5055;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE (Server-Sent Events) 串流設定
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
        proxy_buffering off;
        proxy_cache off;
        
        # 增加超時時間（用於長時間的 AI 生成）
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # 健康檢查端點
    location /health {
        proxy_pass http://localhost:5055/health;
        proxy_set_header Host $host;
    }
    
    # 文件上傳限制
    client_max_body_size 50M;
}
NGINX_CONFIG

# 替換域名佔位符
sudo sed -i "s/DOMAIN_PLACEHOLDER/$DOMAIN/g" /etc/nginx/sites-available/rag

# 啟用配置
sudo ln -sf /etc/nginx/sites-available/rag /etc/nginx/sites-enabled/

# 移除預設配置（如果存在）
sudo rm -f /etc/nginx/sites-enabled/default

# 測試配置
sudo nginx -t

# 重新載入 Nginx
sudo systemctl reload nginx

echo "✓ Nginx 配置完成"
echo ""

# ========================================
# 第四步：安裝 SSL 憑證
# ========================================
echo "[4/5] 安裝 Let's Encrypt SSL 憑證..."

sudo apt install certbot python3-certbot-nginx -y

echo ""
echo "正在為 $DOMAIN 申請 SSL 憑證..."
echo "請按照提示操作（輸入郵箱、同意條款、選擇重定向）"
echo ""

sudo certbot --nginx -d "$DOMAIN"

echo "✓ SSL 憑證安裝完成"
echo ""

# ========================================
# 第五步：驗證配置
# ========================================
echo "[5/5] 驗證配置..."

echo ""
echo "=========================================="
echo "  部署完成！"
echo "=========================================="
echo ""
echo "請進行以下測試："
echo ""
echo "1. 公網 HTTPS 訪問："
echo "   https://$DOMAIN"
echo ""
echo "2. 安全性測試（用沒有 Tailscale 的設備）："
echo "   curl -m 5 http://$(curl -s ifconfig.me):5055/health"
echo "   預期：超時或拒絕連接"
echo ""
echo "3. Tailscale 直連測試："
TAILSCALE_IP=$(ip -4 addr show tailscale0 | grep -oP '(?<=inet\s)\d+(\.\d+){3}' || echo "N/A")
echo "   http://$TAILSCALE_IP:5055/health"
echo "   http://$TAILSCALE_IP:8188"
echo ""
echo "4. 查看 UFW 狀態："
echo "   sudo ufw status"
echo ""
echo "5. 查看 Nginx 狀態："
echo "   sudo systemctl status nginx"
echo ""



