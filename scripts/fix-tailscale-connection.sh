#!/bin/bash
# 快速修復 Tailscale 連線問題腳本

set -e

# 顏色輸出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Tailscale 連線快速修復工具                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

# 檢查是否為 root
if [ "$EUID" -ne 0 ]; then
    echo -e "${YELLOW}⚠️  需要 root 權限，將使用 sudo${NC}"
    SUDO_CMD="sudo"
else
    SUDO_CMD=""
fi

# 步驟 1: 檢查 Tailscale 連接
echo -e "${BLUE}[步驟 1/5]${NC} 檢查 Tailscale 連接狀態..."
if ! tailscale status &> /dev/null; then
    echo -e "   ${RED}❌ Tailscale 未連接${NC}"
    echo "   正在嘗試連接..."
    $SUDO_CMD tailscale up
    sleep 2
    if tailscale status &> /dev/null; then
        echo -e "   ${GREEN}✅ Tailscale 已連接${NC}"
    else
        echo -e "   ${RED}❌ 無法連接 Tailscale，請手動執行: sudo tailscale up${NC}"
        exit 1
    fi
else
    echo -e "   ${GREEN}✅ Tailscale 已連接${NC}"
fi

# 顯示當前域名
CURRENT_DOMAIN=$(tailscale status --self 2>/dev/null | grep -oE '[a-zA-Z0-9-]+\.tail[\w-]+\.ts\.net' | head -n1 || echo "")
if [ -n "$CURRENT_DOMAIN" ]; then
    echo "   當前 MagicDNS: $CURRENT_DOMAIN"
fi
echo ""

# 步驟 2: 檢查本地服務
echo -e "${BLUE}[步驟 2/5]${NC} 檢查本地服務狀態..."
FRONTEND_PORT=8188

check_port() {
    local port=$1
    if command -v nc &> /dev/null; then
        nc -z localhost "$port" 2>/dev/null
    elif command -v ss &> /dev/null; then
        ss -lnt | grep -q ":$port "
    else
        timeout 1 bash -c "echo >/dev/tcp/localhost/$port" 2>/dev/null
    fi
}

if check_port "$FRONTEND_PORT"; then
    echo -e "   ${GREEN}✅ 前端服務正在運行 (端口 $FRONTEND_PORT)${NC}"
else
    echo -e "   ${YELLOW}⚠️  前端服務未運行 (端口 $FRONTEND_PORT)${NC}"
    echo "   正在檢查 Docker 容器..."
    
    if command -v docker &> /dev/null; then
        if docker ps --format "{{.Names}}" | grep -q "open_notebook"; then
            echo "   發現 Docker 容器，嘗試重啟..."
            cd "$(dirname "$0")/.." || exit 1
            if [ -f "docker-compose.yml" ]; then
                docker-compose restart open_notebook 2>/dev/null || docker compose restart open_notebook 2>/dev/null || true
                echo "   等待服務啟動..."
                sleep 5
            fi
        fi
    fi
    
    # 再次檢查
    if check_port "$FRONTEND_PORT"; then
        echo -e "   ${GREEN}✅ 前端服務已啟動${NC}"
    else
        echo -e "   ${RED}❌ 前端服務仍無法訪問${NC}"
        echo "   請手動檢查: docker-compose logs open_notebook"
    fi
fi
echo ""

# 步驟 3: 檢查當前 Serve 配置
echo -e "${BLUE}[步驟 3/5]${NC} 檢查 Tailscale Serve 配置..."
SERVE_STATUS=$($SUDO_CMD tailscale serve status 2>/dev/null || echo "")
if [ -z "$SERVE_STATUS" ] || ! echo "$SERVE_STATUS" | grep -q "$FRONTEND_PORT"; then
    echo -e "   ${YELLOW}⚠️  Serve 配置不正確或未配置${NC}"
    echo "   將重新配置..."
else
    echo -e "   ${GREEN}✅ Serve 配置正確${NC}"
    echo "   當前配置:"
    echo "$SERVE_STATUS" | sed 's/^/   /'
    echo ""
    read -p "是否仍要重新配置? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "跳過重新配置"
        SKIP_RECONFIGURE=true
    fi
fi
echo ""

# 步驟 4: 重新配置 Serve（如果需要）
if [ "$SKIP_RECONFIGURE" != "true" ]; then
    echo -e "${BLUE}[步驟 4/5]${NC} 重新配置 Tailscale Serve..."
    
    # 重置現有配置
    echo "   重置現有配置..."
    $SUDO_CMD tailscale serve reset >/dev/null 2>&1 || true
    sleep 1
    
    # 配置新的 Serve
    echo "   配置前端服務 (端口 $FRONTEND_PORT)..."
    # 使用新的 Tailscale Serve 語法 (v1.90+)
    # 直接使用端口號，Tailscale 會自動映射到根路徑，避免 404 錯誤
    # --https 443: 映射到 HTTPS 標準端口 443（推薦，更安全）
    # 只配置 HTTPS，因為 Tailscale Serve 會自動處理 HTTPS 證書
    if $SUDO_CMD tailscale serve --bg --https 443 $FRONTEND_PORT; then
        echo -e "   ${GREEN}✅ Serve 配置成功（HTTPS）${NC}"
    else
        echo -e "   ${RED}❌ Serve 配置失敗${NC}"
        echo "   請確認："
        echo "   1. Tailscale Serve 功能已在 Admin Console 啟用"
        echo "   2. 有足夠的權限執行 tailscale 命令"
        exit 1
    fi
    
    # 顯示配置結果
    echo ""
    echo "   當前 Serve 狀態:"
    $SUDO_CMD tailscale serve status | sed 's/^/   /'
else
    echo -e "${BLUE}[步驟 4/5]${NC} 跳過重新配置"
fi
echo ""

# 步驟 5: 清除 DNS 緩存
echo -e "${BLUE}[步驟 5/5]${NC} 清除 DNS 緩存..."
if command -v systemd-resolve &> /dev/null; then
    $SUDO_CMD systemd-resolve --flush-caches 2>/dev/null && echo -e "   ${GREEN}✅ 已清除 systemd-resolved 緩存${NC}" || true
elif command -v resolvectl &> /dev/null; then
    $SUDO_CMD resolvectl flush-caches 2>/dev/null && echo -e "   ${GREEN}✅ 已清除 resolvectl 緩存${NC}" || true
fi

# 重啟 Tailscale 服務以刷新 DNS
if systemctl is-active --quiet tailscaled 2>/dev/null; then
    echo "   重啟 Tailscale 服務以刷新 DNS..."
    $SUDO_CMD systemctl restart tailscaled 2>/dev/null && sleep 2 && echo -e "   ${GREEN}✅ Tailscale 服務已重啟${NC}" || echo -e "   ${YELLOW}⚠️  無法重啟 Tailscale 服務（可能需要手動重啟）${NC}"
fi
echo ""

# 最終驗證
echo -e "${BLUE}════════════════════════════════════════════${NC}"
echo -e "${BLUE}修復完成！驗證結果${NC}"
echo -e "${BLUE}════════════════════════════════════════════${NC}"
echo ""

# 顯示訪問信息
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null | head -n1 || echo "")
if [ -n "$TAILSCALE_IP" ]; then
    echo -e "${GREEN}Tailscale IP:${NC} https://$TAILSCALE_IP"
fi

if [ -n "$CURRENT_DOMAIN" ]; then
    echo -e "${GREEN}MagicDNS:${NC} https://$CURRENT_DOMAIN"
    echo ""
    echo "測試連接:"
    echo "  curl -I https://$CURRENT_DOMAIN"
fi

echo ""
echo -e "${YELLOW}注意事項：${NC}"
echo "1. 如果仍然無法連接，請清除客戶端的 DNS 緩存"
echo "2. 確認 Tailscale Serve 功能已在 Admin Console 啟用"
echo "3. 等待 1-2 分鐘讓 DNS 傳播"
echo ""
echo -e "${BLUE}如需詳細診斷，請執行:${NC}"
echo "  ./scripts/diagnose-tailscale-connection.sh"
echo ""

