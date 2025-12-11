#!/bin/bash
# Tailscale 連線診斷腳本
# 用於診斷 q.java-geological.ts.net 連線問題

set -e

# 顏色輸出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Tailscale 連線診斷工具                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════╝${NC}"
echo ""

TARGET_DOMAIN="q.java-geological.ts.net"

# 1. 檢查 Tailscale 是否安裝
echo -e "${BLUE}[1/8]${NC} 檢查 Tailscale 安裝狀態..."
if command -v tailscale &> /dev/null; then
    echo -e "   ${GREEN}✅ Tailscale 已安裝${NC}"
    TAILSCALE_VERSION=$(tailscale version 2>/dev/null | head -n1 || echo "unknown")
    echo "   版本: $TAILSCALE_VERSION"
else
    echo -e "   ${RED}❌ Tailscale 未安裝${NC}"
    echo "   請安裝: https://tailscale.com/download"
    exit 1
fi
echo ""

# 2. 檢查 Tailscale 連接狀態
echo -e "${BLUE}[2/8]${NC} 檢查 Tailscale 連接狀態..."
if tailscale status &> /dev/null; then
    echo -e "   ${GREEN}✅ Tailscale 已連接${NC}"
    
    # 獲取當前節點信息
    TAILSCALE_IP=$(tailscale ip -4 2>/dev/null | head -n1 || echo "")
    if [ -n "$TAILSCALE_IP" ]; then
        echo "   Tailscale IP: $TAILSCALE_IP"
    fi
    
    # 獲取 MagicDNS 域名
    CURRENT_DOMAIN=$(tailscale status --self 2>/dev/null | grep -oE '[a-zA-Z0-9-]+\.tail[\w-]+\.ts\.net' | head -n1 || echo "")
    if [ -n "$CURRENT_DOMAIN" ]; then
        echo "   當前 MagicDNS: $CURRENT_DOMAIN"
        if [ "$CURRENT_DOMAIN" != "$TARGET_DOMAIN" ]; then
            echo -e "   ${YELLOW}⚠️  當前域名 ($CURRENT_DOMAIN) 與目標域名 ($TARGET_DOMAIN) 不符${NC}"
        fi
    else
        echo -e "   ${YELLOW}⚠️  無法獲取 MagicDNS 域名${NC}"
    fi
else
    echo -e "   ${RED}❌ Tailscale 未連接${NC}"
    echo "   請執行: sudo tailscale up"
    exit 1
fi
echo ""

# 3. 檢查目標域名解析
echo -e "${BLUE}[3/8]${NC} 檢查目標域名解析 ($TARGET_DOMAIN)..."
if command -v dig &> /dev/null; then
    DIG_RESULT=$(dig +short "$TARGET_DOMAIN" 2>/dev/null || echo "")
    if [ -n "$DIG_RESULT" ]; then
        echo -e "   ${GREEN}✅ DNS 解析成功${NC}"
        echo "   解析結果: $DIG_RESULT"
    else
        echo -e "   ${YELLOW}⚠️  DNS 解析失敗或無結果${NC}"
        echo "   可能原因："
        echo "   - DNS 緩存問題"
        echo "   - MagicDNS 未啟用"
        echo "   - 域名不正確"
    fi
elif command -v nslookup &> /dev/null; then
    NSLOOKUP_RESULT=$(nslookup "$TARGET_DOMAIN" 2>/dev/null | grep -A1 "Name:" || echo "")
    if [ -n "$NSLOOKUP_RESULT" ]; then
        echo -e "   ${GREEN}✅ DNS 解析成功${NC}"
        echo "$NSLOOKUP_RESULT"
    else
        echo -e "   ${YELLOW}⚠️  DNS 解析失敗${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠️  無法檢查 DNS（dig/nslookup 未安裝）${NC}"
fi
echo ""

# 4. 檢查 Tailscale Serve 狀態
echo -e "${BLUE}[4/8]${NC} 檢查 Tailscale Serve 配置狀態..."
SERVE_STATUS=$(tailscale serve status 2>/dev/null || echo "")
if [ -n "$SERVE_STATUS" ]; then
    echo -e "   ${GREEN}✅ Tailscale Serve 已配置${NC}"
    echo "   當前配置:"
    echo "$SERVE_STATUS" | sed 's/^/   /'
    
    # 檢查是否配置了正確的端口
    if echo "$SERVE_STATUS" | grep -q "8188\|5055"; then
        echo -e "   ${GREEN}✅ 端口配置正確${NC}"
    else
        echo -e "   ${YELLOW}⚠️  未檢測到預期端口 (8188/5055)${NC}"
    fi
else
    echo -e "   ${RED}❌ Tailscale Serve 未配置${NC}"
    echo "   請執行: sudo ./scripts/setup-tailscale-serve.sh"
fi
echo ""

# 5. 檢查本地服務是否運行
echo -e "${BLUE}[5/8]${NC} 檢查本地服務狀態..."
FRONTEND_PORT=8188
API_PORT=5055

check_port() {
    local port=$1
    local name=$2
    
    if command -v nc &> /dev/null; then
        if nc -z localhost "$port" 2>/dev/null; then
            echo -e "   ${GREEN}✅ $name 正在運行 (端口 $port)${NC}"
            return 0
        else
            echo -e "   ${RED}❌ $name 未運行 (端口 $port)${NC}"
            return 1
        fi
    elif command -v ss &> /dev/null; then
        if ss -lnt | grep -q ":$port "; then
            echo -e "   ${GREEN}✅ $name 正在運行 (端口 $port)${NC}"
            return 0
        else
            echo -e "   ${RED}❌ $name 未運行 (端口 $port)${NC}"
            return 1
        fi
    else
        if timeout 1 bash -c "echo >/dev/tcp/localhost/$port" 2>/dev/null; then
            echo -e "   ${GREEN}✅ $name 正在運行 (端口 $port)${NC}"
            return 0
        else
            echo -e "   ${RED}❌ $name 未運行 (端口 $port)${NC}"
            return 1
        fi
    fi
}

check_port "$FRONTEND_PORT" "前端服務"
check_port "$API_PORT" "API 服務"
echo ""

# 6. 檢查 Docker 容器狀態
echo -e "${BLUE}[6/8]${NC} 檢查 Docker 容器狀態..."
if command -v docker &> /dev/null; then
    if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "open_notebook\|rag-notebook"; then
        echo -e "   ${GREEN}✅ Docker 容器正在運行${NC}"
        docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "open_notebook|rag-notebook|NAMES" | sed 's/^/   /'
    else
        echo -e "   ${YELLOW}⚠️  未找到相關 Docker 容器${NC}"
        echo "   請檢查: docker ps"
    fi
else
    echo -e "   ${YELLOW}⚠️  Docker 未安裝或不可用${NC}"
fi
echo ""

# 7. 測試本地連接
echo -e "${BLUE}[7/8]${NC} 測試本地服務連接..."
if curl -s -f -o /dev/null -w "   響應時間: %{time_total}s\n" "http://localhost:$FRONTEND_PORT" 2>/dev/null; then
    echo -e "   ${GREEN}✅ 本地前端服務可訪問${NC}"
else
    echo -e "   ${RED}❌ 本地前端服務無法訪問${NC}"
fi

if curl -s -f -o /dev/null -w "   響應時間: %{time_total}s\n" "http://localhost:$API_PORT/health" 2>/dev/null; then
    echo -e "   ${GREEN}✅ 本地 API 服務可訪問${NC}"
else
    echo -e "   ${YELLOW}⚠️  本地 API 服務無法訪問或無 /health 端點${NC}"
fi
echo ""

# 8. 測試 Tailscale 域名連接
echo -e "${BLUE}[8/8]${NC} 測試 Tailscale 域名連接..."
if curl -s -f -o /dev/null -w "   響應時間: %{time_total}s\n" "https://$TARGET_DOMAIN" 2>/dev/null; then
    echo -e "   ${GREEN}✅ Tailscale 域名可訪問${NC}"
else
    echo -e "   ${RED}❌ Tailscale 域名無法訪問${NC}"
    echo "   可能原因："
    echo "   1. Tailscale Serve 未正確配置"
    echo "   2. DNS 緩存問題（嘗試清除 DNS 緩存）"
    echo "   3. Tailscale Serve 功能未在 admin console 啟用"
    echo "   4. 防火牆或網路問題"
fi
echo ""

# 總結和建議
echo -e "${BLUE}════════════════════════════════════════════${NC}"
echo -e "${BLUE}診斷總結${NC}"
echo -e "${BLUE}════════════════════════════════════════════${NC}"
echo ""

# 檢查常見問題
ISSUES_FOUND=0

if ! tailscale status &> /dev/null; then
    echo -e "${RED}❌ 問題 1: Tailscale 未連接${NC}"
    echo "   解決方案: sudo tailscale up"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi

if [ -z "$SERVE_STATUS" ]; then
    echo -e "${RED}❌ 問題 2: Tailscale Serve 未配置${NC}"
    echo "   解決方案: sudo ./scripts/setup-tailscale-serve.sh"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi

if ! nc -z localhost "$FRONTEND_PORT" 2>/dev/null && ! ss -lnt 2>/dev/null | grep -q ":$FRONTEND_PORT "; then
    echo -e "${RED}❌ 問題 3: 前端服務未運行${NC}"
    echo "   解決方案: docker-compose up -d 或檢查容器狀態"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi

if [ $ISSUES_FOUND -eq 0 ]; then
    echo -e "${GREEN}✅ 未發現明顯問題${NC}"
    echo ""
    echo "如果仍然無法連接，請嘗試："
    echo "1. 清除 DNS 緩存："
    echo "   - Linux: sudo systemd-resolve --flush-caches 或 sudo resolvectl flush-caches"
    echo "   - Mac: sudo dscacheutil -flushcache; sudo killall -HUP mDNSResponder"
    echo "   - Windows: ipconfig /flushdns"
    echo ""
    echo "2. 重新配置 Tailscale Serve："
    echo "   sudo tailscale serve reset"
    echo "   sudo ./scripts/setup-tailscale-serve.sh"
    echo ""
    echo "3. 檢查 Tailscale Admin Console："
    echo "   - 確認 Serve 功能已啟用"
    echo "   - 確認節點名稱和域名正確"
    echo ""
    echo "4. 檢查防火牆："
    echo "   sudo ufw status"
    echo "   sudo iptables -L -n"
else
    echo ""
    echo -e "${YELLOW}發現 $ISSUES_FOUND 個問題，請先解決上述問題${NC}"
fi

echo ""
echo -e "${BLUE}常用命令：${NC}"
echo "  查看 Tailscale 狀態: tailscale status"
echo "  查看 Serve 狀態: tailscale serve status"
echo "  重置 Serve: sudo tailscale serve reset"
echo "  重新配置 Serve: sudo ./scripts/setup-tailscale-serve.sh"
echo "  查看 Docker 日誌: docker-compose logs -f"
echo ""



