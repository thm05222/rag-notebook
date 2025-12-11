#!/bin/bash
# 服務狀態快速檢查腳本

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== 服務狀態檢查 ===${NC}"
echo ""

# 1. 檢查 Docker 服務
echo -e "${BLUE}1. Docker 服務狀態：${NC}"
if systemctl is-active --quiet docker 2>/dev/null; then
    echo -e "   ${GREEN}✅ Docker 正在運行${NC}"
else
    echo -e "   ${RED}❌ Docker 未運行${NC}"
    echo -e "   ${YELLOW}   提示: 執行 'sudo systemctl start docker' 啟動${NC}"
fi
echo ""

# 2. 檢查 Docker 容器狀態
echo -e "${BLUE}2. Docker 容器狀態：${NC}"
if [ -d "/home/qiyoo/rag-notebook" ]; then
    cd /home/qiyoo/rag-notebook
    if command -v docker &> /dev/null && docker ps &> /dev/null; then
        CONTAINER_STATUS=$(docker compose ps 2>/dev/null | grep -c "Up" || echo "0")
        if [ "$CONTAINER_STATUS" -gt 0 ]; then
            echo -e "   ${GREEN}✅ 有容器正在運行${NC}"
            docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null | grep -v "^NAME" || true
        else
            echo -e "   ${YELLOW}⚠️  沒有容器在運行${NC}"
            echo -e "   ${YELLOW}   提示: 執行 'docker compose up -d' 啟動${NC}"
        fi
    else
        echo -e "   ${RED}❌ 無法檢查容器狀態（Docker 未運行或無權限）${NC}"
    fi
else
    echo -e "   ${RED}❌ 專案目錄不存在${NC}"
fi
echo ""

# 3. 檢查 API 服務
echo -e "${BLUE}3. API 服務狀態：${NC}"
if command -v curl &> /dev/null; then
    if curl -s -f http://localhost:5055/health > /dev/null 2>&1; then
        echo -e "   ${GREEN}✅ API 服務正常運行（端口 5055）${NC}"
        # 嘗試獲取健康檢查信息
        HEALTH_RESPONSE=$(curl -s http://localhost:5055/health 2>/dev/null)
        if [ -n "$HEALTH_RESPONSE" ]; then
            echo -e "   ${GREEN}   健康狀態: 正常${NC}"
        fi
    else
        echo -e "   ${RED}❌ API 服務無響應（端口 5055）${NC}"
        echo -e "   ${YELLOW}   提示: 檢查 Docker 容器是否運行${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠️  無法測試（curl 未安裝）${NC}"
fi
echo ""

# 4. 檢查前端服務
echo -e "${BLUE}4. 前端服務狀態：${NC}"
if command -v curl &> /dev/null; then
    if curl -s -f http://localhost:8188 > /dev/null 2>&1; then
        echo -e "   ${GREEN}✅ 前端服務正常運行（端口 8188）${NC}"
    else
        echo -e "   ${YELLOW}⚠️  前端服務無響應（端口 8188）${NC}"
        echo -e "   ${YELLOW}   提示: 檢查 Docker 容器是否運行${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠️  無法測試（curl 未安裝）${NC}"
fi
echo ""

# 5. 檢查 Tailscale 服務
echo -e "${BLUE}5. Tailscale 服務狀態：${NC}"
if systemctl is-active --quiet tailscaled 2>/dev/null; then
    echo -e "   ${GREEN}✅ Tailscale 服務正在運行${NC}"
    
    # 檢查連接狀態
    if command -v tailscale &> /dev/null; then
        if tailscale status &> /dev/null; then
            echo -e "   ${GREEN}✅ Tailscale 已連接${NC}"
            # 顯示基本信息
            TAILSCALE_IP=$(tailscale ip -4 2>/dev/null | head -n1)
            if [ -n "$TAILSCALE_IP" ]; then
                echo -e "   ${BLUE}   Tailscale IP: ${TAILSCALE_IP}${NC}"
            fi
            TAILSCALE_DOMAIN=$(tailscale status --self 2>/dev/null | grep -oP '\S+\.tail[\w-]+\.ts\.net' | head -n1)
            if [ -n "$TAILSCALE_DOMAIN" ]; then
                echo -e "   ${BLUE}   MagicDNS: ${TAILSCALE_DOMAIN}${NC}"
            fi
        else
            echo -e "   ${YELLOW}⚠️  Tailscale 服務運行但未連接${NC}"
            echo -e "   ${YELLOW}   提示: 執行 'sudo tailscale up' 連接${NC}"
        fi
    fi
else
    echo -e "   ${RED}❌ Tailscale 服務未運行${NC}"
    echo -e "   ${YELLOW}   提示: 執行 'sudo systemctl start tailscaled' 啟動${NC}"
fi
echo ""

# 6. 檢查 Tailscale Serve
echo -e "${BLUE}6. Tailscale Serve 狀態：${NC}"
if command -v tailscale &> /dev/null && tailscale status &> /dev/null; then
    SERVE_STATUS=$(tailscale serve status 2>/dev/null)
    if [ -n "$SERVE_STATUS" ] && echo "$SERVE_STATUS" | grep -q "http"; then
        echo -e "   ${GREEN}✅ Tailscale Serve 已配置${NC}"
        echo -e "   ${BLUE}   配置信息:${NC}"
        echo "$SERVE_STATUS" | sed 's/^/      /'
    else
        echo -e "   ${YELLOW}⚠️  Tailscale Serve 未配置${NC}"
        echo -e "   ${YELLOW}   提示: 執行 './scripts/setup-tailscale-serve.sh' 配置${NC}"
    fi
else
    echo -e "   ${YELLOW}⚠️  無法檢查（Tailscale 未連接）${NC}"
fi
echo ""

# 總結
echo -e "${BLUE}=== 檢查完成 ===${NC}"
echo ""
echo -e "${YELLOW}提示：${NC}"
echo "  - 如果服務未運行，請參考 '服務管理操作手冊.md' 進行操作"
echo "  - 查看詳細日誌: docker compose logs <服務名>"
echo ""



