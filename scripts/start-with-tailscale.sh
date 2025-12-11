#!/bin/bash
# 啟動 Docker 服務並配置 Tailscale Serve 的完整腳本

set -e

# 顏色輸出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  RAG Notebook + Tailscale Serve 啟動  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# 檢查 Tailscale
if ! command -v tailscale &> /dev/null; then
    echo -e "${RED}❌ Tailscale 未安裝${NC}"
    echo "請先安裝 Tailscale: https://tailscale.com/download"
    exit 1
fi

# 檢查 Tailscale 連接狀態
if ! tailscale status &> /dev/null; then
    echo -e "${YELLOW}⚠️  Tailscale 未連接${NC}"
    echo "請先連接 Tailscale 或執行: sudo tailscale up"
    exit 1
fi

# 檢查 Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker 未安裝${NC}"
    exit 1
fi

# 檢查 Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo -e "${RED}❌ Docker Compose 未安裝${NC}"
    exit 1
fi

# 獲取 Docker Compose 命令
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

echo -e "${GREEN}✓${NC} 環境檢查通過"
echo ""

# 步驟 1: 啟動 Docker 服務
echo -e "${BLUE}[步驟 1/3]${NC} 啟動 Docker 服務..."
$DOCKER_COMPOSE -f docker-compose.dev.yml up -d --build

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Docker 服務啟動失敗${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Docker 服務已啟動"
echo ""

# 步驟 2: 等待服務就緒
echo -e "${BLUE}[步驟 2/3]${NC} 等待服務就緒..."
sleep 5

MAX_WAIT=60
WAITED=0
FRONTEND_PORT=${TAILSCALE_FRONTEND_PORT:-8188}

echo -n "等待前端服務"
while [ $WAITED -lt $MAX_WAIT ]; do
    if nc -z localhost $FRONTEND_PORT 2>/dev/null; then
        echo ""
        echo -e "${GREEN}✓${NC} 前端服務已就緒"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    echo -n "."
done

if ! nc -z localhost $FRONTEND_PORT 2>/dev/null; then
    echo ""
    echo -e "${YELLOW}⚠️  前端服務啟動超時，但繼續配置 Tailscale Serve${NC}"
fi

echo ""

# 步驟 3: 配置 Tailscale Serve
echo -e "${BLUE}[步驟 3/3]${NC} 配置 Tailscale Serve..."
./scripts/setup-tailscale-serve.sh

echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ 所有服務已啟動並配置完成！${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""

# 顯示訪問信息
TAILSCALE_DOMAIN=$(tailscale status --self | grep -oP '\S+\.tail[\w-]+\.ts\.net' | head -n1 || echo "")
TAILSCALE_IP=$(tailscale ip -4 | head -n1)

echo -e "${BLUE}訪問地址:${NC}"
if [ -n "$TAILSCALE_DOMAIN" ]; then
    echo -e "  ${GREEN}https://$TAILSCALE_DOMAIN${NC}"
else
    echo -e "  ${GREEN}https://$TAILSCALE_IP${NC}"
fi

echo ""
echo -e "${BLUE}本地訪問:${NC}"
echo -e "  前端: ${GREEN}http://localhost:8188${NC}"
echo -e "  API:  ${GREEN}http://localhost:5055${NC}"

echo ""
echo -e "${YELLOW}常用命令:${NC}"
echo "  查看服務狀態: $DOCKER_COMPOSE -f docker-compose.dev.yml ps"
echo "  查看日誌:     $DOCKER_COMPOSE -f docker-compose.dev.yml logs -f"
echo "  停止服務:     $DOCKER_COMPOSE -f docker-compose.dev.yml down"
echo "  停止 Tailscale Serve: ./scripts/stop-tailscale-serve.sh"

