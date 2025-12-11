#!/bin/bash
# RAG Notebook 服務重啟腳本
# 用途：快速重啟 RAG Notebook 的 Docker 服務

set -e

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 項目路徑（請根據實際情況修改）
PROJECT_DIR="/home/qiyoo/rag-notebook"

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║    RAG Notebook 服務重啟腳本              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# 檢查項目目錄是否存在
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}❌ 錯誤：找不到項目目錄${NC}"
    echo -e "${RED}   預期路徑: $PROJECT_DIR${NC}"
    echo -e "${YELLOW}   請確認路徑是否正確或修改腳本中的 PROJECT_DIR 變數${NC}"
    exit 1
fi

# 切換到項目目錄
cd "$PROJECT_DIR" || {
    echo -e "${RED}❌ 無法進入項目目錄${NC}"
    exit 1
}

# 檢查 Docker 是否安裝
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker 未安裝${NC}"
    exit 1
fi

# 檢查 Docker Compose 是否可用
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    echo -e "${RED}❌ Docker Compose 未安裝${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} 環境檢查通過"
echo ""

# 顯示當前服務狀態
echo -e "${BLUE}[步驟 1/3]${NC} 檢查當前服務狀態..."
$DOCKER_COMPOSE ps
echo ""

# 重啟服務
echo -e "${BLUE}[步驟 2/3]${NC} 正在重啟服務..."
if $DOCKER_COMPOSE restart; then
    echo -e "${GREEN}✓${NC} 服務重啟成功"
else
    echo -e "${RED}❌ 服務重啟失敗${NC}"
    exit 1
fi
echo ""

# 等待服務就緒
echo -e "${BLUE}[步驟 3/3]${NC} 等待服務就緒..."
sleep 3

# 檢查服務健康狀態
echo ""
echo -e "${BLUE}服務狀態檢查:${NC}"
$DOCKER_COMPOSE ps

echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ RAG Notebook 服務已重啟完成！${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""

# 顯示訪問信息
echo -e "${BLUE}訪問地址:${NC}"
echo -e "  本地前端: ${GREEN}http://localhost:8502${NC}"
echo -e "  本地 API:  ${GREEN}http://localhost:5055${NC}"
echo ""

# 檢查 Tailscale 是否可用（可選）
if command -v tailscale &> /dev/null && tailscale status &> /dev/null; then
    TAILSCALE_DOMAIN=$(tailscale status --self | grep -oP '\S+\.tail[\w-]+\.ts\.net' | head -n1 || echo "")
    if [ -n "$TAILSCALE_DOMAIN" ]; then
        echo -e "${BLUE}Tailscale 訪問:${NC}"
        echo -e "  ${GREEN}https://$TAILSCALE_DOMAIN${NC}"
        echo ""
    fi
fi

echo -e "${YELLOW}常用命令:${NC}"
echo "  查看日誌:     cd $PROJECT_DIR && $DOCKER_COMPOSE logs -f"
echo "  停止服務:     cd $PROJECT_DIR && $DOCKER_COMPOSE down"
echo "  查看狀態:     cd $PROJECT_DIR && $DOCKER_COMPOSE ps"
