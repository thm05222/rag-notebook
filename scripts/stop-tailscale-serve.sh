#!/bin/bash
# 停止 Tailscale Serve 配置腳本

set -e

# 顏色輸出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== 停止 Tailscale Serve ===${NC}"
echo ""

# 檢查是否需要 sudo
if tailscale serve status &> /dev/null; then
    USE_SUDO=""
else
    USE_SUDO="sudo"
fi

# 顯示當前狀態
echo "當前 Tailscale Serve 狀態:"
tailscale serve status 2>/dev/null || echo "  尚未配置"
echo ""

# 確認操作
read -p "確定要停止 Tailscale Serve 嗎? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 停止 Serve
echo -e "${YELLOW}正在停止 Tailscale Serve...${NC}"
$USE_SUDO tailscale serve reset

echo ""
echo -e "${GREEN}✅ Tailscale Serve 已停止${NC}"

