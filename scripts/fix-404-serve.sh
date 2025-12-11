#!/bin/bash
# 修復 Tailscale Serve 404 錯誤

echo "修復 Tailscale Serve 404 錯誤..."
echo ""

# 檢查是否需要 sudo
if [ "$EUID" -ne 0 ]; then
    SUDO_CMD="sudo"
    echo "需要 root 權限，將使用 sudo"
else
    SUDO_CMD=""
fi

# 1. 重置配置
echo "1. 重置現有配置..."
$SUDO_CMD tailscale serve reset
echo ""

# 2. 檢查本地服務
echo "2. 檢查本地服務..."
if curl -s -f http://localhost:8188 > /dev/null 2>&1; then
    echo "   ✅ 本地服務運行正常"
else
    echo "   ❌ 本地服務無法訪問，請先確保服務運行"
    exit 1
fi
echo ""

# 3. 配置 HTTPS (使用簡化語法，讓 Tailscale 自動處理路徑)
echo "3. 配置 HTTPS 映射（端口 443）..."
# 直接使用端口號（Tailscale 會自動映射到根路徑），避免 404 錯誤
# 只配置 HTTPS（推薦，更安全），Tailscale 會自動處理 HTTPS 證書
$SUDO_CMD tailscale serve --bg --https 443 8188
echo ""
echo "注意：只配置了 HTTPS。如果需要 HTTP，可以額外執行："
echo "  sudo tailscale serve --bg --http 80 8188"
echo ""

# 5. 顯示配置狀態
echo "5. 當前配置狀態："
$SUDO_CMD tailscale serve status
echo ""

echo "✅ 配置完成！"
echo ""
echo "請測試以下地址："
echo "  https://q.java-geological.ts.net"
echo "  http://q.java-geological.ts.net"
echo ""
echo "如果仍然出現 404，請嘗試："
echo "  1. 清除瀏覽器緩存"
echo "  2. 等待 1-2 分鐘讓配置生效"
echo "  3. 檢查 tailscale serve status 確認路徑映射正確"
echo ""

