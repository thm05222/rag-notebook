#!/bin/bash
# 快速修復 Tailscale Serve 配置（使用新語法）

echo "快速修復 Tailscale Serve 配置..."
echo ""

# 重置現有配置
echo "1. 重置現有配置..."
sudo tailscale serve reset
echo ""

# 配置 HTTPS (端口 443) - 只配置 HTTPS，更安全
echo "2. 配置 HTTPS 映射（端口 443）..."
# 直接使用端口號，避免路徑映射問題
# Tailscale Serve 會自動處理 HTTPS 證書
sudo tailscale serve --bg --https 443 8188
echo ""
echo "注意：只配置了 HTTPS（推薦）。如果需要 HTTP，可以額外執行："
echo "  sudo tailscale serve --bg --http 80 8188"
echo ""

# 顯示配置狀態
echo "4. 當前配置狀態："
sudo tailscale serve status
echo ""

echo "✅ 配置完成！"
echo ""
echo "現在應該可以使用以下地址訪問（不需要端口號）："
echo "  https://q.java-geological.ts.net"
echo "  http://q.java-geological.ts.net"
echo ""

