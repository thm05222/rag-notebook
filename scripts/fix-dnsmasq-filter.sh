#!/bin/bash
# 修正 dnsmasq filter-AAAA 配置

set -e

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

DNSMASQ_CONF="/etc/dnsmasq.conf"
BACKUP_FILE="/etc/dnsmasq.conf.backup.$(date +%Y%m%d_%H%M%S)"

echo -e "${BLUE}=== 修正 dnsmasq filter-AAAA 配置 ===${NC}"
echo ""

# 檢查是否為 root 或使用 sudo
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}[警告]${NC} 需要 root 權限，將使用 sudo"
    SUDO_CMD="sudo"
else
    SUDO_CMD=""
fi

# 1. 備份原始配置
echo -e "${BLUE}[1/5]${NC} 備份原始配置..."
$SUDO_CMD cp "$DNSMASQ_CONF" "$BACKUP_FILE"
echo -e "   ${GREEN}✅${NC} 備份至: $BACKUP_FILE"
echo ""

# 2. 移除錯誤的 filter-AAAA 配置
echo -e "${BLUE}[2/5]${NC} 移除錯誤的 filter-AAAA 配置..."
if grep -q "^filter-AAAA=/polaris-x.com/" "$DNSMASQ_CONF"; then
    # 移除錯誤的配置行
    $SUDO_CMD sed -i '/^filter-AAAA=\/polaris-x.com\//d' "$DNSMASQ_CONF"
    # 也移除相關的註釋（如果有的話）
    $SUDO_CMD sed -i '/^# 過濾 polaris-x.com 的 IPv6 記錄/d' "$DNSMASQ_CONF"
    echo -e "   ${GREEN}✅${NC} 已移除錯誤的配置"
else
    echo -e "   ${YELLOW}⚠️${NC} 未找到錯誤的配置"
fi
echo ""

# 3. 檢查並添加正確的配置
echo -e "${BLUE}[3/5]${NC} 檢查並添加正確的配置..."

# 檢查是否已有全域 filter-AAAA
if grep -q "^filter-AAAA$" "$DNSMASQ_CONF"; then
    echo -e "   ${YELLOW}⚠️${NC} 全域 filter-AAAA 已存在"
else
    # 在 address=/polaris-x.com/ 後面添加註釋和 filter-AAAA
    if grep -q "^address=/polaris-x.com/" "$DNSMASQ_CONF"; then
        # 在 address 行後面添加正確的配置
        $SUDO_CMD sed -i "/^address=\/polaris-x.com\//a # 過濾所有 AAAA (IPv6) 記錄，只返回 IPv4\n# 注意：這是全域設定，會影響所有域名的 IPv6 解析\nfilter-AAAA" "$DNSMASQ_CONF"
        echo -e "   ${GREEN}✅${NC} 已添加正確的 filter-AAAA 配置"
    else
        echo -e "   ${RED}❌${NC} 未找到 address=/polaris-x.com/ 配置"
        echo "   請先配置 address=/polaris-x.com/ 映射"
        exit 1
    fi
fi
echo ""

# 4. 驗證配置語法
echo -e "${BLUE}[4/5]${NC} 驗證配置語法..."
if $SUDO_CMD dnsmasq --test 2>/dev/null; then
    echo -e "   ${GREEN}✅${NC} 配置語法正確"
else
    echo -e "   ${RED}❌${NC} 配置語法錯誤"
    echo "   錯誤詳情："
    $SUDO_CMD dnsmasq --test 2>&1 | head -5
    echo -e "   ${YELLOW}恢復備份...${NC}"
    $SUDO_CMD cp "$BACKUP_FILE" "$DNSMASQ_CONF"
    exit 1
fi
echo ""

# 5. 重啟服務
echo -e "${BLUE}[5/5]${NC} 重啟 dnsmasq 服務..."
if $SUDO_CMD systemctl restart dnsmasq; then
    echo -e "   ${GREEN}✅${NC} dnsmasq 服務已重啟"
    
    # 等待服務啟動
    sleep 2
    
    # 檢查服務狀態
    if $SUDO_CMD systemctl is-active --quiet dnsmasq; then
        echo -e "   ${GREEN}✅${NC} dnsmasq 服務運行正常"
    else
        echo -e "   ${RED}❌${NC} dnsmasq 服務啟動失敗"
        echo -e "   ${YELLOW}恢復備份...${NC}"
        $SUDO_CMD cp "$BACKUP_FILE" "$DNSMASQ_CONF"
        $SUDO_CMD systemctl restart dnsmasq
        exit 1
    fi
else
    echo -e "   ${RED}❌${NC} 無法重啟服務"
    exit 1
fi
echo ""

# 6. 測試 DNS 解析
echo -e "${BLUE}[測試]${NC} 測試 DNS 解析..."
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null | head -n1 || echo "100.77.36.85")

# 測試 IPv4 解析
if dig @127.0.0.1 +short polaris-x.com A 2>/dev/null | grep -q "$TAILSCALE_IP"; then
    echo -e "   ${GREEN}✅${NC} IPv4 解析成功: polaris-x.com -> $TAILSCALE_IP"
else
    echo -e "   ${YELLOW}⚠️${NC} IPv4 解析測試失敗"
fi

# 測試 IPv6 是否被過濾
AAAA_RESULT=$(dig @127.0.0.1 +short polaris-x.com AAAA 2>/dev/null || echo "")
if [ -z "$AAAA_RESULT" ]; then
    echo -e "   ${GREEN}✅${NC} IPv6 記錄已被過濾（符合預期）"
else
    echo -e "   ${YELLOW}⚠️${NC} IPv6 記錄未被過濾: $AAAA_RESULT"
fi

echo ""
echo -e "${GREEN}=== 配置完成 ===${NC}"
echo ""
echo "配置摘要:"
echo "  - 已修正: filter-AAAA（全域設定）"
echo "  - 效果: 所有域名的 IPv6 記錄將被過濾，只返回 IPv4"
echo "  - 注意: 這是全域設定，會影響所有域名"
echo ""
echo "現在在 Windows 上測試應該只會看到 IPv4 地址："
echo "  nslookup polaris-x.com 100.77.36.85"
echo ""

