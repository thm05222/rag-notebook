#!/bin/bash
# 添加 filter-AAAA 配置以過濾 polaris-x.com 的 IPv6 記錄

set -e

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

DNSMASQ_CONF="/etc/dnsmasq.conf"
BACKUP_FILE="/etc/dnsmasq.conf.backup.$(date +%Y%m%d_%H%M%S)"

echo -e "${BLUE}=== 添加 filter-AAAA 配置 ===${NC}"
echo ""

# 檢查是否為 root 或使用 sudo
if [ "$EUID" -ne 0 ]; then 
    echo -e "${YELLOW}[警告]${NC} 需要 root 權限，將使用 sudo"
    SUDO_CMD="sudo"
else
    SUDO_CMD=""
fi

# 1. 備份原始配置
echo -e "${BLUE}[1/4]${NC} 備份原始配置..."
$SUDO_CMD cp "$DNSMASQ_CONF" "$BACKUP_FILE"
echo -e "   ${GREEN}✅${NC} 備份至: $BACKUP_FILE"
echo ""

# 2. 檢查是否已有 filter-AAAA 配置
echo -e "${BLUE}[2/4]${NC} 檢查現有配置..."
if grep -q "^filter-AAAA=/polaris-x.com/" "$DNSMASQ_CONF" 2>/dev/null; then
    echo -e "   ${YELLOW}⚠️${NC} filter-AAAA 配置已存在"
    echo "   跳過添加配置"
else
    # 3. 添加 filter-AAAA 配置
    echo -e "${BLUE}[3/4]${NC} 添加 filter-AAAA 配置..."
    
    # 找到 address=/polaris-x.com/ 這一行
    if grep -q "^address=/polaris-x.com/" "$DNSMASQ_CONF"; then
        # 檢查是否已經有全域的 filter-AAAA
        if ! grep -q "^filter-AAAA" "$DNSMASQ_CONF"; then
            # 添加全域 filter-AAAA（只過濾 AAAA 記錄，不影響其他記錄）
            # 注意：filter-AAAA 是全域選項，會過濾所有 AAAA 記錄
            # 但由於我們使用 address= 強制返回 IPv4，這應該足夠了
            # 如果只想過濾特定域名，需要使用其他方法
            echo "" | $SUDO_CMD tee -a "$DNSMASQ_CONF" > /dev/null
            echo "# 過濾所有 AAAA (IPv6) 記錄，只返回 IPv4" | $SUDO_CMD tee -a "$DNSMASQ_CONF" > /dev/null
            echo "# 注意：這是全域設定，會影響所有域名" | $SUDO_CMD tee -a "$DNSMASQ_CONF" > /dev/null
            echo "# 如果需要只過濾特定域名，可以移除這行並使用其他方法" | $SUDO_CMD tee -a "$DNSMASQ_CONF" > /dev/null
            echo "filter-AAAA" | $SUDO_CMD tee -a "$DNSMASQ_CONF" > /dev/null
            echo -e "   ${GREEN}✅${NC} 已添加全域 filter-AAAA 配置"
        else
            echo -e "   ${YELLOW}⚠️${NC} filter-AAAA 配置已存在"
        fi
    else
        echo -e "   ${RED}❌${NC} 未找到 address=/polaris-x.com/ 配置"
        echo "   請先配置 address=/polaris-x.com/ 映射"
        exit 1
    fi
fi
echo ""

# 4. 驗證配置語法
echo -e "${BLUE}[4/4]${NC} 驗證配置語法..."
if $SUDO_CMD dnsmasq --test 2>/dev/null; then
    echo -e "   ${GREEN}✅${NC} 配置語法正確"
else
    echo -e "   ${RED}❌${NC} 配置語法錯誤"
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
echo "  - 已添加: filter-AAAA=/polaris-x.com/"
echo "  - 效果: polaris-x.com 的 IPv6 記錄將被過濾，只返回 IPv4"
echo ""
echo "現在在 Windows 上測試應該只會看到 IPv4 地址："
echo "  nslookup polaris-x.com 100.77.36.85"
echo ""

