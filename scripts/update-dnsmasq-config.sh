#!/bin/bash
# 更新 dnsmasq 配置以服務整個 Tailscale tailnet

set -e

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

DNSMASQ_CONF="/etc/dnsmasq.conf"
BACKUP_FILE="/etc/dnsmasq.conf.backup.$(date +%Y%m%d_%H%M%S)"

echo -e "${BLUE}=== 更新 dnsmasq 配置 ===${NC}"
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

# 2. 獲取 Tailscale IP
echo -e "${BLUE}[2/5]${NC} 獲取 Tailscale IP..."
TAILSCALE_IP=$(tailscale ip -4 2>/dev/null | head -n1 || echo "")
if [ -z "$TAILSCALE_IP" ]; then
    echo -e "   ${RED}❌${NC} 無法獲取 Tailscale IP"
    echo "   請確認 Tailscale 已連接"
    exit 1
fi
echo -e "   ${GREEN}✅${NC} Tailscale IP: $TAILSCALE_IP"
echo ""

# 3. 更新配置
echo -e "${BLUE}[3/5]${NC} 更新 dnsmasq 配置..."

# 使用 sed 或 awk 來更新配置
# 先檢查是否已經有相關配置
if grep -q "^listen-address=127.0.0.1,100.77.36.85" "$DNSMASQ_CONF" 2>/dev/null; then
    echo -e "   ${YELLOW}⚠️${NC} 配置似乎已經更新過"
else
    # 更新 listen-address
    $SUDO_CMD sed -i "s|^listen-address=127.0.0.1$|listen-address=127.0.0.1,$TAILSCALE_IP|" "$DNSMASQ_CONF"
    
    # 確保有 no-resolv
    if ! grep -q "^no-resolv" "$DNSMASQ_CONF"; then
        $SUDO_CMD sed -i '/^# 不使用 systemd-resolved/a no-resolv' "$DNSMASQ_CONF"
    fi
    
    # 更新上游 DNS 服務器（如果還沒有 Tailscale DNS）
    if ! grep -q "^server=100.100.100.100" "$DNSMASQ_CONF"; then
        # 在 server=8.8.8.8 之前插入 Tailscale DNS
        $SUDO_CMD sed -i "/^server=8.8.8.8/i server=100.100.100.100" "$DNSMASQ_CONF"
    fi
    
    # 確保有 polaris-x.com 映射
    if ! grep -q "^address=/polaris-x.com/" "$DNSMASQ_CONF"; then
        # 在自定義域名映射區域添加
        if grep -q "^# 自定義域名映射" "$DNSMASQ_CONF"; then
            $SUDO_CMD sed -i "/^# 自定義域名映射/a address=/polaris-x.com/$TAILSCALE_IP" "$DNSMASQ_CONF"
        else
            echo "" | $SUDO_CMD tee -a "$DNSMASQ_CONF" > /dev/null
            echo "# 自定義域名映射 - 將 polaris-x.com 解析到本機 Tailscale IP" | $SUDO_CMD tee -a "$DNSMASQ_CONF" > /dev/null
            echo "address=/polaris-x.com/$TAILSCALE_IP" | $SUDO_CMD tee -a "$DNSMASQ_CONF" > /dev/null
        fi
    else
        # 更新現有的映射
        $SUDO_CMD sed -i "s|^address=/polaris-x.com/.*|address=/polaris-x.com/$TAILSCALE_IP|" "$DNSMASQ_CONF"
    fi
fi

echo -e "   ${GREEN}✅${NC} 配置已更新"
echo ""

# 4. 驗證配置語法
echo -e "${BLUE}[4/5]${NC} 驗證配置語法..."
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

# 6. 驗證監聽端口
echo -e "${BLUE}[驗證]${NC} 檢查監聽端口..."
if ss -tuln 2>/dev/null | grep -q ":$TAILSCALE_IP:53" || netstat -tuln 2>/dev/null | grep -q ":$TAILSCALE_IP:53"; then
    echo -e "   ${GREEN}✅${NC} dnsmasq 正在監聽 Tailscale IP ($TAILSCALE_IP:53)"
else
    echo -e "   ${YELLOW}⚠️${NC} 未檢測到在 Tailscale IP 上監聽，可能需要檢查防火牆"
fi

# 測試 DNS 解析
echo ""
echo -e "${BLUE}[測試]${NC} 測試 DNS 解析..."
if dig @127.0.0.1 +short polaris-x.com 2>/dev/null | grep -q "$TAILSCALE_IP"; then
    echo -e "   ${GREEN}✅${NC} 本地解析測試成功: polaris-x.com -> $TAILSCALE_IP"
else
    echo -e "   ${YELLOW}⚠️${NC} 本地解析測試失敗，請檢查配置"
fi

if dig @$TAILSCALE_IP +short polaris-x.com 2>/dev/null | grep -q "$TAILSCALE_IP"; then
    echo -e "   ${GREEN}✅${NC} Tailscale IP 解析測試成功: polaris-x.com -> $TAILSCALE_IP"
else
    echo -e "   ${YELLOW}⚠️${NC} Tailscale IP 解析測試失敗，可能需要檢查防火牆或等待服務完全啟動"
fi

echo ""
echo -e "${GREEN}=== 配置完成 ===${NC}"
echo ""
echo "配置摘要:"
echo "  - 監聽地址: 127.0.0.1, $TAILSCALE_IP"
echo "  - 上游 DNS: 100.100.100.100 (Tailscale), 8.8.8.8, 8.8.4.4, 1.1.1.1"
echo "  - 域名映射: polaris-x.com -> $TAILSCALE_IP"
echo ""
echo "其他 tailnet 設備可以使用 $TAILSCALE_IP 作為 DNS 服務器來解析 polaris-x.com"
echo ""

