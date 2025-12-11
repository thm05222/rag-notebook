#!/bin/bash
# 清除 DNS 緩存腳本

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=== 清除 DNS 緩存 ===${NC}"
echo ""

# 檢查是否為 root 或使用 sudo
if [ "$EUID" -ne 0 ]; then 
    SUDO_CMD="sudo"
else
    SUDO_CMD=""
fi

# 清除 systemd-resolved 緩存
if command -v resolvectl &> /dev/null; then
    echo -e "${BLUE}[1/3]${NC} 使用 resolvectl 清除緩存..."
    if $SUDO_CMD resolvectl flush-caches 2>/dev/null; then
        echo -e "   ${GREEN}✅${NC} systemd-resolved 緩存已清除"
    else
        echo -e "   ${YELLOW}⚠️${NC} 無法清除 systemd-resolved 緩存"
    fi
elif command -v systemd-resolve &> /dev/null; then
    echo -e "${BLUE}[1/3]${NC} 使用 systemd-resolve 清除緩存..."
    if $SUDO_CMD systemd-resolve --flush-caches 2>/dev/null; then
        echo -e "   ${GREEN}✅${NC} systemd-resolved 緩存已清除"
    else
        echo -e "   ${YELLOW}⚠️${NC} 無法清除 systemd-resolved 緩存"
    fi
else
    echo -e "${YELLOW}⚠️${NC} 未找到 systemd-resolved"
fi

# 清除 nscd 緩存
if systemctl is-active --quiet nscd 2>/dev/null; then
    echo -e "${BLUE}[2/3]${NC} 重啟 nscd 服務..."
    if $SUDO_CMD systemctl restart nscd 2>/dev/null; then
        echo -e "   ${GREEN}✅${NC} nscd 緩存已清除"
    else
        echo -e "   ${YELLOW}⚠️${NC} 無法重啟 nscd"
    fi
else
    echo -e "${YELLOW}⚠️${NC} nscd 未運行"
fi

# 清除 dnsmasq 緩存
if systemctl is-active --quiet dnsmasq 2>/dev/null; then
    echo -e "${BLUE}[3/3]${NC} 重啟 dnsmasq 服務..."
    if $SUDO_CMD systemctl restart dnsmasq 2>/dev/null; then
        echo -e "   ${GREEN}✅${NC} dnsmasq 緩存已清除"
    else
        echo -e "   ${YELLOW}⚠️${NC} 無法重啟 dnsmasq"
    fi
else
    echo -e "${YELLOW}⚠️${NC} dnsmasq 未運行"
fi

echo ""
echo -e "${GREEN}=== 完成 ===${NC}"
echo ""
echo "測試 DNS 解析："
echo "  dig +short polaris-x.com"
echo "  ping -c 1 polaris-x.com"

