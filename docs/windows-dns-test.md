# Windows DNS 測試指令

## 方法 1：使用 nslookup（Windows 內建）

```cmd
# 測試 polaris-x.com 解析
nslookup polaris-x.com

# 指定使用您的 dnsmasq 服務器測試
nslookup polaris-x.com 100.77.36.85

# 測試其他域名（確認 DNS 服務器正常運作）
nslookup google.com 100.77.36.85
```

## 方法 2：使用 PowerShell Resolve-DnsName

```powershell
# 測試 polaris-x.com 解析
Resolve-DnsName polaris-x.com

# 指定使用您的 dnsmasq 服務器測試
Resolve-DnsName polaris-x.com -Server 100.77.36.85

# 只顯示 IPv4 地址
Resolve-DnsName polaris-x.com -Server 100.77.36.85 -Type A | Select-Object -ExpandProperty IPAddress
```

## 方法 3：安裝 dig（如果沒有）

### 使用 Chocolatey 安裝：
```powershell
# 以管理員身份執行 PowerShell
choco install bind-toolsonly
```

### 或下載 BIND tools：
- 下載：https://www.isc.org/download/
- 安裝後即可使用 dig 命令

安裝後使用：
```cmd
dig @100.77.36.85 polaris-x.com
dig @100.77.36.85 +short polaris-x.com
```

## 方法 4：使用 ping 測試（最簡單）

```cmd
# 測試 polaris-x.com 解析
ping polaris-x.com

# 查看解析到的 IP 地址
ping -n 1 polaris-x.com
```

## 快速測試腳本（PowerShell）

將以下內容儲存為 `test-dns.ps1`：

```powershell
# 測試 DNS 解析
Write-Host "=== 測試 DNS 解析 ===" -ForegroundColor Cyan
Write-Host ""

# 測試 1: 使用預設 DNS
Write-Host "1. 使用預設 DNS 解析 polaris-x.com:" -ForegroundColor Yellow
try {
    $result = Resolve-DnsName polaris-x.com -Type A -ErrorAction Stop
    $result | ForEach-Object { Write-Host "   $($_.IPAddress)" -ForegroundColor Green }
} catch {
    Write-Host "   解析失敗: $_" -ForegroundColor Red
}
Write-Host ""

# 測試 2: 使用您的 dnsmasq 服務器
Write-Host "2. 使用 dnsmasq (100.77.36.85) 解析 polaris-x.com:" -ForegroundColor Yellow
try {
    $result = Resolve-DnsName polaris-x.com -Server 100.77.36.85 -Type A -ErrorAction Stop
    $result | ForEach-Object { Write-Host "   $($_.IPAddress)" -ForegroundColor Green }
} catch {
    Write-Host "   解析失敗: $_" -ForegroundColor Red
}
Write-Host ""

# 測試 3: 測試其他域名（確認 DNS 服務器正常）
Write-Host "3. 使用 dnsmasq 解析 google.com (測試服務器是否正常):" -ForegroundColor Yellow
try {
    $result = Resolve-DnsName google.com -Server 100.77.36.85 -Type A -ErrorAction Stop
    $result | Select-Object -First 1 | ForEach-Object { Write-Host "   $($_.IPAddress)" -ForegroundColor Green }
} catch {
    Write-Host "   解析失敗: $_" -ForegroundColor Red
}
Write-Host ""

Write-Host "=== 測試完成 ===" -ForegroundColor Cyan
```

執行方式：
```powershell
powershell -ExecutionPolicy Bypass -File test-dns.ps1
```

