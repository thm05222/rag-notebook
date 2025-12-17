#!/usr/bin/env python3
"""
SearXNG 連線診斷腳本
用於檢查 SearXNG 服務是否正常運行和可訪問
"""

import os
import sys
import subprocess
import httpx
from typing import List, Tuple

def check_docker_container() -> Tuple[bool, str]:
    """檢查 SearXNG 容器是否運行"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=searxng", "--format", "{{.Names}}\t{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, result.stdout.strip()
        else:
            return False, "SearXNG 容器未運行"
    except FileNotFoundError:
        return False, "Docker 命令未找到，請確認 Docker 已安裝"
    except Exception as e:
        return False, f"檢查容器時發生錯誤: {str(e)}"

def check_network() -> Tuple[bool, str]:
    """檢查網路配置"""
    try:
        result = subprocess.run(
            ["docker", "network", "inspect", "rag-notebook_rag_network"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return True, "網路 rag-notebook_rag_network 存在"
        else:
            # 嘗試其他可能的網路名稱
            result2 = subprocess.run(
                ["docker", "network", "ls", "--format", "{{.Name}}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            networks = result2.stdout.strip().split('\n')
            rag_networks = [n for n in networks if 'rag' in n.lower()]
            if rag_networks:
                return False, f"找不到 rag-notebook_rag_network，但找到類似網路: {', '.join(rag_networks)}"
            return False, "找不到 rag_network，請確認 docker-compose.yml 中的網路配置"
    except Exception as e:
        return False, f"檢查網路時發生錯誤: {str(e)}"

def check_searxng_connectivity(url: str) -> Tuple[bool, str]:
    """檢查 SearXNG 連線"""
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{url}/search", params={"q": "test", "format": "json"})
            if response.status_code == 200:
                data = response.json()
                return True, f"連線成功！收到 {len(data.get('results', []))} 個結果"
            else:
                return False, f"HTTP {response.status_code}: {response.text[:200]}"
    except httpx.ConnectError as e:
        return False, f"連線失敗: {str(e)}"
    except Exception as e:
        return False, f"錯誤: {str(e)}"

def main():
    print("=" * 60)
    print("SearXNG 連線診斷工具")
    print("=" * 60)
    print()

    # 1. 檢查容器
    print("1. Checking SearXNG container status...")
    container_ok, container_msg = check_docker_container()
    print(f"   [{'OK' if container_ok else 'FAIL'}] {container_msg}")
    print()

    # 2. 檢查網路
    print("2. Checking Docker network configuration...")
    network_ok, network_msg = check_network()
    print(f"   [{'OK' if network_ok else 'FAIL'}] {network_msg}")
    print()

    # 3. 檢查連線
    print("3. Testing SearXNG connectivity...")
    urls_to_test = [
        ("http://localhost:8080", "localhost"),
        ("http://searxng:8080", "Docker service name (searxng)"),
    ]
    
    # 添加環境變數中的 URL
    env_url = os.getenv("SEARXNG_URL")
    if env_url:
        urls_to_test.insert(0, (env_url, "Environment variable SEARXNG_URL"))

    all_failed = True
    for url, description in urls_to_test:
        print(f"   Testing {description}: {url}")
        ok, msg = check_searxng_connectivity(url)
        status = "[OK]" if ok else "[FAIL]"
        print(f"   {status} {msg}")
        if ok:
            all_failed = False
            print(f"\n   [SUCCESS] Connected to: {url}")
            break
        print()

    print()
    print("=" * 60)
    if all_failed:
        print("Diagnosis: [FAIL] Cannot connect to SearXNG")
        print()
        print("Suggested fixes:")
        print("1. Check container: docker ps | grep searxng")
        print("2. Restart SearXNG: docker-compose restart searxng")
        print("3. View logs: docker logs searxng")
        print("4. Check network: docker network inspect rag-notebook_rag_network")
        print("5. If calling from container, ensure open_notebook is in same network")
        sys.exit(1)
    else:
        print("Diagnosis: [OK] SearXNG connectivity is working")
        sys.exit(0)

if __name__ == "__main__":
    main()
