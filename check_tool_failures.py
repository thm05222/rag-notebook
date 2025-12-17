"""
診斷腳本：檢查會話中的工具失敗情況

使用方法：
    python check_tool_failures.py <session_id>

例如：
    python check_tool_failures.py 4qhisqhs2jvrur6aqnuc
"""

import asyncio
import sys
import json
import os
from typing import Dict, Any

# 設置 Windows 控制台編碼為 UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 假設 API 運行在本地
API_BASE_URL = "http://localhost:5055"


async def check_tool_failures(session_id: str):
    """檢查指定會話的工具失敗情況"""
    import aiohttp
    
    # 移除 chat_session: 前綴（如果存在），API 端點會自動處理
    clean_session_id = session_id.replace("chat_session:", "")
    url = f"{API_BASE_URL}/api/chat/sessions/{clean_session_id}/diagnostics"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                if response.status == 404:
                    print(f"❌ 會話 {session_id} 不存在")
                    return
                elif response.status != 200:
                    error_text = await response.text()
                    print(f"❌ 錯誤 ({response.status}): {error_text}")
                    return
                
                data = await response.json()
                
                print("=" * 80)
                print(f"會話診斷報告: {session_id}")
                print("=" * 80)
                print()
                
                # 工具失敗摘要
                tool_failure_summary = data.get("tool_failure_summary", {})
                if tool_failure_summary:
                    print("📊 工具失敗摘要:")
                    print("-" * 80)
                    for tool_name, info in tool_failure_summary.items():
                        count = info.get("count", 0)
                        print(f"  • {tool_name}: {count} 次失敗")
                        for error_info in info.get("errors", [])[:3]:  # 只顯示前3個錯誤
                            iteration = error_info.get("iteration", "?")
                            error_msg = error_info.get("error", "Unknown")[:100]
                            print(f"    - 迭代 {iteration}: {error_msg}")
                    print()
                else:
                    print("✅ 沒有工具失敗記錄")
                    print()
                
                # 不可用的工具
                unavailable_tools = data.get("unavailable_tools", [])
                if unavailable_tools:
                    print("⚠️  不可用的工具:")
                    print("-" * 80)
                    for tool in unavailable_tools:
                        print(f"  • {tool}")
                    print()
                else:
                    print("✅ 所有工具都可用")
                    print()
                
                # 失敗的搜尋
                failed_searches = data.get("failed_searches", [])
                if failed_searches:
                    print("🔍 失敗的搜尋:")
                    print("-" * 80)
                    for search in failed_searches[:10]:  # 只顯示前10個
                        tool = search.get("tool", "unknown")
                        query = search.get("query", "")[:50]
                        error = search.get("error", "Unknown")
                        print(f"  • {tool}: {query}... | 錯誤: {error[:80]}")
                    print()
                
                # 統計信息
                print("📈 統計信息:")
                print("-" * 80)
                print(f"  總迭代次數: {data.get('total_iterations', 0)}")
                print(f"  總錯誤數: {data.get('total_errors', 0)}")
                print(f"  工具失敗數: {data.get('total_tool_failures', 0)}")
                print()
                
                # 決策歷史（最近）
                decision_history = data.get("decision_history", [])
                if decision_history:
                    print("🤔 最近的決策歷史 (最後10個):")
                    print("-" * 80)
                    for decision in decision_history[-10:]:
                        if isinstance(decision, str):
                            print(f"  • {decision}")
                        elif isinstance(decision, dict):
                            action = decision.get("action", "unknown")
                            tool = decision.get("tool_name", "N/A")
                            print(f"  • {action}: {tool}")
                    print()
                
                # 完整錯誤歷史（可選，如果用戶想要詳細信息）
                if len(sys.argv) > 2 and sys.argv[2] == "--verbose":
                    error_history = data.get("error_history", [])
                    if error_history:
                        print("📋 完整錯誤歷史:")
                        print("-" * 80)
                        for error in error_history:
                            step = error.get("step", "unknown")
                            tool = error.get("tool", "N/A")
                            iteration = error.get("iteration", "?")
                            error_msg = error.get("error", "Unknown")[:200]
                            print(f"  [{iteration}] {step} - {tool}: {error_msg}")
                        print()
                
        except aiohttp.ClientError as e:
            print(f"❌ 連接錯誤: {e}")
            print(f"   請確保 API 服務運行在 {API_BASE_URL}")
        except Exception as e:
            print(f"❌ 錯誤: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python check_tool_failures.py <session_id> [--verbose]")
        print("例如: python check_tool_failures.py 4qhisqhs2jvrur6aqnuc")
        sys.exit(1)
    
    session_id = sys.argv[1]
    asyncio.run(check_tool_failures(session_id))
