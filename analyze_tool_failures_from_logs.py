"""
從日誌片段分析工具失敗情況

根據您提供的日誌片段，分析可能的工具失敗原因。
"""

import sys
import os

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

def analyze_logs():
    """分析日誌片段中的工具失敗情況"""
    
    print("=" * 80)
    print("工具失敗分析報告（基於日誌片段）")
    print("=" * 80)
    print()
    
    # 從日誌片段中提取的信息
    print("📊 觀察到的問題：")
    print("-" * 80)
    print("1. 循環推理檢測 (Circular Reasoning)")
    print("   - 多次出現 'Stopped due: circular_reasoning'")
    print("   - 這表示系統檢測到重複的決策模式")
    print("   - 可能原因：")
    print("     • 工具執行成功但返回結果不足")
    print("     • 答案質量評估過於嚴格，導致重複嘗試")
    print("     • 工具選擇邏輯陷入循環")
    print()
    
    print("2. 答案被拒絕 (Answer Rejection)")
    print("   - 多次出現 'Generated answer rejected'")
    print("   - Quality score: 0.25 (低於閾值)")
    print("   - Hallucination risk: 1.00 (極高風險)")
    print("   - 可能原因：")
    print("     • 收集的結果不足以生成高質量答案")
    print("     • 答案與源文檔不匹配（幻覺檢測）")
    print("     • 評估標準過於嚴格")
    print()
    
    print("3. 工具執行情況")
    print("   - 從日誌中看到 vector_search 執行成功")
    print("   - 但沒有看到 pageindex_search 的執行記錄")
    print("   - 可能原因：")
    print("     • PageIndex 工具未被調用（Orchestrator 未選擇）")
    print("     • PageIndex 工具執行失敗但未記錄")
    print("     • PageIndex 工具返回空結果")
    print()
    
    print("🔍 建議檢查的事項：")
    print("-" * 80)
    print("1. 檢查 error_history 和 unavailable_tools 狀態")
    print("   - 使用診斷端點: GET /api/chat/sessions/{session_id}/diagnostics")
    print()
    print("2. 檢查 search_history 中的失敗記錄")
    print("   - 查看哪些搜尋返回了空結果")
    print("   - 檢查工具執行是否真的失敗，還是只是結果不足")
    print()
    print("3. 檢查 PageIndex 工具狀態")
    print("   - 確認 PageIndex 是否已構建")
    print("   - 檢查 PageIndex 工具是否在可用工具列表中")
    print()
    print("4. 檢查答案質量評估標準")
    print("   - Quality score 閾值可能過高")
    print("   - Hallucination risk 檢測可能過於嚴格")
    print()
    
    print("💡 可能的解決方案：")
    print("-" * 80)
    print("1. 調整答案質量評估閾值")
    print("   - 降低 quality score 要求")
    print("   - 調整 hallucination risk 閾值")
    print()
    print("2. 改進工具選擇邏輯")
    print("   - 確保 PageIndex 工具被優先考慮（如果可用）")
    print("   - 避免重複使用相同工具")
    print()
    print("3. 改進循環推理檢測")
    print("   - 當前檢測可能過於嚴格")
    print("   - 考慮區分「工具失敗」和「結果不足」")
    print()
    print("4. 增強錯誤日誌記錄")
    print("   - 記錄工具執行的詳細結果")
    print("   - 記錄為什麼工具未被選擇")
    print()


if __name__ == "__main__":
    analyze_logs()
