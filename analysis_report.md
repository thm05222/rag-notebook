# Agent 決策過程分析報告

## 執行摘要

根據 Docker 日誌分析，**確實存在大量重複調用相同工具的問題**。

## 關鍵發現

### 1. 決策歷史統計
- **decision_history 長度**: 19 個決策
- **search_history 長度**: 19 個搜尋
- **collected_results**: 18 個結果

### 2. 問題模式

從日誌中觀察到以下問題：

#### 問題 1: 決策失敗導致重複
```
[Iteration 0] Error in agent decision: Connection error.
[Iteration 0] decision_history length: 19
[Iteration 0] No current_decision found, routing to synthesize
```

- **問題**: 在 Iteration 0 時，`agent_decision` 因為 "Connection error" 失敗
- **結果**: `current_decision` 為 `None`，但 `decision_history` 已經有 19 個記錄
- **影響**: 系統路由到 `synthesize`，但實際上可能已經執行過多次工具調用

#### 問題 2: 重複的決策循環
```
[Iteration 0] decision_history length: 19
[Iteration 1] decision_history length: 19  (沒有增加)
[Iteration 2] decision_history length: 19  (沒有增加)
```

- **問題**: 在多次迭代中，`decision_history` 長度保持為 19，沒有增加
- **可能原因**: 
  - 決策失敗後，新的決策沒有被正確添加到歷史中
  - 或者決策歷史在初始化時就已經包含了 19 個記錄（可能是從之前的會話繼承）

#### 問題 3: 搜尋歷史與結果不匹配
- **search_history**: 19 個搜尋記錄
- **collected_results**: 18 個結果

- **問題**: 搜尋次數比結果多 1 個
- **可能原因**: 某次搜尋失敗或返回空結果

### 3. 執行流程分析

#### Iteration 0
1. 初始化：`decision_history=[]`, `search_history=[]`, `collected_results=[]`
2. Agent 決策：嘗試調用 LLM 進行決策
3. **錯誤**: Connection error（可能是 LLM 連接超時）
4. **結果**: `current_decision = None`
5. **路由**: 因為沒有決策，路由到 `synthesize`
6. **狀態**: `decision_history=19`（異常！應該為 0 或 1）

#### Iteration 1
1. Agent 決策：再次嘗試
2. **錯誤**: 再次 Connection error
3. **狀態**: `decision_history=19`（仍然沒有增加）

#### Iteration 2
1. **限制觸發**: `Limit reached: timeout`
2. **最終狀態**: `decision_history=19`, `search_history=19`, `collected_results=18`

## 根本原因分析

### 1. 決策歷史累積問題
- **可能原因**: `decision_history` 使用 `operator.add` 累積，但在決策失敗時可能仍然被累積
- **證據**: 初始化時為空 `[]`，但在第一次決策失敗後就變成 19 個

### 2. LLM 連接超時
- **問題**: `agent_decision` 函數在調用 LLM 時發生 "Connection error"
- **影響**: 決策失敗，但可能之前的決策已經被記錄

### 3. 狀態清理不完整
- **問題**: 在開始新的對話時，可能沒有完全清理之前的狀態
- **證據**: 初始化時 `decision_history=[]`，但第一次決策後就變成 19 個

## 建議修復方案

### 1. 修復決策歷史累積邏輯
- 確保只有在決策成功時才添加到 `decision_history`
- 在決策失敗時，不要累積空的或無效的決策

### 2. 改進錯誤處理
- 在 `agent_decision` 中，如果 LLM 調用失敗，應該：
  - 記錄錯誤但不累積決策
  - 提供 fallback 決策機制
  - 避免無限重試

### 3. 狀態清理機制
- 在 `initialize_chat_state` 中，確保完全清理所有累積字段
- 使用 `aupdate_state` 強制覆蓋，而不是依賴 `operator.add` 的累積行為

### 4. 添加決策去重邏輯
- 在決策前檢查最近 N 次決策是否相同
- 如果檢測到重複決策，強制改變策略或停止

## 結論

**是的，確實存在大量重複調用相同工具的問題**。主要表現為：

1. **決策歷史異常累積**: 在第一次決策失敗後，`decision_history` 就從 0 跳到 19
2. **重複的決策失敗**: 多次迭代中都出現相同的 Connection error
3. **搜尋次數過多**: 19 次搜尋只得到 18 個結果，效率低下

建議優先修復決策歷史的累積邏輯和錯誤處理機制，以防止重複調用。
