# Hallucination 檢測準則

本文檔整理 hallucination（幻覺）檢測的判定準則，結合現有實現和業界最佳實踐。

## 目錄

1. [Hallucination 定義](#hallucination-定義)
2. [檢測方法概覽](#檢測方法概覽)
3. [規則基礎檢測（Rule-based）](#規則基礎檢測-rule-based)
4. [LLM-as-a-Judge 檢測](#llm-as-a-judge-檢測)
5. [綜合判定準則](#綜合判定準則)
6. [風險分級](#風險分級)
7. [實施建議](#實施建議)

---

## Hallucination 定義

**Hallucination（幻覺）** 是指 LLM 生成的答案中包含以下類型的錯誤信息：

1. **未支持的聲明（Unsupported Claims）**：答案中的聲明在提供的來源中找不到支持證據
2. **無效引用（Invalid Citations）**：引用的來源 ID 不存在於收集的結果中
3. **矛盾信息（Contradictory Information）**：答案中的信息與來源內容相矛盾
4. **過度推斷（Over-extrapolation）**：答案超出了來源所支持的範圍
5. **未引用斷言（Uncited Claims）**：長句子或重要斷言沒有提供引用

---

## 檢測方法概覽

系統採用**混合檢測方法**，結合規則基礎檢測和 LLM-as-a-Judge 評估：

### 方法 1：規則基礎檢測（Rule-based Detection）

**位置**：`open_notebook/services/evaluation_service.py::detect_hallucination`

**特點**：
- 快速、低成本
- 基於引用驗證和統計分析
- 適用於實時檢測

### 方法 2：LLM-as-a-Judge 評估

**位置**：`open_notebook/services/evaluation_service.py::evaluate_results` + `prompts/chat_agentic/evaluation.jinja`

**特點**：
- 更準確的語義理解
- 能檢測複雜的幻覺模式
- 需要 LLM 調用，成本較高

### 方法 3：兩階段提示（Two-step Prompting）

**參考**：[Datadog 的 LLM Hallucination Detection](https://www.datadoghq.com/blog/ai/llm-hallucination-detection/)

**特點**：
- 第一階段：鼓勵自我批評和多角度思考
- 第二階段：結構化輸出，轉換為標準格式
- 提高檢測準確性

---

## 規則基礎檢測（Rule-based）

### 檢測步驟

#### 步驟 1：提取引用

```python
# 從答案中提取所有引用（格式：[id]）
citations = re.findall(r'\[([^\]]+)\]', answer)
citation_ids = set(citations)
```

**判定準則**：
- 答案中應包含引用標記 `[source_id]`
- 引用格式必須正確（方括號內為來源 ID）

#### 步驟 2：驗證引用有效性

```python
# 從搜尋結果中獲取所有有效的 result ID
result_ids = set()
for result in results:
    result_id = result.get("id", "")
    parent_id = result.get("parent_id", "")
    # 收集所有有效的 ID

# 驗證引用
valid_citations = citation_ids.intersection(result_ids)
invalid_citations = citation_ids - result_ids
```

**判定準則**：
- ✅ **有效引用**：引用 ID 存在於 `collected_results` 中
- ❌ **無效引用**：引用 ID 不存在於結果中 → **高風險指標**

#### 步驟 3：檢測未引用斷言

```python
# 找出長句子（>50 字符）但沒有引用的句子
sentences = re.split(r'[.!?]+', answer)
uncited_sentences = []
for sentence in sentences:
    if len(sentence.strip()) > 50 and not re.search(r'\[[^\]]+\]', sentence):
        uncited_sentences.append(sentence.strip()[:100])
```

**判定準則**：
- ⚠️ **未引用長句子**：長度 > 50 字符且無引用 → **中等風險指標**
- ✅ **有引用的句子**：即使很長，只要有引用標記，風險較低

#### 步驟 4：計算風險分數

```python
# 計算總斷言數（長度 > 20 字符的句子）
total_claims = len([s for s in sentences if len(s.strip()) > 20])

# 計算引用比例
citation_ratio = len(valid_citations) / max(total_claims, 1)

# 風險分數 = 1 - 引用比例（引用越少，風險越高）
hallucination_risk = 1.0 - min(citation_ratio, 1.0)
```

**判定準則**：
- **引用比例** = 有效引用數 / 總斷言數
- **風險分數** = 1 - 引用比例
- **風險判定**：`hallucination_risk > 0.3` → `has_hallucination_risk = True`

### 規則基礎檢測的輸出

```python
{
    "has_hallucination_risk": bool,        # 是否有風險（風險分數 > 0.3）
    "hallucination_risk_score": float,    # 風險分數 (0.0-1.0)
    "valid_citations": List[str],          # 有效的引用 ID
    "invalid_citations": List[str],       # 無效的引用 ID
    "uncited_sentences": List[str],       # 未引用的長句子（前 3 個）
    "citation_ratio": float,              # 引用比例
}
```

---

## LLM-as-a-Judge 檢測

### 評估 Prompt 結構

**位置**：`prompts/chat_agentic/evaluation.jinja`

### 評估維度

根據 prompt 模板，LLM 評估器需要檢查以下維度：

#### 1. 信息完整性（Information Completeness）
- 搜尋結果是否足以完整回答問題？
- 答案是否涵蓋了問題的所有方面？

#### 2. 相關性（Relevance）
- 搜尋結果是否直接針對問題？
- 答案是否與問題相關？

#### 3. 答案質量（Answer Quality）
- 答案是否準確、完整、結構良好？
- 語言是否清晰、邏輯是否連貫？

#### 4. 引用質量（Citation Quality）
- 來源是否正確引用？
- 引用是否與聲明相關？

#### 5. **Hallucination 檢測（CRITICAL）**

**必須檢測的幻覺類型**：

1. **未支持的聲明**
   - 答案中的聲明在搜尋結果中找不到支持
   - 檢查每個重要斷言是否有對應的來源

2. **矛盾信息**
   - 答案中的信息與來源內容相矛盾
   - 檢查邏輯一致性

3. **過度推斷**
   - 答案超出了來源所支持的範圍
   - 檢查是否有合理的推斷邊界

4. **無效引用**
   - 引用的來源 ID 必須存在於 `collected_results` 中
   - 檢查引用格式和有效性

5. **未引用斷言**
   - 標記高風險句子（做出斷言但無證據）
   - 特別關注長句子（>50 字符）且無引用

### LLM 評估輸出格式

```json
{
    "score": 0.85,                    // 總體質量分數 (0.0-1.0)
    "reasoning": "...",               // 評估理由
    "confidence": 0.9,                // 評估信心 (0.0-1.0)
    "decision": "synthesize",         // 決策：continue/refine_search/synthesize
    "hallucination_risk": 0.1,       // 幻覺風險分數 (0.0-1.0)
    "hallucination_notes": "..."      // 幻覺檢測說明
}
```

### 兩階段提示方法（參考 Datadog）

**階段 1：自由格式評估**
- 鼓勵自我批評
- 考慮多種解釋
- 不限制輸出格式

**階段 2：結構化輸出**
- 使用較小的 LLM 進行格式轉換
- 將自由格式輸出轉換為標準 JSON
- 節省成本

**提示設計技巧**：
- 將 context 稱為 "expert advice"（專家建議）
- 將 answer 稱為 "candidate answer"（候選答案）
- 創造不對稱性，強調 context 是權威來源

---

## 綜合判定準則

### 判定流程

```
1. 規則基礎檢測（快速篩選）
   ↓
   如果風險分數 > 0.6 → 直接判定為高風險
   ↓
2. LLM-as-a-Judge 評估（深度分析）
   ↓
   結合規則分數和 LLM 分數
   ↓
3. 綜合判定
```

### 判定標準

#### 高風險（High Risk）判定條件

滿足以下**任一條件**即判定為高風險：

1. **規則基礎檢測**：
   - `hallucination_risk_score > 0.6`
   - 或 `invalid_citations` 數量 > 0（存在無效引用）
   - 或 `uncited_sentences` 數量 >= 3（多個未引用長句子）

2. **LLM 評估**：
   - `hallucination_risk > 0.6`
   - 或 `hallucination_notes` 明確指出存在未支持的聲明

3. **綜合指標**：
   - `citation_ratio < 0.3`（引用比例低於 30%）
   - 且 `total_claims >= 3`（有多個斷言）

#### 中等風險（Moderate Risk）判定條件

滿足以下條件：

1. **規則基礎檢測**：
   - `0.3 < hallucination_risk_score <= 0.6`
   - 或 `uncited_sentences` 數量 = 1-2

2. **LLM 評估**：
   - `0.3 < hallucination_risk <= 0.6`
   - 或評估指出部分聲明缺乏支持

#### 低風險（Low Risk）判定條件

滿足以下**所有條件**：

1. **規則基礎檢測**：
   - `hallucination_risk_score <= 0.3`
   - 且 `invalid_citations` 數量 = 0
   - 且 `uncited_sentences` 數量 = 0

2. **LLM 評估**：
   - `hallucination_risk <= 0.3`
   - 且所有重要斷言都有有效引用

3. **引用質量**：
   - `citation_ratio >= 0.7`（引用比例高於 70%）

---

## 風險分級

### 風險分數對應的行為

| 風險分數 | 風險等級 | 系統行為 | 說明 |
|---------|---------|---------|------|
| `0.0 - 0.3` | 低風險 | ✅ 接受答案 | 引用充分，風險低 |
| `0.3 - 0.6` | 中等風險 | ⚠️ 優化搜尋 | 部分斷言缺乏支持，需要更多信息 |
| `0.6 - 1.0` | 高風險 | ❌ 拒絕答案，繼續搜尋 | 存在明顯幻覺，需要重新搜尋 |

### 特殊情況處理

#### 接近迭代限制時

```python
is_near_limit = iteration >= max_iterations - 3

if has_risk and risk_score > 0.6:
    if is_near_limit:
        # 強制接受（避免無限循環）
        # 但仍標記為高風險
    else:
        # 正常拒絕，繼續搜尋
```

**判定準則**：
- 即使風險高，如果接近迭代限制（剩餘 <= 3 次），強制接受答案
- 但會在日誌和返回結果中標記高風險

#### 空答案處理

```python
if not answer or len(answer.strip()) == 0:
    return {
        "has_hallucination_risk": False,
        "hallucination_risk_score": 0.0,
        "note": "Answer is empty, cannot detect hallucinations"
    }
```

**判定準則**：
- 空答案不進行幻覺檢測
- 返回無風險（因為沒有內容可檢測）

#### 無有效斷言處理

```python
if total_claims == 0:
    return {
        "has_hallucination_risk": False,
        "hallucination_risk_score": 0.0,
        "note": "No valid claims found, cannot detect hallucinations"
    }
```

**判定準則**：
- 如果沒有有效斷言（所有句子都 < 20 字符），不判定為高風險
- 避免誤判簡短回答

---

## 實施建議

### 1. 多層檢測策略

**建議流程**：
1. **第一層**：規則基礎檢測（快速、低成本）
2. **第二層**：如果規則檢測發現中等風險，進行 LLM 評估
3. **第三層**：如果 LLM 評估也發現高風險，拒絕答案

### 2. 引用規範

**強制要求**：
- 所有重要斷言（>50 字符的句子）必須包含引用
- 引用格式：`[source_id]`
- 引用 ID 必須存在於 `collected_results` 中

**建議**：
- 在生成答案的 prompt 中明確要求引用格式
- 在評估 prompt 中強調引用驗證的重要性

### 3. 風險閾值調整

**當前閾值**：
- 風險判定：`> 0.3`
- 高風險：`> 0.6`

**調整建議**：
- 根據實際使用情況調整閾值
- 可以根據問題類型（事實性 vs 開放性）使用不同閾值
- 可以根據來源質量調整閾值

### 4. 評估模型選擇

**建議**：
- 規則基礎檢測：無需模型，快速執行
- LLM 評估：使用較強的模型（如 GPT-4o）進行評估
- 兩階段提示：第一階段用強模型，第二階段用較小模型

### 5. 日誌和監控

**建議記錄**：
- 每次檢測的風險分數
- 無效引用列表
- 未引用斷言列表
- LLM 評估的 reasoning

**監控指標**：
- 平均風險分數
- 高風險答案比例
- 無效引用頻率
- 未引用斷言頻率

---

## 參考資料

1. **現有實現**：
   - `open_notebook/services/evaluation_service.py::detect_hallucination`
   - `open_notebook/graphs/chat.py::synthesize_answer`
   - `prompts/chat_agentic/evaluation.jinja`

2. **業界最佳實踐**：
   - [Datadog: Detecting hallucinations in RAG LLM applications](https://www.datadoghq.com/blog/ai/llm-hallucination-detection/)
   - LLM-as-a-Judge 方法
   - Two-step prompting 技術

3. **相關研究**：
   - HaluBench: Hallucination Benchmark
   - RAGTruth: Hallucination Corpus
   - Faithfulness Evaluation Format

---

## 附錄：檢測檢查清單

在實施或審查 hallucination 檢測時，使用以下檢查清單：

### 規則基礎檢測檢查清單

- [ ] 是否提取了所有引用？
- [ ] 是否驗證了引用的有效性？
- [ ] 是否檢測了未引用的長句子？
- [ ] 是否計算了引用比例？
- [ ] 是否正確計算了風險分數？
- [ ] 是否處理了空答案情況？
- [ ] 是否處理了無有效斷言情況？

### LLM 評估檢查清單

- [ ] 是否檢查了未支持的聲明？
- [ ] 是否檢查了矛盾信息？
- [ ] 是否檢查了過度推斷？
- [ ] 是否驗證了引用有效性？
- [ ] 是否標記了未引用斷言？
- [ ] 是否返回了結構化的評估結果？

### 綜合判定檢查清單

- [ ] 是否結合了規則分數和 LLM 分數？
- [ ] 是否正確處理了高風險情況？
- [ ] 是否正確處理了接近迭代限制的情況？
- [ ] 是否記錄了詳細的檢測日誌？
- [ ] 是否返回了足夠的診斷信息？

---

**文檔版本**：1.0  
**最後更新**：2025-01-27  
**維護者**：RAG Notebook Team

