# RAG 系統腳本整理文檔

本文檔整理所有與 RAG 系統相關的腳本，重點關注 Agentic RAG 的 hallucination 判定、tool 使用、LLM 使用、prompt 及提示詞組合。

## 目錄

1. [核心 Agentic RAG 流程](#核心-agentic-rag-流程)
2. [Hallucination 判定系統](#hallucination-判定系統)
3. [Tool 使用系統](#tool-使用系統)
4. [LLM 使用系統](#llm-使用系統)
5. [Prompt 及提示詞組合](#prompt-及提示詞組合)
6. [相關配置與模型管理](#相關配置與模型管理)

---

## 核心 Agentic RAG 流程

### 主要文件

#### 1. `open_notebook/graphs/agentic_ask.py`

**功能**：Agentic RAG 的核心工作流程圖（LangGraph）

**關鍵組件**：

- **狀態管理** (`AgenticAskState`)：
  - `question`: 用戶問題
  - `iteration_count`: 迭代次數
  - `search_history`: 搜尋歷史
  - `collected_results`: 累積的搜尋結果
  - `evaluation_result`: 評估結果
  - `partial_answer`: 部分答案
  - `final_answer`: 最終答案
  - `reasoning_trace`: 推理追蹤
  - `token_count`: Token 使用量
  - `decision_history`: 決策歷史（用於循環檢測）

- **核心節點**：
  1. `initialize_state`: 初始化狀態
  2. `agent_decision`: Agent 決策（決定下一步行動）
  3. `execute_tool`: 執行工具
  4. `evaluate_results`: 評估結果
  5. `refine_query`: 優化查詢
  6. `synthesize_answer`: 生成答案

- **工作流程**：
```
START → initialize → agent_decision
                    ↓
        ┌───────────┴───────────┐
        ↓                       ↓
   use_tool              evaluate/synthesize/finish
        ↓                       ↓
   execute_tool          evaluate_results
        ↓                       ↓
   agent_decision ←─────────────┘
        ↓
   synthesize_answer
        ↓
   should_accept_answer → accept/reject → END
```

**關鍵函數**：

```python
# 決策模型
class Decision(BaseModel):
    action: str  # "use_tool", "evaluate", "synthesize", "finish"
    tool_name: Optional[str]
    parameters: Optional[Dict[str, Any]]
    reasoning: str

# 循環推理檢測
def detect_circular_reasoning(state: AgenticAskState) -> bool:
    """檢測是否陷入循環推理"""
    if len(state["decision_history"]) < 3:
        return False
    recent = state["decision_history"][-3:]
    return len(set(recent)) == 1

# 限制檢查
def check_limits(state: AgenticAskState) -> str:
    """檢查是否超過限制（迭代、超時、Token、循環推理）"""
    # 檢查迭代限制
    # 檢查超時
    # 檢查 Token 限制
    # 檢查循環推理
```

---

## Hallucination 判定系統

### 主要文件

#### 1. `open_notebook/services/evaluation_service.py`

**功能**：評估服務，提供規則基礎和 LLM 基礎的評估，包含 hallucination 檢測

**核心方法**：

##### `evaluate_results()` - 綜合評估

**流程**：
1. **規則基礎評估** (`_rule_based_evaluation`)
   - 快速驗證（無 LLM 成本）
   - 檢查結果數量、答案完整性、引用數量

2. **LLM 評估**（兩階段或單階段）
   - `_llm_evaluation_two_stage()`: 兩階段評估（推薦）
   - `_llm_evaluation()`: 單階段評估

3. **Hallucination 檢測** (`detect_hallucination`)
   - 規則基礎檢測
   - 合併規則和 LLM 檢測結果

4. **綜合分數計算**
   - 規則分數：30%
   - LLM 總分：40%
   - 評估維度平均：30%（完整性、相關性、引用質量、一致性）

5. **決策確定** (`_determine_decision`)
   - 根據綜合分數和 hallucination 風險決定下一步

**評估維度**：
- `completeness_score`: 信息完整性
- `relevance_score`: 相關性
- `citation_quality_score`: 引用質量
- `consistency_score`: 一致性（答案內部邏輯）

##### `detect_hallucination()` - Hallucination 檢測

**檢測步驟**：

1. **提取引用**
```python
citations = re.findall(r'\[([^\]]+)\]', answer)
citation_ids = set(citations)
```

2. **驗證引用有效性**
```python
result_ids = set()
for result in results:
    result_id = result.get("id", "")
    parent_id = result.get("parent_id", "")
    result_ids.add(result_id)
    result_ids.add(parent_id)

valid_citations = citation_ids.intersection(result_ids)
invalid_citations = citation_ids - result_ids
```

3. **檢測未引用斷言**
```python
sentences = re.split(r'[.!?]+', answer)
uncited_sentences = []
for sentence in sentences:
    if len(sentence.strip()) > 50 and not re.search(r'\[[^\]]+\]', sentence):
        uncited_sentences.append(sentence.strip()[:100])
```

4. **計算風險分數**
```python
total_claims = len([s for s in sentences if len(s.strip()) > 20])
citation_ratio = len(valid_citations) / max(total_claims, 1)
hallucination_risk = 1.0 - min(citation_ratio, 1.0)
```

**返回格式**：
```python
{
    "has_hallucination_risk": bool,
    "hallucination_risk_score": float,  # 0.0-1.0
    "valid_citations": List[str],
    "invalid_citations": List[str],
    "uncited_sentences": List[str],
    "citation_ratio": float,
}
```

##### `_llm_evaluation_two_stage()` - 兩階段 LLM 評估

**階段 1：自由格式評估**
- 使用強模型（如 GPT-4o）
- 鼓勵自我批評和多角度思考
- 不限制輸出格式
- Prompt: `chat_agentic/evaluation_stage1.jinja`

**階段 2：結構化輸出**
- 使用較小模型進行格式轉換
- 將自由格式輸出轉換為標準 JSON
- 節省成本
- Prompt: `chat_agentic/evaluation_stage2.jinja`

**輸出模型** (`open_notebook/services/evaluation_models.py`):
```python
class HallucinationDetails(BaseModel):
    has_risk: bool
    risk_score: float
    unsupported_claims: List[str]
    invalid_citations: List[str]
    contradictory_info: List[str]
    over_extrapolation: List[str]
    notes: Optional[str]

class EvaluationResult(BaseModel):
    score: float
    reasoning: str
    confidence: float
    decision: Literal["continue", "refine_search", "synthesize", "reject"]
    hallucination: HallucinationDetails
    completeness_score: float
    relevance_score: float
    citation_quality_score: float
    consistency_score: float
    # ... 詳細說明字段
```

#### 2. `docs/hallucination_detection_guidelines.md`

**功能**：Hallucination 檢測的完整指南文檔

**內容**：
- Hallucination 定義
- 檢測方法概覽（規則基礎、LLM-as-a-Judge、兩階段提示）
- 判定準則
- 風險分級
- 實施建議

**風險分級**：
- **低風險** (0.0-0.3): 接受答案
- **中等風險** (0.3-0.6): 優化搜尋
- **高風險** (0.6-1.0): 拒絕答案，繼續搜尋

---

## Tool 使用系統

### 主要文件

#### 1. `open_notebook/services/tool_service.py`

**功能**：工具服務抽象層，提供統一的工具接口

**核心類**：

##### `BaseTool` - 工具基類

```python
class BaseTool(ABC):
    def __init__(
        self,
        name: str,
        description: str,
        timeout: float = 30.0,
        retry_count: int = 2,
        enabled: bool = True,
    ):
        # ...
    
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """執行工具，返回統一格式"""
        raise NotImplementedError
    
    async def execute_with_retry(self, **kwargs) -> Dict[str, Any]:
        """帶重試邏輯的執行"""
        # 重試邏輯
        # 超時處理
        # 錯誤處理
```

**統一返回格式**：
```python
{
    "tool_name": str,
    "success": bool,
    "data": Any,
    "error": Optional[str],
    "execution_time": float,
    "metadata": Dict[str, Any]
}
```

##### `ToolRegistry` - 工具註冊表

```python
class ToolRegistry:
    async def register(self, tool: BaseTool) -> None
    async def unregister(self, tool_name: str) -> None
    async def get_tool(self, tool_name: str) -> Optional[BaseTool]
    async def list_tools(self) -> List[Dict[str, Any]]
    async def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]
```

**內建工具**：

1. **VectorSearchTool** - 向量搜尋工具
   - 使用 Qdrant 進行語義相似度搜尋
   - 參數：`query`, `limit`, `minimum_score`, `search_sources`, `notebook_ids`
   - 超時：60 秒

2. **TextSearchTool** - 文本搜尋工具
   - 使用 SurrealDB 全文搜尋（BM25）
   - 參數：`query`, `limit`, `search_sources`
   - 超時：30 秒

3. **CalculationTool** - 計算工具
   - 數學計算和單位轉換
   - 參數：`expression`
   - 超時：5 秒

4. **InternetSearchTool** - 網路搜尋工具
   - 使用 DuckDuckGo 搜尋
   - 參數：`query`, `limit`
   - 超時：30 秒

5. **MCPToolWrapper** - MCP 工具包裝器
   - 包裝 MCP 伺服器工具
   - 動態註冊 MCP 工具

**工具執行流程**：
```
agent_decision → execute_tool → tool_registry.execute_tool
                                    ↓
                            tool.execute_with_retry
                                    ↓
                            tool.execute (具體實現)
                                    ↓
                            返回統一格式結果
```

#### 2. `open_notebook/graphs/tools.py`

**功能**：LangChain 工具定義

**工具**：
- `get_current_timestamp()`: 獲取當前時間戳

---

## LLM 使用系統

### 主要文件

#### 1. `open_notebook/graphs/utils.py`

**功能**：LLM 模型配置和提供

**核心函數**：

##### `provision_langchain_model()`

```python
async def provision_langchain_model(
    content, model_id, default_type, **kwargs
) -> BaseChatModel:
    """
    根據上下文大小和配置返回最適合的模型
    
    邏輯：
    1. 如果 context > 105,000 tokens → 使用 large_context_model
    2. 如果指定了 model_id → 使用該模型
    3. 否則 → 使用 default_type 的默認模型
    """
    tokens = token_count(content)
    
    if tokens > 105_000:
        model = await model_manager.get_default_model("large_context", **kwargs)
    elif model_id:
        model = await model_manager.get_model(model_id, **kwargs)
    else:
        model = await model_manager.get_default_model(default_type, **kwargs)
    
    # 處理 Anthropic 模型的參數衝突
    if isinstance(langchain_model, ChatAnthropic):
        if langchain_model.temperature and langchain_model.top_p:
            langchain_model = langchain_model.copy(update={'top_p': None})
    
    return langchain_model
```

**模型類型**：
- `chat`: 聊天模型
- `transformation`: 轉換模型
- `tools`: 工具模型
- `embedding`: 嵌入模型
- `large_context`: 大上下文模型

#### 2. `open_notebook/domain/models.py`

**功能**：模型管理系統

**核心類**：

##### `ModelManager` - 模型管理器

```python
class ModelManager:
    async def get_model(self, model_id: str, **kwargs) -> Optional[ModelType]
    async def get_default_model(self, model_type: str, **kwargs) -> Optional[ModelType]
    async def get_chat_model(self) -> Optional[LanguageModel]
    async def get_embedding_model(self) -> Optional[EmbeddingModel]
```

**模型配置** (`DefaultModels`):
```python
class DefaultModels(RecordModel):
    default_chat_model: Optional[str]
    default_transformation_model: Optional[str]
    large_context_model: Optional[str]
    default_embedding_model: Optional[str]
    default_tools_model: Optional[str]
```

**LLM 使用場景**：

1. **決策階段** (`agent_decision`)
   - 使用 `tools` 類型模型
   - 結構化輸出（JSON）
   - Max tokens: 2000

2. **評估階段** (`evaluate_results`)
   - 階段 1：使用 `evaluation_model`（強模型）
   - 階段 2：使用 `evaluation_formatting_model`（較小模型，可選）
   - Max tokens: 2000 (階段1), 1000 (階段2)

3. **答案生成階段** (`synthesize_answer`)
   - 使用 `synthesis_model` 或 `model_id`
   - Max tokens: 4000

4. **查詢優化階段** (`refine_query`)
   - 使用 `decision_model`
   - 結構化輸出（JSON）
   - Max tokens: 1000

---

## Prompt 及提示詞組合

### Prompt 目錄結構

```
prompts/
├── agentic_ask/          # Agentic RAG 專用提示詞
│   ├── decision.jinja    # 決策提示詞
│   ├── evaluation.jinja   # 評估提示詞
│   ├── refinement.jinja  # 查詢優化提示詞
│   └── synthesis.jinja   # 答案生成提示詞
└── chat_agentic/         # 聊天 Agentic RAG 提示詞
    ├── decision.jinja
    ├── evaluation_stage1.jinja  # 兩階段評估 - 階段1
    ├── evaluation_stage2.jinja  # 兩階段評估 - 階段2
    ├── evaluation.jinja
    ├── refinement.jinja
    ├── synthesis.jinja
    └── fallback_answer.jinja
```

### 關鍵 Prompt 模板

#### 1. 決策 Prompt (`agentic_ask/decision.jinja`)

**用途**：Agent 決策下一步行動

**輸入變量**：
- `question`: 用戶問題
- `iteration_count`: 當前迭代次數
- `max_iterations`: 最大迭代次數
- `token_count`: Token 使用量
- `max_tokens`: Token 限制
- `search_history`: 搜尋歷史（最近3次）
- `collected_results`: 累積結果
- `partial_answer`: 部分答案
- `reasoning_trace`: 推理追蹤（最近3次）
- `available_tools`: 可用工具列表

**輸出格式**：
```json
{
    "action": "use_tool|evaluate|synthesize|finish",
    "tool_name": "vector_search|text_search|...",
    "parameters": {
        "query": "...",
        "limit": 10
    },
    "reasoning": "..."
}
```

#### 2. 答案生成 Prompt (`agentic_ask/synthesis.jinja`)

**用途**：從收集的結果生成最終答案

**輸入變量**：
- `question`: 用戶問題
- `collected_results`: 收集的結果列表

**要求**：
1. **準確性**：只使用收集來源中的信息
2. **完整性**：回答問題的所有方面
3. **引用**：使用格式 `[document_id]` 引用每個聲明
4. **結構**：清晰組織答案
5. **清晰度**：寫作清晰易懂

**引用格式**：
- 格式：`[document_id]`
- 只引用存在於 `collected_results` 中的來源
- 不要編造 document ID

#### 3. 兩階段評估 Prompt

##### 階段 1 (`chat_agentic/evaluation_stage1.jinja`)

**用途**：自由格式的深度評估

**設計理念**：
- 將 context 稱為 "expert advice"（專家建議）
- 將 answer 稱為 "candidate answer"（候選答案）
- 創造不對稱性，強調 context 是權威來源

**評估重點**：
1. **自我批評**：質疑每個聲明
2. **多種解釋**：考慮替代解釋
3. **差距分析**：找出缺失信息
4. **矛盾檢查**：檢查與專家建議的矛盾
5. **過度推斷**：檢查是否超出支持範圍
6. **引用驗證**：驗證所有引用

**Hallucination 檢測重點**：
- 未支持的聲明
- 矛盾信息
- 過度推斷
- 無效或缺失引用
- 未引用的重要聲明

##### 階段 2 (`chat_agentic/evaluation_stage2.jinja`)

**用途**：將自由格式評估轉換為結構化 JSON

**輸出結構**：
```json
{
    "score": 0.0-1.0,
    "reasoning": "...",
    "confidence": 0.0-1.0,
    "decision": "continue|refine_search|synthesize|reject",
    "hallucination": {
        "has_risk": boolean,
        "risk_score": 0.0-1.0,
        "unsupported_claims": [...],
        "invalid_citations": [...],
        "contradictory_info": [...],
        "over_extrapolation": [...],
        "notes": "..."
    },
    "completeness_score": 0.0-1.0,
    "relevance_score": 0.0-1.0,
    "citation_quality_score": 0.0-1.0,
    "consistency_score": 0.0-1.0,
    "completeness_notes": "...",
    "relevance_notes": "...",
    "citation_notes": "...",
    "consistency_notes": "..."
}
```

#### 4. 答案生成 Prompt (`chat_agentic/synthesis.jinja`)

**用途**：生成帶有防幻覺機制的答案

**特殊要求**：
- **防幻覺機制**：
  - 每個事實聲明必須有至少一個來源支持
  - 如果找不到信息，明確說明 "I don't have information about this in the available sources"
  - 不要推斷或推測超出明確陳述範圍的信息
  - 不要以創造新聲明的方式組合來源信息
  - 如果來源相互矛盾，提及兩種觀點

**輸出格式要求**：
- 不使用思考標籤（`<think>`）
- 答案應該是最終響應，可直接顯示給用戶
- 不使用特殊標籤或標記包裹答案

### Prompt 組合策略

#### 1. 決策流程中的 Prompt 組合

```
用戶問題
    ↓
決策 Prompt (decision.jinja)
    ↓
[使用工具] → 工具執行結果
    ↓
評估 Prompt (evaluation_stage1.jinja + evaluation_stage2.jinja)
    ↓
[生成答案] → 答案生成 Prompt (synthesis.jinja)
    ↓
Hallucination 檢測
    ↓
最終答案
```

#### 2. 兩階段評估的 Prompt 組合

```
階段 1: 自由格式評估
    ↓
evaluation_stage1.jinja
    ↓
LLM 生成深度評估文本
    ↓
階段 2: 結構化輸出
    ↓
evaluation_stage2.jinja
    ↓
LLM 轉換為 JSON 格式
    ↓
EvaluationResult 模型驗證
```

#### 3. 查詢優化 Prompt (`agentic_ask/refinement.jinja`)

**用途**：基於之前的嘗試優化搜尋查詢

**輸入**：
- `question`: 原始問題
- `search_history`: 搜尋歷史（最近5次）
- `collected_results`: 已收集的結果

**輸出**：優化後的查詢決策

---

## 相關配置與模型管理

### 環境變量配置

**Agentic RAG 配置**：
- `AGENTIC_MAX_ITERATIONS`: 最大迭代次數（默認：10）
- `AGENTIC_MAX_TOKENS`: Token 限制（默認：50000）
- `AGENTIC_MAX_DURATION`: 最大持續時間（秒，默認：300）

### 模型配置

**可配置的模型類型**：
- `decision_model`: 決策模型
- `evaluation_model`: 評估模型（階段1）
- `evaluation_formatting_model`: 評估格式化模型（階段2，可選）
- `synthesis_model`: 答案生成模型
- `model_id`: 通用模型覆蓋

**模型選擇邏輯**：
1. 如果指定了特定模型 ID → 使用該模型
2. 如果上下文 > 105K tokens → 使用大上下文模型
3. 否則 → 使用默認模型（根據類型）

### 配置傳遞

**通過 `RunnableConfig` 傳遞**：
```python
config = {
    "configurable": {
        "decision_model": "gpt-4o",
        "evaluation_model": "gpt-4o",
        "evaluation_formatting_model": "gpt-3.5-turbo",  # 可選
        "evaluation_use_two_stage": True,
        "synthesis_model": "gpt-4o",
        "model_id": "gpt-4o",  # 通用覆蓋
    }
}
```

---

## 總結

### 核心流程

1. **初始化** → 設置狀態和限制
2. **決策** → Agent 決定下一步行動（使用工具/評估/生成）
3. **執行工具** → 搜尋、計算等
4. **評估** → 規則基礎 + LLM 兩階段評估 + Hallucination 檢測
5. **生成答案** → 基於收集結果生成答案
6. **驗證** → 檢查答案質量，決定接受/拒絕

### 關鍵特性

1. **多層 Hallucination 檢測**：
   - 規則基礎檢測（快速）
   - LLM 兩階段評估（深度）
   - 綜合判定

2. **靈活的工具系統**：
   - 統一接口
   - 重試機制
   - 超時處理
   - 錯誤處理

3. **智能模型選擇**：
   - 根據上下文大小選擇模型
   - 支持模型覆蓋
   - 參數衝突處理

4. **結構化 Prompt 設計**：
   - 清晰的角色定義
   - 明確的輸入輸出格式
   - 防幻覺機制

### 相關文件清單

**核心流程**：
- `open_notebook/graphs/agentic_ask.py`

**Hallucination 檢測**：
- `open_notebook/services/evaluation_service.py`
- `open_notebook/services/evaluation_models.py`
- `docs/hallucination_detection_guidelines.md`

**Tool 系統**：
- `open_notebook/services/tool_service.py`
- `open_notebook/graphs/tools.py`

**LLM 管理**：
- `open_notebook/graphs/utils.py`
- `open_notebook/domain/models.py`

**Prompt 模板**：
- `prompts/agentic_ask/*.jinja`
- `prompts/chat_agentic/*.jinja`

---

**文檔版本**：1.0  
**最後更新**：2025-01-27  
**維護者**：RAG Notebook Team

