# PageIndex Structure 儲存機制說明

## 當前儲存方式

### 1. 內存緩存（Memory Cache）

PageIndex structure **目前只儲存在內存中**，使用 Python 字典作為緩存：

```python
# 在 PageIndexService.__init__ 中
self._index_cache: Dict[str, Dict[str, Any]] = {}
```

**儲存位置**：
- **類型**：Python 字典（Dict）
- **Key**：`source_id`（例如：`"source:abc123"`）或 `"file:/path/to/file.pdf"`
- **Value**：PageIndex tree structure（字典或列表格式的樹狀結構）

**儲存流程**：

```python
# 1. 建立索引
index = await self._build_index_for_source(source_id, source, model_id)

# 2. 存入緩存
self._index_cache[source_id] = index

# 3. 後續使用時直接從緩存讀取
if source_id in self._index_cache:
    return self._index_cache[source_id]
```

### 2. 緩存結構

Tree structure 的格式取決於來源：

#### 對於 Markdown/Text Sources（使用 `md_to_tree`）：
```python
# 返回格式：列表或字典
tree_result = await md_to_tree(...)
# 可能是：
# - List[Dict]：樹狀節點列表
# - Dict with 'structure' key：包含結構的字典
```

#### 對於 PDF Documents（使用 `page_index_main`）：
```python
# 返回格式：
{
    'doc_name': 'document.pdf',
    'structure': [...],  # 樹狀結構列表
    'doc_description': '...'  # 可選
}
```

### 3. 緩存 Key 格式

- **Source-based**：`source_id`（例如：`"source:abc123"`）
- **File-based**：`"file:/path/to/document.pdf"`

## 當前問題

### ❌ 沒有持久化儲存

**問題**：
1. **服務重啟後緩存丟失**：所有 PageIndex structure 需要重新建立
2. **內存限制**：大量 sources 會佔用大量內存
3. **無法跨進程共享**：多個服務實例無法共享緩存
4. **建立成本高**：每次重啟都需要重新調用 LLM 建立索引（耗時且昂貴）

### 當前行為

```python
# 服務啟動時
self._index_cache = {}  # 空緩存

# 第一次使用時
index = await self._build_index_for_source(...)  # 需要重新建立（調用 LLM）
self._index_cache[source_id] = index  # 存入內存

# 服務重啟後
self._index_cache = {}  # 緩存丟失，需要重新建立
```

## Tree Structure 數據格式

### 節點結構示例

```python
{
    "title": "章節標題",
    "start_index": 0,      # 起始字符位置
    "end_index": 1000,     # 結束字符位置
    "node_id": "1.1",      # 節點 ID（如果啟用）
    "summary": "...",      # 節點摘要（如果啟用）
    "nodes": [             # 子節點（可選）
        {
            "title": "子章節",
            "start_index": 100,
            "end_index": 500,
            ...
        }
    ]
}
```

### 完整樹結構

```python
[
    {
        "title": "第一章",
        "start_index": 0,
        "end_index": 5000,
        "node_id": "1",
        "summary": "第一章的摘要...",
        "nodes": [
            {
                "title": "1.1 節",
                "start_index": 0,
                "end_index": 2000,
                "node_id": "1.1",
                "summary": "1.1 節的摘要..."
            }
        ]
    },
    {
        "title": "第二章",
        ...
    }
]
```

## 建議的改進方案

### 方案 1：儲存到 SurrealDB（推薦）

將 PageIndex structure 儲存到 SurrealDB，與 Source 關聯：

```python
# 在 Source 模型中添加字段
class Source(ObjectModel):
    pageindex_structure: Optional[Dict[str, Any]] = None  # PageIndex tree structure
    pageindex_built_at: Optional[datetime] = None  # 建立時間
```

**優點**：
- ✅ 持久化儲存，服務重啟後仍可用
- ✅ 與 Source 數據關聯，易於管理
- ✅ 可以追蹤建立時間和版本
- ✅ 支持查詢和更新

**實現**：
```python
async def _save_index_to_database(self, source_id: str, structure: Dict[str, Any]):
    """將索引儲存到數據庫"""
    source = await Source.get(source_id)
    if source:
        source.pageindex_structure = structure
        source.pageindex_built_at = datetime.now()
        await source.save()

async def _load_index_from_database(self, source_id: str) -> Optional[Dict[str, Any]]:
    """從數據庫載入索引"""
    source = await Source.get(source_id)
    if source and source.pageindex_structure:
        return source.pageindex_structure
    return None
```

### 方案 2：儲存到文件系統

將結構儲存為 JSON 文件：

```python
PAGEINDEX_CACHE_DIR = Path("data/pageindex_cache")

async def _save_index_to_file(self, source_id: str, structure: Dict[str, Any]):
    """將索引儲存到文件"""
    cache_file = PAGEINDEX_CACHE_DIR / f"{source_id.replace(':', '_')}.json"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(structure, f, ensure_ascii=False, indent=2)

async def _load_index_from_file(self, source_id: str) -> Optional[Dict[str, Any]]:
    """從文件載入索引"""
    cache_file = PAGEINDEX_CACHE_DIR / f"{source_id.replace(':', '_')}.json"
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None
```

**優點**：
- ✅ 簡單易實現
- ✅ 可以手動檢查和編輯
- ✅ 不依賴數據庫結構

**缺點**：
- ❌ 需要管理文件系統
- ❌ 與 Source 數據分離

### 方案 3：混合方案（內存 + 持久化）

結合內存緩存和持久化儲存：

```python
async def _get_or_create_index_for_source(self, source_id: str, source: Source, model_id: Optional[str] = None):
    """獲取或創建索引（帶持久化）"""
    # 1. 檢查內存緩存
    if source_id in self._index_cache:
        return self._index_cache[source_id]
    
    # 2. 檢查數據庫/文件
    cached_index = await self._load_index_from_database(source_id)
    if cached_index:
        self._index_cache[source_id] = cached_index  # 載入到內存
        return cached_index
    
    # 3. 建立新索引
    index = await self._build_index_for_source(source_id, source, model_id)
    
    # 4. 存入內存和持久化
    self._index_cache[source_id] = index
    await self._save_index_to_database(source_id, index)
    
    return index
```

## 當前代碼位置

- **緩存定義**：`open_notebook/services/pageindex_service.py:30`
- **緩存使用**：`open_notebook/services/pageindex_service.py:84-122`
- **索引建立**：`open_notebook/services/pageindex_service.py:124-207`

## 總結

**當前狀態**：
- ✅ 內存緩存正常工作
- ❌ 沒有持久化儲存
- ❌ 服務重啟後需要重新建立索引

**建議**：
實施**方案 1（SurrealDB 儲存）**或**方案 3（混合方案）**，以實現持久化儲存並避免重複建立索引的成本。

