# PageIndex Search 修復計劃

## 概述
本計劃旨在徹底解決 `pageindex_search` 工具目前遭遇的問題，確保其功能完善和穩定運行。

## 已修復的問題
1. ✅ `get_chat_model()` 方法錯誤 → 已改為 `get_default_model("chat")`
2. ✅ 早期 `full_text` 檢查 → 已移除，允許從文件路徑讀取 PDF
3. ✅ `os` 導入衝突 → 已移除局部重複導入

## 待修復的問題

### 1. 加強 query 參數驗證
**問題**：當前驗證只檢查 `not kwargs["query"]`，但沒有檢查空字符串或僅包含空白字符的情況。

**位置**：`open_notebook/services/pageindex_service.py:950`

**修復方案**：
- 在 `validate_parameters` 方法中，添加對空字符串和僅空白字符的檢查
- 使用 `str.strip()` 驗證 query 是否為有效字符串
- 提供更明確的錯誤信息

**代碼修改**：
```python
# 驗證 query 參數
if "query" not in kwargs or not kwargs["query"]:
    error_msg = "Missing or empty 'query' parameter"
    logger.warning(f"{self.name}: {error_msg}")
    self._last_validation_error = error_msg
    return False

if not isinstance(kwargs["query"], str):
    error_msg = f"'query' must be a string, got {type(kwargs['query']).__name__}"
    logger.warning(f"{self.name}: {error_msg}")
    self._last_validation_error = error_msg
    return False

# 檢查是否為空字符串或僅包含空白字符
if not kwargs["query"].strip():
    error_msg = "'query' cannot be empty or contain only whitespace"
    logger.warning(f"{self.name}: {error_msg}")
    self._last_validation_error = error_msg
    return False
```

### 2. 改進空結果處理
**問題**：當所有 sources 都失敗時，返回空列表，沒有明確的錯誤信息，用戶無法知道原因。

**位置**：`open_notebook/services/pageindex_service.py:830-857` 和 `859-878`

**修復方案**：
- 在搜索過程中記錄失敗的 sources 數量
- 如果所有 sources 都失敗，拋出明確的異常而不是返回空列表
- 提供失敗原因摘要

**代碼修改**：
```python
elif notebook_id:
    # 方式 2: 搜索 notebook 中的所有 sources
    logger.info(f"Searching notebook: {notebook_id}")
    notebook = await Notebook.get(notebook_id)
    sources = await notebook.get_sources()
    
    if not sources:
        logger.warning(f"No sources found in notebook {notebook_id}")
        return []
    
    # 為每個 source 建立索引並搜索
    per_source_limit = max(1, limit // len(sources))
    failed_sources = []
    for source in sources:
        try:
            # 獲取完整的 source（包含 full_text）
            full_source = await Source.get(source.id)
            tree_structure = await self._get_or_create_index_for_source(
                full_source.id, full_source, model_id
            )
            results = await self._tree_search(query, tree_structure, model_id, per_source_limit)
            # 為結果添加 source 信息
            for result in results:
                result["source_id"] = full_source.id
                result["source_title"] = full_source.title
            all_results.extend(results)
        except Exception as e:
            logger.warning(f"Failed to search source {source.id}: {e}")
            failed_sources.append({"source_id": source.id, "error": str(e)})
            continue
    
    # 如果所有 sources 都失敗，拋出異常
    if len(failed_sources) == len(sources) and len(all_results) == 0:
        error_summary = "; ".join([f"{fs['source_id']}: {fs['error']}" for fs in failed_sources[:3]])
        raise ToolExecutionError(
            f"All {len(sources)} sources failed to search. "
            f"First few errors: {error_summary}. "
            f"Please check if sources have content and PageIndex structures are built."
        )
```

### 3. 添加 source_id 預驗證
**問題**：如果 source_id 不存在，會拋出異常，但沒有提前驗證，導致不必要的處理。

**位置**：`open_notebook/services/pageindex_service.py:864-878`

**修復方案**：
- 在搜索前驗證 source_id 是否存在
- 如果 source_id 無效，記錄警告並跳過，而不是拋出異常
- 在 `validate_parameters` 中也可以添加可選的預驗證（如果性能允許）

**代碼修改**：
```python
elif source_ids:
    # 方式 3: 搜索指定的 sources
    logger.info(f"Searching sources: {source_ids}")
    per_source_limit = max(1, limit // len(source_ids))
    
    failed_sources = []
    for source_id in source_ids:
        try:
            # 預驗證 source 是否存在
            try:
                source = await Source.get(source_id)
            except Exception as e:
                logger.warning(f"Source {source_id} not found: {e}")
                failed_sources.append({"source_id": source_id, "error": f"Source not found: {str(e)}"})
                continue
            
            tree_structure = await self._get_or_create_index_for_source(
                source.id, source, model_id
            )
            results = await self._tree_search(query, tree_structure, model_id, per_source_limit)
            # 為結果添加 source 信息
            for result in results:
                result["source_id"] = source.id
                result["source_title"] = source.title
            all_results.extend(results)
        except Exception as e:
            logger.warning(f"Failed to search source {source_id}: {e}")
            failed_sources.append({"source_id": source_id, "error": str(e)})
            continue
    
    # 如果所有 sources 都失敗，拋出異常
    if len(failed_sources) == len(source_ids) and len(all_results) == 0:
        error_summary = "; ".join([f"{fs['source_id']}: {fs['error']}" for fs in failed_sources[:3]])
        raise ToolExecutionError(
            f"All {len(source_ids)} sources failed to search. "
            f"First few errors: {error_summary}. "
            f"Please check if sources exist and have content."
        )
```

### 4. 改進參數注入邏輯
**問題**：如果 query 最終仍為空，參數注入後可能仍會導致驗證失敗，但沒有明確的警告。

**位置**：`open_notebook/graphs/chat.py:547-555`

**修復方案**：
- 在參數注入後，如果 query 仍然為空，記錄更明確的錯誤
- 提供建議：用戶應該提供問題或查詢

**代碼修改**：
```python
elif tool_name == "pageindex_search":
    # 確保 query 參數存在（從 question 或 parameters 中獲取）
    if "query" not in parameters or not parameters.get("query"):
        question = state.get("question", "")
        if question and question.strip():
            parameters["query"] = question.strip()
            logger.info(f"[Iteration {iteration}] PageIndex search: Using question as query: {question[:50]}...")
        else:
            error_msg = "No query parameter provided and no question in state"
            logger.error(f"[Iteration {iteration}] PageIndex search: {error_msg}")
            # 不設置 query，讓驗證邏輯處理
```

### 5. 添加 document_path 驗證
**問題**：如果 document_path 無效或文件不存在，會在後續處理中才發現，沒有提前驗證。

**位置**：`open_notebook/services/pageindex_service.py:997-1010`

**修復方案**：
- 在 `validate_parameters` 中，如果提供了 `document_path`，驗證文件是否存在
- 提供明確的錯誤信息

**代碼修改**：
```python
# 驗證 document_path 參數（如果提供）
if "document_path" in kwargs and kwargs.get("document_path"):
    doc_path = kwargs["document_path"]
    if not isinstance(doc_path, str):
        error_msg = f"'document_path' must be a string, got {type(doc_path).__name__}"
        logger.warning(f"{self.name}: {error_msg}")
        self._last_validation_error = error_msg
        return False
    
    # 驗證文件是否存在（同步檢查，因為這是驗證階段）
    if not os.path.exists(doc_path):
        error_msg = f"Document file not found: {doc_path}"
        logger.warning(f"{self.name}: {error_msg}")
        self._last_validation_error = error_msg
        return False
    
    # 驗證文件擴展名（僅支持 PDF）
    if not doc_path.lower().endswith('.pdf'):
        error_msg = f"Document path must be a PDF file, got: {doc_path}"
        logger.warning(f"{self.name}: {error_msg}")
        self._last_validation_error = error_msg
        return False
```

## 實施順序
1. 加強 query 參數驗證（最關鍵，影響所有搜索）
2. 改進空結果處理（提升用戶體驗）
3. 添加 source_id 預驗證（提升錯誤處理）
4. 改進參數注入邏輯（提升調試能力）
5. 添加 document_path 驗證（完整性檢查）

## 測試建議
1. 測試空字符串 query 的情況
2. 測試僅空白字符的 query
3. 測試所有 sources 都失敗的情況
4. 測試無效 source_id 的情況
5. 測試無效 document_path 的情況
6. 測試正常搜索流程，確保沒有回歸

## 預期效果
- 更明確的錯誤信息，幫助用戶理解問題
- 更好的錯誤處理，避免靜默失敗
- 更早的驗證，減少不必要的處理
- 更穩定的搜索功能



