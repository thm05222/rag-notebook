"""
PageIndex local integration service.
Provides reasoning-based hierarchical search for long documents.
"""

import asyncio
import json
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from pydantic import BaseModel, Field, field_validator, model_validator

from open_notebook.domain.notebook import Notebook, Source
from open_notebook.exceptions import ToolNotFoundError
from open_notebook.services.tool_service import BaseTool, ToolExecutionError
from open_notebook.domain.models import model_manager

# PageIndex structure 類型定義
PageIndexStructure = Union[List[Dict[str, Any]], Dict[str, Any]]


class PageIndexService:
    """Service for PageIndex local integration."""

    def __init__(self):
        self.initialized = False
        self.pageindex_module = None
        self.pageindex_path: Optional[Path] = None
        # 索引緩存：source_id -> tree structure
        self._index_cache: Dict[str, PageIndexStructure] = {}
        # 索引建立狀態：source_id -> asyncio.Task
        self._index_building_tasks: Dict[str, asyncio.Task] = {}
        # 並發控制鎖：source_id -> asyncio.Lock
        self._building_locks: Dict[str, asyncio.Lock] = {}

    async def _ensure_initialized(self) -> None:
        """Ensure PageIndex is initialized."""
        if self.initialized:
            return

        # Get PageIndex path from environment or default
        submodule_path = os.getenv("PAGEINDEX_SUBMODULE_PATH", "pageindex")
        self.pageindex_path = Path(__file__).parent.parent.parent / submodule_path

        if not self.pageindex_path.exists():
            logger.warning(
                f"PageIndex not found at {self.pageindex_path}. "
                "PageIndex tools will be disabled. "
                "To enable: git clone https://github.com/VectifyAI/PageIndex.git pageindex"
            )
            return

        # Dynamic import
        try:
            # Add to path
            if str(self.pageindex_path) not in sys.path:
                sys.path.insert(0, str(self.pageindex_path))

            # Import PageIndex modules
            try:
                from pageindex import page_index, md_to_tree  # type: ignore
                from pageindex.utils import ConfigLoader  # type: ignore
                self.pageindex_module = {
                    "page_index": page_index,
                    "md_to_tree": md_to_tree,
                    "ConfigLoader": ConfigLoader,
                }
            except ImportError as e:
                logger.warning(
                    f"Could not import PageIndex from {self.pageindex_path}: {e}. "
                    "Please ensure PageIndex dependencies are installed."
                )
                return

            self.initialized = True
            logger.info("PageIndex initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize PageIndex: {e}")
            logger.exception(e)
            self.initialized = False

    def is_available(self) -> bool:
        """Check if PageIndex is available."""
        return self.initialized

    def clear_cache_for_source(self, source_id: str) -> None:
        """清除指定 source 的緩存"""
        if source_id in self._index_cache:
            del self._index_cache[source_id]
            logger.debug(f"Cleared PageIndex cache for source {source_id}")
        
        # 清理相關的任務和鎖
        if source_id in self._index_building_tasks:
            task = self._index_building_tasks[source_id]
            if not task.done():
                task.cancel()
            del self._index_building_tasks[source_id]
        
        if source_id in self._building_locks:
            del self._building_locks[source_id]

    def clear_all_cache(self) -> None:
        """清除所有緩存"""
        cache_size = len(self._index_cache)
        self._index_cache.clear()
        logger.info(f"Cleared all PageIndex cache ({cache_size} entries)")

    async def cleanup_orphaned_cache(self) -> None:
        """清理孤立的緩存（對應的 source 已不存在）"""
        try:
            from open_notebook.domain.notebook import Source
            
            orphaned_keys = []
            for cache_key in list(self._index_cache.keys()):
                # 如果是 source_id 格式
                if not cache_key.startswith("file:"):
                    try:
                        source = await Source.get(cache_key)
                        if not source:
                            orphaned_keys.append(cache_key)
                    except Exception:
                        orphaned_keys.append(cache_key)
            
            for key in orphaned_keys:
                if key in self._index_cache:
                    del self._index_cache[key]
                    logger.debug(f"Removed orphaned cache entry: {key}")
            
            if orphaned_keys:
                logger.info(f"Cleaned up {len(orphaned_keys)} orphaned cache entries")
        except Exception as e:
            logger.warning(f"Failed to cleanup orphaned cache: {e}")

    def _normalize_structure(self, structure: Any) -> Optional[PageIndexStructure]:
        """標準化 PageIndex structure 格式，確保返回 List[Dict] 或 Dict"""
        if structure is None:
            logger.warning("Structure is None")
            return None
        
        # 如果是字典且包含 'structure' key，提取 structure
        original_structure = structure
        if isinstance(structure, dict) and 'structure' in structure:
            structure = structure['structure']
        
        # 驗證格式
        if isinstance(structure, list):
            # 驗證列表中的每個元素都是字典
            if len(structure) == 0:
                logger.warning(
                    f"Structure is an empty list. "
                    f"Original structure type: {type(original_structure).__name__}, "
                    f"value: {str(original_structure)[:200]}. "
                    f"This may indicate that the document does not have a hierarchical structure "
                    f"suitable for PageIndex (e.g., no headings or sections)."
                )
                return None
            if all(isinstance(item, dict) for item in structure):
                return structure
            else:
                # 找出非字典項目的類型和位置
                non_dict_items = [(i, type(item).__name__) for i, item in enumerate(structure) if not isinstance(item, dict)]
                logger.warning(f"Invalid structure format: list contains non-dict items at indices {non_dict_items[:5]}")
                return None
        elif isinstance(structure, dict):
            return structure
        else:
            logger.warning(f"Invalid structure type: {type(structure).__name__}, expected list or dict. Value preview: {str(structure)[:200]}")
            return None

    def _validate_structure(self, structure: PageIndexStructure) -> bool:
        """驗證 PageIndex structure 的基本格式"""
        if structure is None:
            return False
        
        if isinstance(structure, list):
            if len(structure) == 0:
                return False
            # 檢查列表中的每個節點是否至少包含 title
            for node in structure:
                if not isinstance(node, dict) or 'title' not in node:
                    return False
            return True
        elif isinstance(structure, dict):
            # 字典格式應該至少包含 title 或 structure key
            if 'title' in structure or 'structure' in structure:
                return True
            # 或者是一個節點字典
            return True
        return False

    def _truncate_structure_smartly(self, structure: PageIndexStructure, max_chars: int) -> PageIndexStructure:
        """智能截斷 structure，保留最重要的部分（根節點和部分子節點）"""
        if isinstance(structure, list):
            # 對於列表格式，保留前幾個根節點
            truncated = []
            current_size = 0
            for node in structure:
                node_json = json.dumps(node, ensure_ascii=False)
                node_size = len(node_json)
                if current_size + node_size > max_chars * 0.8:  # 保留 20% 空間
                    break
                # 如果節點有子節點，只保留部分
                if 'nodes' in node and isinstance(node['nodes'], list):
                    node_copy = node.copy()
                    # 只保留前 3 個子節點
                    node_copy['nodes'] = node['nodes'][:3]
                    truncated.append(node_copy)
                else:
                    truncated.append(node)
                current_size += node_size
            return truncated if truncated else structure[:1]  # 至少保留第一個節點
        elif isinstance(structure, dict):
            # 對於字典格式，保留主要字段
            truncated = {}
            for key in ['title', 'node_id', 'summary']:
                if key in structure:
                    truncated[key] = structure[key]
            # 如果有 nodes，只保留部分
            if 'nodes' in structure and isinstance(structure['nodes'], list):
                truncated['nodes'] = structure['nodes'][:3]
            return truncated
        return structure

    def _check_version_compatibility(self, version: Optional[str]) -> bool:
        """檢查 PageIndex 版本是否兼容"""
        if not version:
            return False
        
        # 當前支持的版本
        CURRENT_VERSION = "1.0"
        SUPPORTED_VERSIONS = ["1.0"]
        
        return version in SUPPORTED_VERSIONS

    async def _load_index_from_database(self, source_id: str) -> Optional[PageIndexStructure]:
        """從數據庫載入 PageIndex structure"""
        try:
            source = await Source.get(source_id)
            if source and source.pageindex_structure:
                logger.debug(f"Found PageIndex structure in database for source {source_id}")
                
                # 檢查版本兼容性
                if not self._check_version_compatibility(source.pageindex_version):
                    logger.warning(
                        f"PageIndex version {source.pageindex_version} for source {source_id} "
                        f"is not compatible, will rebuild"
                    )
                    return None
                
                # 標準化結構格式
                structure = self._normalize_structure(source.pageindex_structure)
                if structure and self._validate_structure(structure):
                    return structure
                else:
                    logger.warning(f"Invalid PageIndex structure format for source {source_id}, will rebuild")
                    return None
            return None
        except Exception as e:
            logger.warning(f"Error loading PageIndex from database for source {source_id}: {e}")
            return None

    def _check_structure_size(self, structure: PageIndexStructure) -> tuple[bool, int]:
        """檢查 structure 的大小（JSON 序列化後的字節數）"""
        try:
            json_str = json.dumps(structure, ensure_ascii=False)
            size_bytes = len(json_str.encode('utf-8'))
            max_size = 10 * 1024 * 1024  # 10MB
            return size_bytes <= max_size, size_bytes
        except Exception as e:
            logger.warning(f"Failed to check structure size: {e}")
            return False, 0

    async def _save_index_to_database(
        self, source_id: str, structure: PageIndexStructure, model_id: Optional[str] = None
    ) -> None:
        """將 PageIndex structure 保存到數據庫"""
        try:
            from datetime import datetime
            
            # 驗證結構格式
            normalized = self._normalize_structure(structure)
            if not normalized:
                raise ValueError(f"Invalid structure format for source {source_id}")
            
            if not self._validate_structure(normalized):
                raise ValueError(f"Structure validation failed for source {source_id}")
            
            # 檢查大小
            size_ok, size_bytes = self._check_structure_size(normalized)
            if not size_ok:
                logger.warning(
                    f"PageIndex structure for source {source_id} is too large "
                    f"({size_bytes / 1024 / 1024:.2f}MB), but saving anyway"
                )
            
            source = await Source.get(source_id)
            if not source:
                logger.warning(f"Source {source_id} not found, cannot save PageIndex")
                return
            
            # 更新 PageIndex 相關字段
            source.pageindex_structure = normalized
            source.pageindex_built_at = datetime.now()
            source.pageindex_model = model_id or "gpt-4o-2024-11-20"  # 默認模型
            source.pageindex_version = "1.0"  # 當前版本
            
            await source.save()
            logger.info(
                f"Saved PageIndex structure to database for source {source_id} "
                f"(size: {size_bytes / 1024:.2f}KB)"
            )
        except Exception as e:
            logger.error(f"Error saving PageIndex to database for source {source_id}: {e}")
            logger.exception(e)
            raise

    async def _get_index_for_source(
        self, source_id: str
    ) -> Optional[PageIndexStructure]:
        """
        只讀方法：獲取已存在的 PageIndex 結構（不會自動建立）
        
        用於搜索操作 - PageIndex 應該由用戶預先建立，
        搜索時不應該觸發自動建立。
        
        Returns:
            PageIndexStructure if found, None if not exists
        """
        # 1. 檢查內存緩存
        if source_id in self._index_cache:
            logger.debug(f"PageIndex cache hit for source {source_id}")
            return self._index_cache[source_id]
        
        # 2. 從數據庫載入
        try:
            cached_structure = await self._load_index_from_database(source_id)
            if cached_structure:
                logger.info(f"Loaded PageIndex from database for source {source_id}")
                self._index_cache[source_id] = cached_structure
                return cached_structure
        except Exception as e:
            logger.warning(f"Failed to load PageIndex from database for source {source_id}: {e}")
        
        # 3. 不存在則返回 None（不自動建立）
        return None

    async def _get_or_create_index_for_source(
        self, source_id: str, source: Source, model_id: Optional[str] = None
    ) -> PageIndexStructure:
        """
        為 source 獲取或創建 PageIndex 樹狀結構（優先從數據庫讀取）
        
        注意：此方法會在找不到索引時自動建立，僅用於明確要求建立索引的場景。
        對於搜索操作，請使用 _get_index_for_source()。
        """
        # 1. 檢查內存緩存
        if source_id in self._index_cache:
            logger.debug(f"PageIndex cache hit for source {source_id}")
            return self._index_cache[source_id]

        # 2. 獲取或創建鎖
        if source_id not in self._building_locks:
            self._building_locks[source_id] = asyncio.Lock()
        lock = self._building_locks[source_id]

        # 3. 使用鎖確保並發安全
        async with lock:
            # 再次檢查緩存（可能在等待鎖時已經建立）
            if source_id in self._index_cache:
                logger.debug(f"PageIndex cache hit for source {source_id} (after lock)")
                return self._index_cache[source_id]

            # 4. 檢查數據庫中是否已有索引
            try:
                cached_structure = await self._load_index_from_database(source_id)
                if cached_structure:
                    # 檢查模型是否變更（如果提供 model_id）
                    if model_id:
                        source = await Source.get(source_id)
                        if source and source.pageindex_model and source.pageindex_model != model_id:
                            logger.info(
                                f"Model changed for source {source_id} "
                                f"(old: {source.pageindex_model}, new: {model_id}), will rebuild"
                            )
                            # 清除舊索引，重新建立
                            cached_structure = None
                    
                    if cached_structure:
                        logger.info(f"Loaded PageIndex from database for source {source_id}")
                        self._index_cache[source_id] = cached_structure
                        return cached_structure
            except Exception as e:
                logger.warning(f"Failed to load PageIndex from database for source {source_id}: {e}")
                # 繼續嘗試建立新索引

            # 5. 檢查是否正在建立索引（在鎖內檢查）
            if source_id in self._index_building_tasks:
                task = self._index_building_tasks[source_id]
                if not task.done():
                    logger.info(f"Waiting for index building for source {source_id}...")
                    try:
                        await task
                        if source_id in self._index_cache:
                            return self._index_cache[source_id]
                    except Exception as e:
                        logger.error(f"Index building failed for source {source_id}: {e}")
                        # 清理失敗的任務
                        if source_id in self._index_building_tasks:
                            del self._index_building_tasks[source_id]
                        raise

            # 6. 開始建立新索引
            logger.info(f"Building new PageIndex for source {source_id}...")
            task = asyncio.create_task(
                self._build_index_for_source(source_id, source, model_id)
            )
            self._index_building_tasks[source_id] = task

            try:
                index = await task
                # 存入內存緩存
                self._index_cache[source_id] = index
                # 保存到數據庫
                try:
                    await self._save_index_to_database(source_id, index, model_id)
                except Exception as e:
                    logger.warning(f"Failed to save PageIndex to database for source {source_id}: {e}")
                    # 不影響返回結果，索引已存入內存緩存
                return index
            except Exception as e:
                logger.error(f"Failed to build index for source {source_id}: {e}")
                raise
            finally:
                # 清理任務和鎖
                if source_id in self._index_building_tasks:
                    del self._index_building_tasks[source_id]
                # 保留鎖，以便後續使用（避免重複創建）

    async def _build_index_for_source(
        self, source_id: str, source: Source, model_id: Optional[str] = None, max_retries: int = 2
    ) -> PageIndexStructure:
        """為 source 建立 PageIndex 樹狀結構（帶重試機制）"""
        # 注意：不在此處檢查 full_text，因為 _build_index_for_source_once 會處理
        # 從文件路徑讀取的情況（對於 PDF 文件）
        
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await self._build_index_for_source_once(source_id, source, model_id)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 2  # 指數退避：2s, 4s
                    logger.warning(
                        f"Failed to build PageIndex for source {source_id} (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Failed to build PageIndex for source {source_id} after {max_retries + 1} attempts")
                    raise ToolExecutionError(f"Failed to build PageIndex after {max_retries + 1} attempts: {str(e)}") from e
        
        # 這不應該到達，但為了類型檢查
        raise ToolExecutionError(f"Failed to build PageIndex: {str(last_error)}") from last_error

    async def _build_index_for_source_once(
        self, source_id: str, source: Source, model_id: Optional[str] = None
    ) -> PageIndexStructure:
        """為 source 建立 PageIndex 樹狀結構（單次嘗試）"""
        # 如果沒有 full_text，嘗試直接從文件路徑讀取（僅 PDF）
        if not source.full_text:
            logger.info(f"Source {source_id} has no full_text, attempting to read from file path")
            
            # 檢查是否有文件路徑
            if not source.asset or not source.asset.file_path:
                raise ValueError(
                    f"Source {source_id} has no content (full_text is None) and no file path. "
                    f"Please process the source first to extract text content."
                )
            
            file_path = source.asset.file_path
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # 只支持 PDF 文件直接讀取
            if file_ext != '.pdf':
                raise ValueError(
                    f"Source {source_id} has no full_text and file is not PDF ({file_ext}). "
                    f"Please process the source first to extract text content."
                )
            
            # 解析文件路徑
            # 獲取 UPLOADS_FOLDER（避免循環導入）
            # 注意：os 已在文件頂部導入，無需重複導入
            try:
                from open_notebook.config import UPLOADS_FOLDER
            except ImportError:
                # 如果無法導入，使用默認路徑
                project_root = Path(__file__).parent.parent.parent
                UPLOADS_FOLDER = str(project_root / "notebook_data" / "uploads")
            
            safe_root = os.path.realpath(UPLOADS_FOLDER)
            
            # 嘗試解析文件路徑
            resolved_path = None
            if os.path.isabs(file_path):
                resolved_path = os.path.realpath(file_path)
            else:
                # 嘗試多種路徑組合
                combined_path = os.path.join(UPLOADS_FOLDER, file_path)
                if os.path.exists(combined_path):
                    resolved_path = os.path.realpath(combined_path)
                else:
                    clean_path = file_path.replace("uploads/", "").replace("uploads\\", "")
                    if clean_path != file_path:
                        combined_path = os.path.join(UPLOADS_FOLDER, clean_path)
                        if os.path.exists(combined_path):
                            resolved_path = os.path.realpath(combined_path)
            
            if not resolved_path or not resolved_path.startswith(safe_root):
                raise ValueError(
                    f"Source {source_id} file path cannot be resolved or is outside safe directory: {file_path}"
                )
            
            if not os.path.exists(resolved_path):
                raise ValueError(f"Source {source_id} file not found: {resolved_path}")
            
            logger.info(f"Reading PDF directly from file path: {resolved_path}")
            # 使用 _load_document_index 直接從文件讀取
            return await self._load_document_index(resolved_path, model_id)

        # 獲取模型配置
        if not model_id:
            try:
                chat_model = await model_manager.get_default_model("chat")
                if chat_model:
                    # 嘗試從 model 提取 OpenAI 模型名稱
                    # 檢查是否是 OpenAI 模型
                    model_name = getattr(chat_model, "model_name", None) or str(chat_model)
                    # 如果是 OpenAI 模型，提取模型名稱
                    if "gpt" in model_name.lower():
                        model_id = model_name
                    else:
                        # 默認使用 gpt-4o
                        model_id = "gpt-4o-2024-11-20"
                else:
                    model_id = "gpt-4o-2024-11-20"
            except Exception as e:
                logger.warning(f"Failed to get default chat model: {e}, using default gpt-4o-2024-11-20")
                model_id = "gpt-4o-2024-11-20"

        # 確保 OpenAI API key 被設置
        # PageIndex 使用 CHATGPT_API_KEY 環境變數
        if not os.getenv("CHATGPT_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            # 嘗試從 model_manager 獲取 API key
            try:
                chat_model = await model_manager.get_default_model("chat")
                if chat_model and hasattr(chat_model, "openai_api_key"):
                    api_key = chat_model.openai_api_key
                    if api_key:
                        os.environ["CHATGPT_API_KEY"] = api_key
                        logger.info("Set CHATGPT_API_KEY from model_manager")
            except Exception:
                pass

        # 使用 md_to_tree 從文本建立樹狀結構
        # 將 full_text 寫入臨時文件或直接使用
        try:
            from pageindex.page_index_md import md_to_tree  # type: ignore

            # 創建臨時 markdown 文件
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as f:
                f.write(source.full_text)
                temp_md_path = f.name

            try:
                # 檢查內容長度，如果太短可能無法建立有效的樹狀結構
                content_length = len(source.full_text) if source.full_text else 0
                if content_length < 100:
                    logger.warning(
                        f"Source {source_id} has very short content ({content_length} chars), "
                        f"PageIndex may not be able to build a meaningful structure. "
                        f"Consider using regular text search instead."
                    )
                
                # 建立樹狀結構
                # md_to_tree 返回 tree_structure（列表）
                tree_result = await md_to_tree(
                    md_path=temp_md_path,
                    if_thinning=False,
                    if_add_node_summary='yes',
                    summary_token_threshold=200,
                    model=model_id,
                    if_add_doc_description='no',
                    if_add_node_text='no',
                    if_add_node_id='yes'
                )

                # md_to_tree 返回 tree_structure（列表或字典）
                # 記錄 tree_result 的類型以便調試
                logger.debug(f"md_to_tree returned type: {type(tree_result)}, value preview: {str(tree_result)[:200] if tree_result else 'None'}")
                
                # 標準化和驗證結構
                normalized = self._normalize_structure(tree_result)
                if not normalized:
                    # 檢查是否是空結構的情況
                    structure_value = None
                    if isinstance(tree_result, dict) and 'structure' in tree_result:
                        structure_value = tree_result['structure']
                    
                    if isinstance(structure_value, list) and len(structure_value) == 0:
                        # 空結構 - 文檔可能沒有適合 PageIndex 的層次結構
                        raise ToolExecutionError(
                            f"PageIndex could not build a structure for source {source_id}. "
                            f"The document may not have a hierarchical structure suitable for PageIndex "
                            f"(e.g., no headings, sections, or chapters). "
                            f"Content length: {content_length} chars. "
                            f"PageIndex works best with structured documents like financial reports, "
                            f"legal documents, academic papers, or technical manuals with clear headings. "
                            f"Consider using vector search or text search instead for this document."
                        )
                    else:
                        # 其他標準化失敗的情況
                        error_details = f"tree_result type: {type(tree_result)}, value: {str(tree_result)[:500] if tree_result else 'None'}"
                        logger.error(f"Failed to normalize PageIndex structure for source {source_id}. {error_details}")
                        raise ToolExecutionError(
                            f"Failed to normalize PageIndex structure for source {source_id}. "
                            f"Received type: {type(tree_result).__name__}, "
                            f"expected list or dict. This may indicate an issue with md_to_tree output. "
                            f"Content length: {content_length} chars. "
                            f"Please check if the source content is sufficient for PageIndex processing."
                        )
                
                if not self._validate_structure(normalized):
                    raise ToolExecutionError(f"PageIndex structure validation failed for source {source_id}")
                
                return normalized

            finally:
                # 清理臨時文件
                try:
                    os.unlink(temp_md_path)
                except Exception:
                    pass

        except Exception as e:
            logger.error(f"Failed to build PageIndex for source {source_id}: {e}")
            logger.exception(e)
            raise ToolExecutionError(f"Failed to build PageIndex: {str(e)}") from e

    async def _load_document_index(self, document_path: str, model_id: Optional[str] = None) -> PageIndexStructure:
        """從文件路徑載入或建立 PageIndex"""
        # 檢查緩存（使用文件路徑作為 key）
        cache_key = f"file:{document_path}"
        if cache_key in self._index_cache:
            return self._index_cache[cache_key]

        # 獲取模型配置
        if not model_id:
            try:
                chat_model = await model_manager.get_default_model("chat")
                if chat_model:
                    model_name = getattr(chat_model, "model_name", None) or str(chat_model)
                    if "gpt" in model_name.lower():
                        model_id = model_name
                    else:
                        model_id = "gpt-4o-2024-11-20"
                else:
                    model_id = "gpt-4o-2024-11-20"
            except Exception as e:
                logger.warning(f"Failed to get default chat model: {e}, using default gpt-4o-2024-11-20")
                model_id = "gpt-4o-2024-11-20"

        # 確保 OpenAI API key 被設置
        if not os.getenv("CHATGPT_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            try:
                chat_model = await model_manager.get_default_model("chat")
                if chat_model and hasattr(chat_model, "openai_api_key"):
                    api_key = chat_model.openai_api_key
                    if api_key:
                        os.environ["CHATGPT_API_KEY"] = api_key
                        logger.info("Set CHATGPT_API_KEY from model_manager")
            except Exception:
                pass

        try:
            from pageindex import page_index  # type: ignore
            from pageindex.utils import ConfigLoader  # type: ignore

            config_loader = ConfigLoader()
            default_opt = config_loader.load({
                'model': model_id,
                'if_add_node_id': 'yes',
                'if_add_node_summary': 'yes',
                'if_add_doc_description': 'no',
                'if_add_node_text': 'no'
            })

            # 建立索引（在線程池中執行，因為 page_index_main 是同步的）
            from pageindex.page_index import page_index_main  # type: ignore
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, page_index_main, document_path, default_opt)
            
            # 提取 structure
            if isinstance(result, dict) and 'structure' in result:
                structure = result['structure']
            elif isinstance(result, list):
                structure = result
            elif isinstance(result, dict):
                structure = result
            else:
                logger.warning(f"Unexpected result type: {type(result)}")
                structure = result

            # 標準化和驗證結構
            normalized = self._normalize_structure(structure)
            if not normalized or not self._validate_structure(normalized):
                logger.error(f"Invalid structure format from document {document_path}")
                raise ToolExecutionError(f"Invalid PageIndex structure format")

            # 緩存結果
            self._index_cache[cache_key] = normalized
            return normalized

        except Exception as e:
            logger.error(f"Failed to load PageIndex for document {document_path}: {e}")
            logger.exception(e)
            raise ToolExecutionError(f"Failed to load PageIndex: {str(e)}") from e

    async def _tree_search(
        self, query: str, tree_structure: PageIndexStructure, model_id: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """使用 LLM 進行 tree search，找到相關的節點"""
        # 標準化結構格式
        normalized = self._normalize_structure(tree_structure)
        if not normalized:
            logger.error("Invalid tree structure format for search")
            return []
        
        if not self._validate_structure(normalized):
            logger.error("Tree structure validation failed")
            return []
        
        # 獲取模型配置
        if not model_id:
            chat_model = await model_manager.get_chat_model()
            if chat_model:
                model_name = getattr(chat_model, "model_name", None) or str(chat_model)
                if "gpt" in model_name.lower():
                    model_id = model_name
                else:
                    model_id = "gpt-4o-2024-11-20"
            else:
                model_id = "gpt-4o-2024-11-20"

        # 確保 OpenAI API key 被設置
        if not os.getenv("CHATGPT_API_KEY") and not os.getenv("OPENAI_API_KEY"):
            try:
                chat_model = await model_manager.get_chat_model()
                if chat_model and hasattr(chat_model, "openai_api_key"):
                    api_key = chat_model.openai_api_key
                    if api_key:
                        os.environ["CHATGPT_API_KEY"] = api_key
            except Exception:
                pass

        # 構建 tree search prompt
        # 限制 tree_json 的大小，避免超過 token 限制
        tree_json = json.dumps(normalized, ensure_ascii=False, indent=2)
        # 如果 tree_json 太長，截斷它（更精確的計算）
        max_tree_length = 50000  # 大約的字符數限制
        if len(tree_json) > max_tree_length:
            logger.warning(
                f"Tree structure too large ({len(tree_json)} chars, "
                f"{len(tree_json.encode('utf-8')) / 1024:.2f}KB), truncating..."
            )
            # 嘗試智能截斷：保留根節點和部分子節點
            try:
                truncated = self._truncate_structure_smartly(normalized, max_tree_length)
                tree_json = json.dumps(truncated, ensure_ascii=False, indent=2)
                if len(tree_json) > max_tree_length:
                    tree_json = tree_json[:max_tree_length] + "... (truncated)"
            except Exception as e:
                logger.warning(f"Failed to smart truncate structure: {e}, using simple truncation")
                tree_json = tree_json[:max_tree_length] + "... (truncated)"
        
        prompt = f"""You are given a query and the tree structure of a document.
You need to find all nodes that are likely to contain the answer.

Query: {query}

Document tree structure: {tree_json}

Reply in the following JSON format:
{{
    "thinking": <your reasoning about which nodes are relevant>,
    "node_list": [node_id1, node_id2, ...]
}}

Directly return the final JSON structure. Do not output anything else."""

        try:
            # 使用 OpenAI API（PageIndex 使用的方式）
            from pageindex.utils import ChatGPT_API_async, extract_json  # type: ignore
            
            response = await ChatGPT_API_async(model=model_id, prompt=prompt)
            result = extract_json(response)
            
            node_ids = result.get("node_list", [])
            if not node_ids:
                logger.warning(f"Tree search returned no node IDs for query: {query}")
                return []

            # 根據 node_ids 提取節點內容
            nodes = self._extract_nodes_by_ids(tree_structure, node_ids)
            
            # 限制結果數量
            return nodes[:limit]

        except Exception as e:
            logger.error(f"Tree search failed: {e}")
            logger.exception(e)
            # 降級：返回所有葉節點
            return self._get_leaf_nodes(tree_structure)[:limit]

    def _extract_nodes_by_ids(self, tree_structure: PageIndexStructure, node_ids: List[str]) -> List[Dict[str, Any]]:
        """從樹狀結構中提取指定 node_id 的節點"""
        nodes = []
        
        def traverse(node: Any):
            if isinstance(node, dict):
                node_id = node.get("node_id")
                if node_id and node_id in node_ids:
                    nodes.append(node)
                # 遞歸搜索子節點
                if "nodes" in node and isinstance(node["nodes"], list):
                    for child in node["nodes"]:
                        traverse(child)
            elif isinstance(node, list):
                for item in node:
                    traverse(item)

        traverse(tree_structure)
        return nodes

    def _get_leaf_nodes(self, tree_structure: PageIndexStructure) -> List[Dict[str, Any]]:
        """獲取所有葉節點"""
        leaf_nodes = []
        
        def traverse(node: Any):
            if isinstance(node, dict):
                if "nodes" not in node or not node.get("nodes") or not isinstance(node["nodes"], list) or len(node["nodes"]) == 0:
                    # 葉節點
                    leaf_nodes.append(node)
                else:
                    # 遞歸搜索子節點
                    for child in node["nodes"]:
                        traverse(child)
            elif isinstance(node, list):
                for item in node:
                    traverse(item)

        traverse(tree_structure)
        return leaf_nodes

    def _merge_and_rank_results(self, all_results: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
        """合併和排序結果"""
        # 簡單的去重（基於 node_id）
        seen_ids = set()
        unique_results = []
        for result in all_results:
            node_id = result.get("node_id") or result.get("id", "")
            if node_id and node_id not in seen_ids:
                seen_ids.add(node_id)
                unique_results.append(result)

        # 按相關性排序（這裡簡單地保持原有順序，實際可以根據 summary 相關性排序）
        return unique_results[:limit]

    async def search(
        self, query: str, document_path: Optional[str] = None, notebook_id: Optional[str] = None,
        source_ids: Optional[List[str]] = None, limit: int = 10, model_id: Optional[str] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Execute PageIndex search.
        
        Supports three modes:
        1. document_path: Search in a specific document file
        2. notebook_id: Search in all sources of a notebook
        3. source_ids: Search in specific sources
        """
        await self._ensure_initialized()
        
        if not self.initialized or not self.pageindex_module:
            raise ToolNotFoundError(
                "PageIndex is not available. "
                "Please ensure PageIndex is installed: git clone https://github.com/VectifyAI/PageIndex.git pageindex"
            )

        if not query:
            raise ValueError("Query cannot be empty")

        all_results = []

        try:
            if document_path:
                # 方式 1: 搜索特定文檔
                logger.info(f"Searching document: {document_path}")
                tree_structure = await self._load_document_index(document_path, model_id)
                results = await self._tree_search(query, tree_structure, model_id, limit)
                all_results.extend(results)

            elif notebook_id:
                # 方式 2: 搜索 notebook 中的所有 sources
                logger.info(f"Searching notebook: {notebook_id}")
                notebook = await Notebook.get(notebook_id)
                sources = await notebook.get_sources()
                
                if not sources:
                    logger.warning(f"No sources found in notebook {notebook_id}")
                    return []

                # 為每個 source 搜索（只使用已存在的 PageIndex，不自動建立）
                per_source_limit = max(1, limit // len(sources))
                failed_sources = []
                skipped_sources = []  # 沒有 PageIndex 的 sources
                for source in sources:
                    try:
                        # 獲取完整的 source
                        full_source = await Source.get(source.id)
                        
                        # 只讀取已存在的 PageIndex，不自動建立
                        tree_structure = await self._get_index_for_source(full_source.id)
                        if not tree_structure:
                            logger.info(f"Source {full_source.id} does not have PageIndex, skipping")
                            skipped_sources.append(full_source.id)
                            continue
                        
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
                
                # 記錄跳過的 sources
                if skipped_sources:
                    logger.info(
                        f"Skipped {len(skipped_sources)} sources without PageIndex: {skipped_sources[:5]}..."
                        if len(skipped_sources) > 5 else 
                        f"Skipped {len(skipped_sources)} sources without PageIndex: {skipped_sources}"
                    )
                
                # 如果所有 sources 都沒有 PageIndex 或失敗，拋出異常
                if len(failed_sources) + len(skipped_sources) == len(sources) and len(all_results) == 0:
                    if skipped_sources:
                        raise ToolExecutionError(
                            f"None of the {len(sources)} sources have PageIndex structure. "
                            f"Please build PageIndex for sources first before using pageindex_search. "
                            f"Consider using vector_search as an alternative."
                        )
                    else:
                    error_summary = "; ".join([f"{fs['source_id']}: {fs['error']}" for fs in failed_sources[:3]])
                    raise ToolExecutionError(
                        f"All {len(sources)} sources failed to search. "
                        f"First few errors: {error_summary}. "
                        f"Please check if sources have content and PageIndex structures are built."
                    )

            elif source_ids:
                # 方式 3: 搜索指定的 sources（只使用已存在的 PageIndex，不自動建立）
                logger.info(f"Searching sources: {source_ids}")
                per_source_limit = max(1, limit // len(source_ids))
                
                failed_sources = []
                skipped_sources = []  # 沒有 PageIndex 的 sources
                for source_id in source_ids:
                    try:
                        # 預驗證 source 是否存在
                        try:
                            source = await Source.get(source_id)
                        except Exception as e:
                            logger.warning(f"Source {source_id} not found: {e}")
                            failed_sources.append({"source_id": source_id, "error": f"Source not found: {str(e)}"})
                            continue
                        
                        # 只讀取已存在的 PageIndex，不自動建立
                        tree_structure = await self._get_index_for_source(source.id)
                        if not tree_structure:
                            logger.info(f"Source {source.id} does not have PageIndex, skipping")
                            skipped_sources.append(source.id)
                            continue
                        
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
                
                # 記錄跳過的 sources
                if skipped_sources:
                    logger.info(
                        f"Skipped {len(skipped_sources)} sources without PageIndex: {skipped_sources[:5]}..."
                        if len(skipped_sources) > 5 else 
                        f"Skipped {len(skipped_sources)} sources without PageIndex: {skipped_sources}"
                    )
                
                # 如果所有 sources 都沒有 PageIndex 或失敗，拋出異常
                if len(failed_sources) + len(skipped_sources) == len(source_ids) and len(all_results) == 0:
                    if skipped_sources:
                        raise ToolExecutionError(
                            f"None of the {len(source_ids)} specified sources have PageIndex structure. "
                            f"Please build PageIndex for sources first before using pageindex_search. "
                            f"Consider using vector_search as an alternative."
                        )
                    else:
                    error_summary = "; ".join([f"{fs['source_id']}: {fs['error']}" for fs in failed_sources[:3]])
                    raise ToolExecutionError(
                        f"All {len(source_ids)} sources failed to search. "
                        f"First few errors: {error_summary}. "
                        f"Please check if sources exist and have content."
                    )
            else:
                raise ValueError("Must provide document_path, notebook_id, or source_ids")

            # 合併、排序、限制結果
            return self._merge_and_rank_results(all_results, limit)

        except Exception as e:
            logger.error(f"PageIndex search failed: {e}")
            logger.exception(e)
            raise ToolExecutionError(f"PageIndex search failed: {str(e)}") from e


def convert_pageindex_result(pageindex_result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert PageIndex result to unified format."""
    return {
        "id": f"pageindex:{pageindex_result.get('node_id', 'unknown')}",
        "title": pageindex_result.get("title", ""),
        "content": pageindex_result.get("summary", "") or pageindex_result.get("text", ""),
        "similarity": 1.0,  # PageIndex returns relevant results
        "source": "pageindex",
        "metadata": {
            "node_id": pageindex_result.get("node_id"),
            "start_index": pageindex_result.get("start_index"),
            "end_index": pageindex_result.get("end_index"),
            "pages": pageindex_result.get("pages", []),
            "source_id": pageindex_result.get("source_id"),
            "source_title": pageindex_result.get("source_title"),
        },
    }


class PageIndexSearchParameters(BaseModel):
    """Parameters for PageIndex search tool."""
    query: str = Field(..., description="Search query")
    document_path: Optional[str] = Field(default=None, description="Optional: Path to specific document file (PDF) to search")
    notebook_id: Optional[str] = Field(default=None, description="Optional: Search all sources in this notebook")
    source_ids: Optional[List[str]] = Field(default=None, description="Optional: List of specific source IDs to search")
    limit: int = Field(default=10, ge=1, description="Maximum number of results")
    
    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or contain only whitespace")
        return v.strip()
    
    @field_validator("source_ids")
    @classmethod
    def validate_source_ids(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Validate source_ids is not empty if provided."""
        if v is not None and len(v) == 0:
            raise ValueError("source_ids cannot be an empty list")
        return v
    
    @field_validator("document_path")
    @classmethod
    def validate_document_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate document_path if provided."""
        if v:
            if not v.lower().endswith('.pdf'):
                raise ValueError("Document path must be a PDF file")
            if not os.path.exists(v):
                raise ValueError(f"Document file not found: {v}")
        return v
    
    @model_validator(mode="after")
    def validate_at_least_one_target(self):
        """Validate that at least one search target is provided."""
        has_document_path = bool(self.document_path)
        has_notebook_id = bool(self.notebook_id)
        has_source_ids = bool(self.source_ids and len(self.source_ids) > 0)
        
        if not (has_document_path or has_notebook_id or has_source_ids):
            raise ValueError(
                "Must provide at least one of: document_path, notebook_id, or source_ids"
            )
        return self


class PageIndexSearchTool(BaseTool):
    """PageIndex search tool wrapper."""

    def __init__(self):
        super().__init__(
            name="pageindex_search",
            description="Reasoning-based hierarchical search for long professional documents. "
            "Best for financial reports, legal documents, academic papers, and technical manuals. "
            "Uses document structure and LLM reasoning for precise retrieval. "
            "Supports searching in specific documents or all sources in a notebook.",
            timeout=300.0,  # 增加 timeout，因為建立索引可能需要時間
            parameter_model=PageIndexSearchParameters,
        )
        self.service = PageIndexService()
    
    async def execute_with_retry(self, **kwargs) -> Dict[str, Any]:
        """Execute tool with retry logic and detailed error messages."""
        if not self.enabled:
            return {
                "tool_name": self.name,
                "success": False,
                "data": None,
                "error": f"Tool {self.name} is disabled",
                "execution_time": 0.0,
                "metadata": {},
            }

        # Validate parameters using Pydantic model
        if not self.validate_parameters(**kwargs):
            # Get detailed validation error from Pydantic
            error_msg = f"Invalid parameters for tool {self.name}"
            if self.parameter_model:
                try:
                    self.parameter_model.model_validate(kwargs)
                except Exception as e:
                    if hasattr(e, "errors"):
                        error_details = [f"{err['loc']}: {err['msg']}" for err in e.errors()]
                        error_msg = f"Invalid parameters for tool {self.name}: {', '.join(error_details)}"
                    else:
                        error_msg = f"Invalid parameters for tool {self.name}: {str(e)}"
            
            # 詳細記錄參數驗證失敗的信息
            logger.error(
                f"{self.name}: Parameter validation failed. "
                f"Error: {error_msg}, "
                f"Provided parameters: {list(kwargs.keys())}, "
                f"Parameter values: {dict((k, str(v)[:200] if not isinstance(v, (str, int, float, bool)) else v) for k, v in kwargs.items())}"
            )
            return {
                "tool_name": self.name,
                "success": False,
                "data": None,
                "error": error_msg,
                "execution_time": 0.0,
                "metadata": {
                    "validation_error": error_msg,
                    "provided_parameters": {k: str(v)[:100] if not isinstance(v, (str, int, float, bool)) else v 
                                          for k, v in kwargs.items()}
                },
            }
        
        # 調用父類的 execute_with_retry 來處理重試邏輯
        return await super().execute_with_retry(**kwargs)

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute PageIndex search."""
        # 關鍵修復：在執行前確保 PageIndex 已初始化
        await self.service._ensure_initialized()
        
        if not self.service.is_available():
            return {
                "tool_name": self.name,
                "success": False,
                "data": None,
                "error": "PageIndex is not available. Please ensure PageIndex is installed.",
                "execution_time": 0.0,
                "metadata": {},
            }

        query = kwargs.get("query", "")
        document_path = kwargs.get("document_path")
        notebook_id = kwargs.get("notebook_id")
        source_ids = kwargs.get("source_ids")
        limit = kwargs.get("limit", 10)

        try:
            results = await self.service.search(
                query=query,
                document_path=document_path,
                notebook_id=notebook_id,
                source_ids=source_ids,
                limit=limit
            )
            
            # Convert to unified format
            formatted_results = [convert_pageindex_result(r) for r in results]

            return {
                "tool_name": self.name,
                "success": True,
                "data": formatted_results,
                "error": None,
                "execution_time": 0.0,  # Will be set by execute_with_retry
                "metadata": {
                    "result_count": len(formatted_results),
                    "query": query,
                    "search_type": "pageindex",
                    "document_path": document_path,
                    "notebook_id": notebook_id,
                    "source_ids": source_ids,
                },
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"{self.name} failed: {error_msg}")
            return {
                "tool_name": self.name,
                "success": False,
                "data": [],
                "error": error_msg,
                "execution_time": 0.0,
                "metadata": {
                    "error_type": type(e).__name__,
                    "error_message": error_msg,
                    "query": query,
                },
                "error_details": {
                    "reason": "PageIndex search failed",
                    "suggestion": "Please check if the document/sources are available and have content",
                }
            }


# Global service instance
pageindex_service = PageIndexService()
