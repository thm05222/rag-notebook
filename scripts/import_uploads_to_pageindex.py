#!/usr/bin/env python3
"""
批量導入 uploads 文件夾中的文件到 SurrealDB，並建立 PageIndex structure。

此腳本會：
1. 掃描 notebook_data/uploads 文件夾
2. 為每個文件創建 Source（如果不存在）
3. 提取文本內容（PDF/EPUB）
4. 建立 PageIndex structure
5. 保存到 SurrealDB

支持斷點續傳：跳過已處理的文件（已有 PageIndex structure）
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

from loguru import logger

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from open_notebook.domain.notebook import Notebook, Source
from open_notebook.services.pageindex_service import PageIndexService
from open_notebook.graphs.source import content_process
from open_notebook.database.schema_init import init_schema


async def ensure_database_initialized():
    """確保數據庫已初始化"""
    try:
        from open_notebook.database.schema_init import needs_init, init_schema
        if await needs_init():
            logger.info("Database schema not initialized, initializing...")
            await init_schema()
        else:
            logger.info("Database schema already initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def get_or_create_notebook(name: str = "價值投資知識庫") -> Notebook:
    """獲取或創建默認 notebook"""
    try:
        notebooks = await Notebook.get_all()
        for notebook in notebooks:
            if notebook.name == name:
                logger.info(f"Found existing notebook: {notebook.id}")
                return notebook
        
        # 創建新 notebook
        notebook = Notebook(name=name, description="價值投資相關文檔集合")
        await notebook.save()
        logger.info(f"Created new notebook: {notebook.id}")
        return notebook
    except Exception as e:
        logger.error(f"Failed to get or create notebook: {e}")
        raise


async def find_source_by_file_path(file_path: str) -> Optional[Source]:
    """根據文件路徑查找已存在的 Source"""
    try:
        from open_notebook.database.repository import repo_query
        
        # 查詢所有 sources，檢查 asset.file_path
        sources = await Source.get_all()
        for source in sources:
            if source.asset and source.asset.file_path == file_path:
                return source
        return None
    except Exception as e:
        logger.warning(f"Error finding source by file path: {e}")
        return None


async def process_file(
    file_path: Path,
    notebook: Notebook,
    pageindex_service: PageIndexService,
    skip_existing: bool = True
) -> bool:
    """處理單個文件：創建 Source、提取文本、建立 PageIndex"""
    file_path_str = str(file_path.absolute())
    file_name = file_path.name
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {file_name}")
    logger.info(f"Path: {file_path_str}")
    
    try:
        # 1. 檢查是否已存在 Source
        existing_source = await find_source_by_file_path(file_path_str)
        
        if existing_source:
            logger.info(f"Found existing source: {existing_source.id}")
            
            # 檢查是否已有 PageIndex structure
            if skip_existing and existing_source.pageindex_structure:
                logger.info(f"Source {existing_source.id} already has PageIndex structure, skipping...")
                return True
            
            source = existing_source
        else:
            # 2. 創建新 Source
            logger.info("Creating new source...")
            source = Source(
                title=file_name,
                asset={"file_path": file_path_str},
                full_text=None  # 將在處理後填充
            )
            await source.save()
            
            # 關聯到 notebook
            await source.relate("reference", notebook.id)
            logger.info(f"Created source: {source.id}")
        
        # 3. 如果沒有 full_text，提取文本內容
        if not source.full_text:
            logger.info("Extracting text content from file...")
            try:
                from open_notebook.graphs.source import content_process
                
                content_state = {
                    "file_path": file_path_str,
                    "document_engine": "auto",
                    "output_format": "markdown"
                }
                
                source_state = {
                    "source_id": source.id,
                    "content_state": content_state,
                    "embed": False
                }
                
                processed_state = await content_process(source_state)
                processed_content = processed_state.get("content_state", {})
                content = processed_content.get("content", "")
                
                if content:
                    source.full_text = content
                    # 更新 title 如果從內容中提取到了
                    if processed_content.get("title") and not source.title:
                        source.title = processed_content.get("title")
                    await source.save()
                    logger.info(f"Extracted {len(content)} characters of text")
                else:
                    logger.warning(f"No content extracted from {file_name}")
                    return False
            except Exception as e:
                logger.error(f"Failed to extract text from {file_name}: {e}")
                logger.exception(e)
                return False
        else:
            logger.info(f"Source already has text content ({len(source.full_text)} chars)")
        
        # 4. 建立 PageIndex structure
        logger.info("Building PageIndex structure...")
        try:
            # 使用 PageIndexService 建立索引（會自動保存到數據庫）
            structure = await pageindex_service._get_or_create_index_for_source(
                source.id,
                source,
                model_id=None  # 使用默認模型
            )
            
            if structure:
                logger.success(f"✓ Successfully built PageIndex for {file_name}")
                logger.info(f"  Structure type: {type(structure).__name__}")
                if isinstance(structure, list):
                    logger.info(f"  Root nodes: {len(structure)}")
                elif isinstance(structure, dict):
                    logger.info(f"  Structure keys: {list(structure.keys())}")
                return True
            else:
                logger.warning(f"PageIndex structure is empty for {file_name}")
                return False
        except Exception as e:
            logger.error(f"Failed to build PageIndex for {file_name}: {e}")
            logger.exception(e)
            return False
            
    except Exception as e:
        logger.error(f"Error processing {file_name}: {e}")
        logger.exception(e)
        return False


async def main():
    """主函數"""
    # 配置日誌
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("="*60)
    logger.info("PageIndex 批量導入腳本")
    logger.info("="*60)
    
    # 1. 確保數據庫已初始化
    await ensure_database_initialized()
    
    # 2. 初始化 PageIndexService
    pageindex_service = PageIndexService()
    await pageindex_service._ensure_initialized()
    
    if not pageindex_service.is_available():
        logger.error("PageIndex is not available. Please ensure PageIndex is installed.")
        logger.error("Install: git clone https://github.com/VectifyAI/PageIndex.git pageindex")
        sys.exit(1)
    
    # 3. 獲取或創建 notebook
    notebook = await get_or_create_notebook()
    
    # 4. 掃描 uploads 文件夾
    uploads_dir = project_root / "notebook_data" / "uploads"
    if not uploads_dir.exists():
        logger.error(f"Uploads directory not found: {uploads_dir}")
        sys.exit(1)
    
    logger.info(f"\nScanning directory: {uploads_dir}")
    
    # 支持的檔案類型
    supported_extensions = {".pdf", ".epub"}
    files = [
        f for f in uploads_dir.iterdir()
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]
    
    if not files:
        logger.warning("No supported files found in uploads directory")
        sys.exit(0)
    
    logger.info(f"Found {len(files)} files to process")
    
    # 5. 處理每個文件
    success_count = 0
    fail_count = 0
    skip_count = 0
    
    for i, file_path in enumerate(files, 1):
        logger.info(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
        
        try:
            result = await process_file(file_path, notebook, pageindex_service, skip_existing=True)
            if result:
                success_count += 1
            else:
                fail_count += 1
        except KeyboardInterrupt:
            logger.warning("\nInterrupted by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path.name}: {e}")
            logger.exception(e)
            fail_count += 1
    
    # 6. 顯示統計
    logger.info("\n" + "="*60)
    logger.info("處理完成")
    logger.info("="*60)
    logger.info(f"總文件數: {len(files)}")
    logger.info(f"成功: {success_count}")
    logger.info(f"失敗: {fail_count}")
    logger.info(f"跳過: {skip_count}")
    logger.info("="*60)


if __name__ == "__main__":
    asyncio.run(main())

