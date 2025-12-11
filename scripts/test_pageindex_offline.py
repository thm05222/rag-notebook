#!/usr/bin/env python3
"""
離線測試 PageIndex 功能
直接測試 PageIndex 能否讀取和處理文件
"""

import sys
import os
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_pageindex_with_file(file_path: Path):
    """測試 PageIndex 處理單個文件"""
    print(f"\n{'='*80}")
    print(f"測試文件: {file_path.name}")
    print(f"文件大小: {file_path.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*80}\n")
    
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return False
    
    try:
        # 測試 1: 檢查 PageIndex 模塊是否可用
        print("1. 檢查 PageIndex 模塊...")
        try:
            # 添加 pageindex 目錄到路徑（與 pageindex_service 相同的方式）
            pageindex_path = project_root / "pageindex"
            if str(pageindex_path) not in sys.path:
                sys.path.insert(0, str(pageindex_path))
            
            from pageindex.page_index import page_index_main  # type: ignore
            from pageindex.utils import ConfigLoader  # type: ignore
            print("   ✅ PageIndex 模塊導入成功")
        except ImportError as e:
            print(f"   ❌ PageIndex 模塊導入失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 測試 2: 檢查文件類型
        print(f"\n2. 檢查文件類型...")
        file_ext = file_path.suffix.lower()
        print(f"   文件擴展名: {file_ext}")
        
        if file_ext not in ['.pdf', '.epub', '.md', '.txt']:
            print(f"   ⚠️  不支持的文件類型: {file_ext}")
            return False
        else:
            print(f"   ✅ 支持的文件類型")
        
        # 測試 3: 嘗試讀取文件內容（簡單檢查）
        print(f"\n3. 檢查文件可讀性...")
        try:
            with open(file_path, 'rb') as f:
                first_bytes = f.read(100)
                print(f"   文件前 100 字節: {first_bytes[:50]}...")
                print(f"   ✅ 文件可讀取")
        except Exception as e:
            print(f"   ❌ 無法讀取文件: {e}")
            return False
        
        # 測試 4: 使用 page_index_main 處理文件（僅 PDF）
        if file_ext == '.pdf':
            print(f"\n4. 使用 page_index_main 處理 PDF 文件...")
            try:
                config_loader = ConfigLoader()
                default_opt = config_loader.load({
                    'model': 'gpt-4o-2024-11-20',
                    'if_add_node_id': 'yes',
                    'if_add_node_summary': 'yes',
                    'if_add_doc_description': 'no',
                    'if_add_node_text': 'no'
                })
                
                print(f"   配置已加載")
                print(f"   開始處理（這可能需要一些時間）...")
                
                result = page_index_main(str(file_path), default_opt)
                
                print(f"   ✅ 處理完成")
                print(f"   結果類型: {type(result)}")
                
                if isinstance(result, dict):
                    if 'structure' in result:
                        structure = result['structure']
                        print(f"   結構類型: {type(structure)}")
                        if isinstance(structure, list):
                            print(f"   根節點數量: {len(structure)}")
                            if structure:
                                print(f"   第一個節點鍵: {list(structure[0].keys()) if isinstance(structure[0], dict) else 'N/A'}")
                        elif isinstance(structure, dict):
                            print(f"   結構鍵: {list(structure.keys())}")
                    else:
                        print(f"   結果鍵: {list(result.keys())}")
                elif isinstance(result, list):
                    print(f"   列表長度: {len(result)}")
                    if result:
                        print(f"   第一個元素類型: {type(result[0])}")
                        if isinstance(result[0], dict):
                            print(f"   第一個元素鍵: {list(result[0].keys())}")
                
                # 檢查結構大小
                import json
                try:
                    json_str = json.dumps(result, ensure_ascii=False)
                    size_mb = len(json_str.encode('utf-8')) / 1024 / 1024
                    print(f"   結構大小: {size_mb:.2f} MB")
                except Exception as e:
                    print(f"   ⚠️  無法計算結構大小: {e}")
                
                return True
                
            except Exception as e:
                print(f"   ❌ page_index_main 處理失敗: {e}")
                import traceback
                traceback.print_exc()
                return False
        else:
            print(f"\n4. EPUB/Markdown 文件需要先轉換為文本，跳過直接處理測試")
            print(f"   ⚠️  對於 EPUB/Markdown，PageIndex 需要從 full_text 建立索引")
            return True  # 不算失敗，只是跳過
        
    except Exception as e:
        print(f"❌ 測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函數"""
    print("="*80)
    print("PageIndex 離線測試")
    print("="*80)
    
    # 獲取上傳目錄
    uploads_dir = project_root / "notebook_data" / "uploads"
    
    if not uploads_dir.exists():
        print(f"❌ 上傳目錄不存在: {uploads_dir}")
        return
    
    print(f"\n上傳目錄: {uploads_dir}")
    
    # 列出所有文件
    files = list(uploads_dir.glob("*"))
    pdf_files = [f for f in files if f.is_file() and f.suffix.lower() == '.pdf']
    epub_files = [f for f in files if f.is_file() and f.suffix.lower() == '.epub']
    
    print(f"\n找到文件:")
    print(f"  PDF: {len(pdf_files)} 個")
    print(f"  EPUB: {len(epub_files)} 個")
    
    if not pdf_files and not epub_files:
        print("❌ 沒有找到可測試的文件")
        return
    
    # 選擇一個較小的文件進行測試
    test_files = []
    if pdf_files:
        # 選擇最小的 PDF
        smallest_pdf = min(pdf_files, key=lambda f: f.stat().st_size)
        test_files.append(smallest_pdf)
        print(f"\n選擇測試文件 (PDF): {smallest_pdf.name} ({smallest_pdf.stat().st_size / 1024:.2f} KB)")
    
    if epub_files:
        # 選擇最小的 EPUB
        smallest_epub = min(epub_files, key=lambda f: f.stat().st_size)
        test_files.append(smallest_epub)
        print(f"選擇測試文件 (EPUB): {smallest_epub.name} ({smallest_epub.stat().st_size / 1024:.2f} KB)")
    
    # 測試每個文件
    results = []
    for test_file in test_files[:1]:  # 只測試第一個文件（PDF）
        result = test_pageindex_with_file(test_file)
        results.append((test_file.name, result))
    
    # 總結
    print(f"\n{'='*80}")
    print("測試總結")
    print(f"{'='*80}\n")
    for name, success in results:
        status = "✅ 成功" if success else "❌ 失敗"
        print(f"  {name}: {status}")

if __name__ == "__main__":
    main()
