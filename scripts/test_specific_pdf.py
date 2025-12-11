#!/usr/bin/env python3
"""
測試 PageIndex 對特定 PDF 文件的處理能力
"""

import sys
import os
from pathlib import Path

# 添加項目根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_pageindex_with_pdf(pdf_path: str):
    """測試 PageIndex 處理指定 PDF 文件"""
    print(f"測試文件: {pdf_path}")
    print("=" * 80)
    
    # 檢查文件是否存在
    if not os.path.exists(pdf_path):
        print(f"❌ 文件不存在: {pdf_path}")
        return False
    
    file_size = os.path.getsize(pdf_path)
    print(f"文件大小: {file_size / 1024:.2f} KB")
    print()
    
    # 1. 檢查 PageIndex 模塊導入
    print("1. 檢查 PageIndex 模塊導入...")
    try:
        # 添加 pageindex 目錄到路徑（與 pageindex_service 相同的方式）
        pageindex_path = project_root / "pageindex"
        if str(pageindex_path) not in sys.path:
            sys.path.insert(0, str(pageindex_path))
            print(f"   已添加 pageindex 路徑: {pageindex_path}")
        
        from pageindex.page_index import page_index_main  # type: ignore
        from pageindex.utils import ConfigLoader  # type: ignore
        print("   ✅ PageIndex 模塊導入成功")
    except ImportError as e:
        print(f"   ❌ PageIndex 模塊導入失敗: {e}")
        print(f"   嘗試的路徑: {pageindex_path}")
        print(f"   pageindex 目錄存在: {pageindex_path.exists()}")
        if pageindex_path.exists():
            print(f"   pageindex 目錄內容: {list(pageindex_path.iterdir())[:10]}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. 配置 PageIndex
    print("\n2. 配置 PageIndex...")
    try:
        config_loader = ConfigLoader()
        default_opt = config_loader.load({
            'model': 'gpt-4o-2024-11-20',
            'if_add_node_id': 'yes',
            'if_add_node_summary': 'yes',
            'if_add_doc_description': 'no',
            'if_add_node_text': 'no'
        })
        print(f"   ✅ 配置成功")
        print(f"   模型: {default_opt.get('model', 'N/A')}")
    except Exception as e:
        print(f"   ❌ 配置失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 處理 PDF 文件
    print("\n3. 開始處理 PDF 文件（這可能需要一些時間）...")
    try:
        result = page_index_main(str(pdf_path), default_opt)
        
        print(f"   ✅ 處理完成")
        print(f"   結果類型: {type(result)}")
        
        # 4. 分析結果結構
        print("\n4. 分析結果結構...")
        
        if result is None:
            print("   ❌ 結果為 None")
            return False
        
        # 檢查結果格式
        if isinstance(result, dict):
            print("   ✅ 結果是字典格式")
            print(f"   字典鍵: {list(result.keys())}")
            
            # 如果有 structure 鍵
            if 'structure' in result:
                structure = result['structure']
                print(f"   structure 類型: {type(structure)}")
                if isinstance(structure, list):
                    print(f"   structure 長度: {len(structure)}")
                    if len(structure) > 0:
                        print(f"   第一個節點類型: {type(structure[0])}")
                        if isinstance(structure[0], dict):
                            print(f"   第一個節點鍵: {list(structure[0].keys())[:10]}")
                elif isinstance(structure, dict):
                    print(f"   structure 鍵: {list(structure.keys())[:10]}")
            
            # 打印結構摘要
            print("\n   結構摘要:")
            print_structure_summary(result, indent="   ")
            
        elif isinstance(result, list):
            print("   ✅ 結果是列表格式")
            print(f"   列表長度: {len(result)}")
            if len(result) > 0:
                print(f"   第一個元素類型: {type(result[0])}")
                if isinstance(result[0], dict):
                    print(f"   第一個元素鍵: {list(result[0].keys())[:10]}")
            
            print("\n   結構摘要:")
            print_structure_summary(result, indent="   ")
        else:
            print(f"   ⚠️  結果是未知類型: {type(result)}")
            print(f"   結果預覽: {str(result)[:500]}")
        
        # 5. 驗證結構有效性
        print("\n5. 驗證結構有效性...")
        is_valid = validate_structure(result)
        if is_valid:
            print("   ✅ 結構驗證通過")
        else:
            print("   ❌ 結構驗證失敗")
        
        return is_valid
        
    except Exception as e:
        print(f"   ❌ page_index_main 處理失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def print_structure_summary(structure, indent="", max_depth=3, current_depth=0):
    """遞歸打印結構摘要"""
    if current_depth >= max_depth:
        print(f"{indent}... (已達最大深度)")
        return
    
    if isinstance(structure, dict):
        for key, value in list(structure.items())[:5]:  # 只顯示前5個鍵
            if isinstance(value, (dict, list)):
                print(f"{indent}{key}: {type(value).__name__}")
                if isinstance(value, list) and len(value) > 0:
                    print(f"{indent}  [長度: {len(value)}]")
                    if isinstance(value[0], dict):
                        print(f"{indent}  第一個元素鍵: {list(value[0].keys())[:5]}")
                print_structure_summary(value, indent + "  ", max_depth, current_depth + 1)
            else:
                value_str = str(value)[:50] if value else "None"
                print(f"{indent}{key}: {value_str}")
        if len(structure) > 5:
            print(f"{indent}... (還有 {len(structure) - 5} 個鍵)")
    elif isinstance(structure, list):
        print(f"{indent}列表長度: {len(structure)}")
        if len(structure) > 0:
            print(f"{indent}第一個元素類型: {type(structure[0]).__name__}")
            if isinstance(structure[0], dict):
                print(f"{indent}第一個元素鍵: {list(structure[0].keys())[:5]}")
                print_structure_summary(structure[0], indent + "  ", max_depth, current_depth + 1)
            elif isinstance(structure[0], list):
                print(f"{indent}第一個元素是列表，長度: {len(structure[0])}")
        if len(structure) > 1:
            print(f"{indent}... (還有 {len(structure) - 1} 個元素)")

def validate_structure(structure) -> bool:
    """驗證結構是否有效"""
    if structure is None:
        return False
    
    # 提取實際的 structure
    if isinstance(structure, dict) and 'structure' in structure:
        structure = structure['structure']
    
    if isinstance(structure, list):
        if len(structure) == 0:
            return False
        # 檢查列表中的每個元素是否為字典
        return all(isinstance(item, dict) for item in structure)
    elif isinstance(structure, dict):
        return True
    else:
        return False

if __name__ == "__main__":
    # 測試指定的 PDF 文件
    pdf_file = "notebook_data/uploads/1 价值投资RAG知识库核心资料清单.pdf"
    
    # 如果提供了命令行參數，使用參數
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    
    # 轉換為絕對路徑
    pdf_path = os.path.abspath(pdf_file)
    
    success = test_pageindex_with_pdf(pdf_path)
    
    print("\n" + "=" * 80)
    if success:
        print("✅ 測試通過：PageIndex 成功讀取並生成了結構")
    else:
        print("❌ 測試失敗：PageIndex 無法正確處理文件")
    print("=" * 80)
    
    sys.exit(0 if success else 1)

