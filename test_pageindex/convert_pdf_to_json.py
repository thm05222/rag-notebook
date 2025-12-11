import os
import json
import sys

# 添加 pageindex 目錄到 Python 路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
pageindex_dir = os.path.join(current_dir, 'pageindex')
sys.path.insert(0, pageindex_dir)

from pageindex.page_index import *
from pageindex.utils import ConfigLoader
CHATGPT_API_KEY = h
def convert_pdf_to_json(pdf_path, output_path=None):
    """
    讀取 PDF 檔案並使用 PageIndex 轉換成 JSON
    
    Args:
        pdf_path: PDF 檔案的路徑（目前為空，需要填入）
        output_path: 輸出的 JSON 檔案路徑（可選，預設為 PDF 檔名 + _structure.json）
    """
    # TODO: 填入 PDF 檔案路徑
    if not pdf_path:
        pdf_path = r"E:\ftp_folder\圆柱绕流一：基础（翻译版） ((Zdravkovich M. M. ) 斯特兰科维奇) (Z-Library).pdf"  # 請在此填入 PDF 檔案路徑
    
    # 驗證 PDF 檔案
    if not pdf_path:
        raise ValueError("請設定 PDF 檔案路徑")
    
    if not pdf_path.lower().endswith('.pdf'):
        raise ValueError("檔案必須是 PDF 格式")
    
    if not os.path.isfile(pdf_path):
        raise ValueError(f"找不到 PDF 檔案: {pdf_path}")
    
    print(f'開始處理 PDF: {pdf_path}')
    
    # 使用 ConfigLoader 載入預設配置
    config_loader = ConfigLoader()
    
    # 設定配置選項（可根據需要調整）
    user_opt = {
        'model': 'gpt-4o-2024-11-20',
        'toc_check_page_num': 20,
        'max_page_num_each_node': 10,
        'max_token_num_each_node': 20000,
        'if_add_node_id': 'yes',
        'if_add_node_summary': 'yes',
        'if_add_doc_description': 'no',
        'if_add_node_text': 'no'
    }
    
    # 載入配置
    opt = config_loader.load(user_opt)
    
    # 處理 PDF
    print('正在解析 PDF...')
    result = page_index_main(pdf_path, opt)
    
    print('解析完成，正在儲存 JSON...')
    
    # 設定輸出檔案路徑
    if not output_path:
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_dir = os.path.dirname(__file__)
        output_path = os.path.join(output_dir, f'{pdf_name}_structure.json')
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # 儲存 JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f'JSON 已儲存至: {output_path}')
    return output_path

if __name__ == "__main__":
    # PDF 檔案路徑（請填入實際路徑）
    pdf_file_path = ""  # TODO: 填入 PDF 檔案路徑
    
    # 可選：指定輸出檔案路徑
    output_file_path = None  # 如果為 None，會自動根據 PDF 檔名產生
    
    try:
        convert_pdf_to_json(pdf_file_path, output_file_path)
    except Exception as e:
        print(f'錯誤: {e}')
        sys.exit(1)

