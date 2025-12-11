#!/bin/bash

# === 可調整參數 ===
SRC_DIR="/home/qiyoo/rag-notebook"
ARCHIVE_NAME="rag-notebook_$(date +%Y%m%d_%H%M%S).tar.gz"

# Windows 目標位置（需事先建立資料夾）
WIN_USER="thm052"
WIN_IP="140.114.129.25"
WIN_DEST_PATH="/e/rag-notebook"

# === 壓縮 ===
echo "==> 壓縮資料夾中: $SRC_DIR"
tar -czvf $ARCHIVE_NAME $SRC_DIR

# === 傳送到 Windows ===
echo "==> 傳送到 Windows: $WIN_USER@$WIN_IP:$WIN_DEST_PATH"
scp $ARCHIVE_NAME $WIN_USER@$WIN_IP:$WIN_DEST_PATH

# === 結果 ===
if [ $? -eq 0 ]; then
    echo "==> 傳送成功：$ARCHIVE_NAME 已送到 Windows 的 $WIN_DEST_PATH"
else
    echo "==> 傳送失敗"
fi
