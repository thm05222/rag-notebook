# Frontend - RAG Notebook

## 依賴版本

### 核心框架

- **Next.js**: 15.4.7（最新穩定版）
- **React**: 19.1.0（最新穩定版）
- **React DOM**: 19.1.0

### 依賴兼容性狀態

#### 已確認兼容 React 19

以下依賴已通過 peerDependencies 檢查，確認支持 React 19：

- `react-resizable-panels` (^2.1.7): peerDependencies 包含 `^19.0.0`
- `use-debounce` (^10.0.6): peerDependencies 為 `react: "*"`，理論上兼容所有版本
- `@radix-ui/*`: 通常跟進很快，當前版本應已支持 React 19

#### 需要手動測試的組件

由於 React 19 引入了許多底層變更（例如 `useRef` 的清理函數、Context API 的變更），以下組件需要重點測試：

1. **`@uiw/react-md-editor`** (^4.0.8)
   - **風險等級**: 高
   - **原因**: Markdown 編輯器涉及大量 DOM 操作，極高機率會受到 React 19 影響
   - **測試項目**:
     - 輸入文字
     - 預覽模式切換
     - 編輯模式切換
     - 保存/加載內容

2. **Radix UI 彈出視窗組件** (`@radix-ui/react-dialog`, `@radix-ui/react-popover`)
   - **風險等級**: 中
   - **原因**: React 19 改變了 Portal 和 Ref 的行為，可能影響這些組件的掛載
   - **測試項目**:
     - Dialog 打開/關閉
     - Popover 顯示/隱藏
     - 焦點管理
     - 鍵盤導航

3. **`react-markdown`** (^10.1.0)
   - **風險等級**: 低
   - **原因**: 渲染庫，相對安全但仍需注意
   - **測試項目**:
     - Markdown 內容渲染
     - 複雜格式（表格、代碼塊等）

## 兼容性檢查

### 運行依賴檢查

```bash
npm run check-deps
```

### 運行兼容性測試

```bash
npm run test-compatibility
```

這會執行構建測試，並提示您手動測試關鍵組件。

## 升級建議

1. **升級前檢查**:
   - 運行 `npm run check-deps` 檢查所有依賴的 peerDependencies
   - 查看各依賴的 GitHub issues 和 release notes

2. **升級後測試**:
   - 執行 `npm run build` 確保構建成功
   - 手動測試上述需要重點關注的組件
   - 運行完整的應用測試套件（如果有的話）

3. **監控問題**:
   - 關注控制台錯誤和警告
   - 檢查 React DevTools 中的組件樹
   - 監控用戶報告的 UI 問題

## 測試結果

### 構建測試（自動）

- **日期**: 2025-01-XX
- **狀態**: ✅ 通過
- **結果**: `npm run build` 成功完成，無類型錯誤或編譯錯誤
- **警告**: 僅有一些未使用變數的 ESLint 警告（不影響功能）

### 手動測試（待執行）

以下組件需要手動測試以確認 React 19 兼容性：

1. **`@uiw/react-md-editor`**
   - [ ] 輸入文字功能正常
   - [ ] 預覽模式切換正常
   - [ ] 編輯模式切換正常
   - [ ] 保存/加載內容正常

2. **Radix UI Dialog**
   - [ ] Dialog 打開/關閉正常
   - [ ] 焦點管理正常
   - [ ] 鍵盤導航正常

3. **Radix UI Popover**
   - [ ] Popover 顯示/隱藏正常
   - [ ] 焦點管理正常

## 已知問題

目前沒有已知的 React 19 兼容性問題。如果發現問題，請記錄在此文件中。

## 開發環境設置

```bash
# 安裝依賴
npm install

# 開發模式
npm run dev

# 構建
npm run build

# 生產模式
npm start
```
