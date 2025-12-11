#!/bin/bash
# 硬體資源檢查腳本 - 評估是否適合使用 Ollama 部署本地模型

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 格式化字節
format_bytes() {
    local bytes=$1
    if [ $bytes -lt 1024 ]; then
        echo "${bytes} B"
    elif [ $bytes -lt 1048576 ]; then
        echo "$((bytes / 1024)) KB"
    elif [ $bytes -lt 1073741824 ]; then
        echo "$((bytes / 1048576)) MB"
    elif [ $bytes -lt 1099511627776 ]; then
        echo "$((bytes / 1073741824)) GB"
    else
        echo "$((bytes / 1099511627776)) TB"
    fi
}

# 檢查 CPU
check_cpu() {
    echo ""
    echo "============================================================"
    echo "CPU 信息"
    echo "============================================================"
    
    if [ -f /proc/cpuinfo ]; then
        # 物理核心數
        PHYSICAL_CORES=$(grep -c "^processor" /proc/cpuinfo 2>/dev/null || echo "無法獲取")
        LOGICAL_CORES=$(nproc 2>/dev/null || echo "無法獲取")
        
        # CPU 型號
        CPU_MODEL=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^[ \t]*//' || echo "無法獲取")
        
        # CPU 頻率
        CPU_FREQ=$(grep -m1 "cpu MHz" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | sed 's/^[ \t]*//' || echo "無法獲取")
        if [ "$CPU_FREQ" != "無法獲取" ]; then
            CPU_FREQ="${CPU_FREQ} MHz"
        fi
        
        echo "  物理核心數: $PHYSICAL_CORES"
        echo "  邏輯核心數: $LOGICAL_CORES"
        echo "  CPU 型號: $CPU_MODEL"
        echo "  CPU 頻率: $CPU_FREQ"
        
        # 保存變數供後續使用
        export PHYSICAL_CORES LOGICAL_CORES
    else
        echo "  ⚠️  無法讀取 CPU 信息"
    fi
}

# 檢查內存
check_memory() {
    echo ""
    echo "============================================================"
    echo "內存信息"
    echo "============================================================"
    
    if [ -f /proc/meminfo ]; then
        # 總內存 (KB)
        TOTAL_MEM_KB=$(grep "^MemTotal:" /proc/meminfo | awk '{print $2}')
        TOTAL_MEM_BYTES=$((TOTAL_MEM_KB * 1024))
        
        # 可用內存 (KB)
        AVAIL_MEM_KB=$(grep "^MemAvailable:" /proc/meminfo | awk '{print $2}')
        if [ -z "$AVAIL_MEM_KB" ]; then
            # 如果沒有 MemAvailable，計算 MemFree + Buffers + Cached
            MEM_FREE=$(grep "^MemFree:" /proc/meminfo | awk '{print $2}')
            BUFFERS=$(grep "^Buffers:" /proc/meminfo | awk '{print $2}')
            CACHED=$(grep "^Cached:" /proc/meminfo | awk '{print $2}')
            AVAIL_MEM_KB=$((MEM_FREE + BUFFERS + CACHED))
        fi
        AVAIL_MEM_BYTES=$((AVAIL_MEM_KB * 1024))
        
        # 已使用內存
        USED_MEM_BYTES=$((TOTAL_MEM_BYTES - AVAIL_MEM_BYTES))
        
        # 使用率
        MEM_USAGE_PERCENT=$((USED_MEM_BYTES * 100 / TOTAL_MEM_BYTES))
        
        # 交換空間
        SWAP_TOTAL_KB=$(grep "^SwapTotal:" /proc/meminfo | awk '{print $2}')
        SWAP_FREE_KB=$(grep "^SwapFree:" /proc/meminfo | awk '{print $2}')
        SWAP_USED_KB=$((SWAP_TOTAL_KB - SWAP_FREE_KB))
        if [ $SWAP_TOTAL_KB -gt 0 ]; then
            SWAP_USAGE_PERCENT=$((SWAP_USED_KB * 100 / SWAP_TOTAL_KB))
        else
            SWAP_USAGE_PERCENT=0
        fi
        
        echo "  總內存: $(format_bytes $TOTAL_MEM_BYTES)"
        echo "  可用內存: $(format_bytes $AVAIL_MEM_BYTES)"
        echo "  已使用內存: $(format_bytes $USED_MEM_BYTES)"
        echo "  內存使用率: ${MEM_USAGE_PERCENT}%"
        echo "  交換空間總量: $(format_bytes $((SWAP_TOTAL_KB * 1024)))"
        echo "  交換空間使用率: ${SWAP_USAGE_PERCENT}%"
        
        # 保存變數供後續使用
        export TOTAL_MEM_GB=$((TOTAL_MEM_KB / 1048576))
        export AVAIL_MEM_GB=$((AVAIL_MEM_KB / 1048576))
    else
        echo "  ⚠️  無法讀取內存信息"
    fi
}

# 檢查 GPU
check_gpu() {
    echo ""
    echo "============================================================"
    echo "GPU 信息"
    echo "============================================================"
    
    GPU_AVAILABLE=false
    GPU_COUNT=0
    
    # 檢查 NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=true
        GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
        
        if [ $GPU_COUNT -gt 0 ]; then
            echo "  檢測到 $GPU_COUNT 個 NVIDIA GPU:"
            nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits 2>/dev/null | while IFS=',' read -r index name total free; do
                index=$(echo $index | xargs)
                name=$(echo $name | xargs)
                total=$(echo $total | xargs)
                free=$(echo $free | xargs)
                total_gb=$((total / 1024))
                free_gb=$((free / 1024))
                echo "    GPU $index:"
                echo "      型號: $name"
                echo "      總顯存: ${total_gb} GB"
                echo "      可用顯存: ${free_gb} GB"
            done
        fi
    fi
    
    # 檢查 AMD GPU (ROCm)
    if command -v rocm-smi &> /dev/null; then
        GPU_AVAILABLE=true
        echo "  檢測到 AMD GPU (ROCm)"
        rocm-smi --showid --showmeminfo vram 2>/dev/null | head -20
    fi
    
    if [ "$GPU_AVAILABLE" = false ]; then
        echo "  ${YELLOW}⚠️  未檢測到 GPU 或 GPU 驅動未安裝${NC}"
        echo "     建議: 安裝 NVIDIA 驅動和 CUDA，或使用 CPU 模式"
    fi
    
    export GPU_AVAILABLE GPU_COUNT
}

# 檢查磁盤空間
check_disk() {
    echo ""
    echo "============================================================"
    echo "磁盤空間"
    echo "============================================================"
    
    df -h / | tail -1 | awk '{print "  掛載點: " $6 "\n  總空間: " $2 "\n  已使用: " $3 "\n  可用空間: " $4 "\n  使用率: " $5}'
    
    # 獲取可用空間 (GB)
    FREE_SPACE_GB=$(df -BG / | tail -1 | awk '{print $4}' | sed 's/G//')
    export FREE_SPACE_GB
}

# 檢查 Ollama 部署建議
check_ollama_requirements() {
    echo ""
    echo "============================================================"
    echo "Ollama 部署建議"
    echo "============================================================"
    
    recommendations=()
    warnings=()
    suitable_models=()
    score=0
    
    # 內存評估
    if [ -n "$TOTAL_MEM_GB" ]; then
        if [ $TOTAL_MEM_GB -ge 32 ]; then
            recommendations+=("${GREEN}✅ 內存充足 (≥32GB)，可以運行大型模型${NC}")
            suitable_models+=("llama3.2:70b (量化版)")
            suitable_models+=("qwen2.5:72b (量化版)")
            suitable_models+=("deepseek-r1:67b (量化版)")
            suitable_models+=("mistral:7b")
            suitable_models+=("llama3.1:8b")
            score=$((score + 2))
        elif [ $TOTAL_MEM_GB -ge 16 ]; then
            recommendations+=("${YELLOW}⚠️  內存中等 (16-32GB)，建議使用量化模型${NC}")
            suitable_models+=("llama3.2:13b (量化版)")
            suitable_models+=("qwen2.5:32b (量化版)")
            suitable_models+=("mistral:7b")
            suitable_models+=("llama3.1:8b")
            warnings+=("${YELLOW}⚠️  大型模型 (70B+) 可能無法運行或速度很慢${NC}")
            score=$((score + 1))
        elif [ $TOTAL_MEM_GB -ge 8 ]; then
            recommendations+=("${YELLOW}⚠️  內存較少 (8-16GB)，只能運行小型模型${NC}")
            suitable_models+=("llama3.2:3b")
            suitable_models+=("mistral:7b (量化版)")
            suitable_models+=("phi3:mini")
            warnings+=("${YELLOW}⚠️  不建議運行超過 13B 的模型${NC}")
            score=$((score + 1))
        else
            recommendations+=("${RED}❌ 內存不足 (<8GB)，不建議運行本地模型${NC}")
            warnings+=("${RED}❌ 建議至少 8GB 內存才能運行小型模型${NC}")
        fi
    fi
    
    # CPU 評估
    if [ -n "$LOGICAL_CORES" ]; then
        if [ $LOGICAL_CORES -ge 16 ]; then
            recommendations+=("${GREEN}✅ CPU 核心數充足 ($LOGICAL_CORES 核心)，CPU 推理速度可接受${NC}")
            score=$((score + 1))
        elif [ $LOGICAL_CORES -ge 8 ]; then
            recommendations+=("${YELLOW}⚠️  CPU 核心數中等 ($LOGICAL_CORES 核心)，CPU 推理速度較慢${NC}")
            warnings+=("${YELLOW}⚠️  建議使用 GPU 加速，或使用較小的模型${NC}")
        else
            recommendations+=("${YELLOW}⚠️  CPU 核心數較少 ($LOGICAL_CORES 核心)，CPU 推理速度很慢${NC}")
            warnings+=("${YELLOW}⚠️  強烈建議使用 GPU 加速${NC}")
        fi
    fi
    
    # GPU 評估
    if [ "$GPU_AVAILABLE" = true ] && [ $GPU_COUNT -gt 0 ]; then
        recommendations+=("${GREEN}✅ 檢測到 GPU，可以使用 GPU 加速推理${NC}")
        score=$((score + 2))
        
        # 獲取 GPU 顯存信息
        if command -v nvidia-smi &> /dev/null; then
            nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | while read -r total; do
                total_gb=$((total / 1024))
                if [ $total_gb -ge 24 ]; then
                    recommendations+=("${GREEN}✅ GPU 顯存充足 (≥24GB)，可以運行大型模型${NC}")
                    suitable_models+=("llama3.2:70b")
                    suitable_models+=("qwen2.5:72b")
                    suitable_models+=("deepseek-r1:67b")
                elif [ $total_gb -ge 16 ]; then
                    recommendations+=("${YELLOW}⚠️  GPU 顯存中等 (16-24GB)，建議使用量化模型${NC}")
                    suitable_models+=("llama3.2:70b (量化版)")
                    suitable_models+=("qwen2.5:32b")
                    suitable_models+=("deepseek-r1:67b (量化版)")
                elif [ $total_gb -ge 8 ]; then
                    recommendations+=("${YELLOW}⚠️  GPU 顯存較少 (8-16GB)，只能運行中型模型${NC}")
                    suitable_models+=("llama3.2:13b")
                    suitable_models+=("mistral:7b")
                    suitable_models+=("qwen2.5:14b")
                else
                    recommendations+=("${YELLOW}⚠️  GPU 顯存不足 (<8GB)，只能運行小型模型${NC}")
                    suitable_models+=("llama3.2:3b")
                    suitable_models+=("mistral:7b (量化版)")
                    suitable_models+=("phi3:mini")
                fi
            done
        fi
    else
        recommendations+=("${RED}❌ 未檢測到 GPU，將使用 CPU 推理（速度較慢）${NC}")
        warnings+=("${YELLOW}⚠️  強烈建議使用 GPU 以獲得可接受的推理速度${NC}")
        warnings+=("${YELLOW}⚠️  CPU 模式下，建議使用量化模型或小型模型${NC}")
    fi
    
    # 磁盤空間評估
    if [ -n "$FREE_SPACE_GB" ]; then
        if [ $FREE_SPACE_GB -ge 100 ]; then
            recommendations+=("${GREEN}✅ 磁盤空間充足 (${FREE_SPACE_GB}GB 可用)${NC}")
            score=$((score + 1))
        elif [ $FREE_SPACE_GB -ge 50 ]; then
            recommendations+=("${YELLOW}⚠️  磁盤空間中等 (${FREE_SPACE_GB}GB 可用)${NC}")
            warnings+=("${YELLOW}⚠️  大型模型需要 20-40GB 磁盤空間${NC}")
        else
            recommendations+=("${RED}❌ 磁盤空間不足 (${FREE_SPACE_GB}GB 可用)${NC}")
            warnings+=("${RED}❌ 建議至少 50GB 可用空間用於模型存儲${NC}")
        fi
    fi
    
    # 輸出建議
    echo ""
    echo "評估結果:"
    for rec in "${recommendations[@]}"; do
        echo -e "  $rec"
    done
    
    if [ ${#warnings[@]} -gt 0 ]; then
        echo ""
        echo -e "${YELLOW}⚠️  警告:${NC}"
        for warning in "${warnings[@]}"; do
            echo -e "  $warning"
        done
    fi
    
    # 去重並顯示推薦模型
    if [ ${#suitable_models[@]} -gt 0 ]; then
        echo ""
        echo -e "${BLUE}💡 推薦的模型 (根據您的硬體):${NC}"
        printf '%s\n' "${suitable_models[@]}" | sort -u | head -10 | while read -r model; do
            echo "  • $model"
        done
    fi
    
    # 總體評估
    echo ""
    echo "============================================================"
    echo "總體評估"
    echo "============================================================"
    
    if [ $score -ge 5 ]; then
        echo -e "${GREEN}✅ 非常適合使用 Ollama 部署本地模型${NC}"
        echo "   建議: 可以運行大型模型，性能良好"
    elif [ $score -ge 3 ]; then
        echo -e "${YELLOW}⚠️  可以使用 Ollama 部署本地模型${NC}"
        echo "   建議: 使用中型或量化模型，性能可接受"
    elif [ $score -ge 1 ]; then
        echo -e "${YELLOW}⚠️  勉強可以使用 Ollama，但性能較差${NC}"
        echo "   建議: 僅使用小型模型，或考慮使用 API 服務"
    else
        echo -e "${RED}❌ 不建議使用 Ollama 部署本地模型${NC}"
        echo "   建議: 使用雲端 API 服務 (如 OpenAI, DeepSeek API)"
    fi
}

# 主函數
main() {
    echo "============================================================"
    echo "硬體資源檢查 - Ollama 部署評估"
    echo "============================================================"
    echo ""
    echo "系統信息: $(uname -s) $(uname -r)"
    echo "Python 版本: $(python3 --version 2>/dev/null || echo '未安裝')"
    
    check_cpu
    check_memory
    check_disk
    check_gpu
    check_ollama_requirements
    
    echo ""
    echo "============================================================"
    echo "檢查完成"
    echo "============================================================"
    echo ""
    echo -e "${BLUE}💡 提示:${NC}"
    echo "  • 安裝 Ollama: https://ollama.com/download"
    echo "  • 下載模型: ollama pull <model-name>"
    echo "  • 量化模型通常以 ':q4_0' 或 ':q8_0' 結尾，體積更小"
    echo "  • GPU 模式需要安裝對應的驅動 (NVIDIA CUDA 或 AMD ROCm)"
}

# 執行主函數
main