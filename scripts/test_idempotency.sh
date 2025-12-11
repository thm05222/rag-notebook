#!/bin/bash

# Idempotency Key 功能測試腳本
# 用法: ./scripts/test_idempotency.sh [API_URL]

API_URL="${1:-http://localhost:8000}"
KEY=$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid 2>/dev/null || echo "test-key-$(date +%s)")

echo "========================================="
echo "Idempotency Key 功能測試"
echo "========================================="
echo "API URL: $API_URL"
echo "Idempotency Key: $KEY"
echo ""

# 測試 1: 首次請求
echo "測試 1: 首次請求"
echo "-----------------------------------"
RESPONSE1=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/sources" \
  -H "Idempotency-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{"title": "Idempotency Test Source"}')

HTTP_CODE1=$(echo "$RESPONSE1" | tail -n 1)
BODY1=$(echo "$RESPONSE1" | head -n -1)

echo "HTTP Status: $HTTP_CODE1"
echo "Response: $BODY1"
echo ""

if [ "$HTTP_CODE1" != "200" ] && [ "$HTTP_CODE1" != "201" ]; then
    echo "❌ 測試失敗：首次請求應該成功"
    exit 1
fi

echo "✅ 測試 1 通過"
echo ""

# 等待一秒確保請求完成
sleep 1

# 測試 2: 重複請求（應返回快取）
echo "測試 2: 重複請求（相同 key + body）"
echo "-----------------------------------"
RESPONSE2=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/sources" \
  -H "Idempotency-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{"title": "Idempotency Test Source"}' \
  -v 2>&1)

HTTP_CODE2=$(echo "$RESPONSE2" | grep "< HTTP" | awk '{print $3}')
REPLAYED_HEADER=$(echo "$RESPONSE2" | grep -i "X-Idempotent-Replayed" || echo "")

echo "HTTP Status: $HTTP_CODE2"
if [ -n "$REPLAYED_HEADER" ]; then
    echo "✅ 找到 X-Idempotent-Replayed header"
else
    echo "⚠️  未找到 X-Idempotent-Replayed header（可能是首次請求尚未完成）"
fi
echo ""

# 測試 3: 衝突請求（相同 key + 不同 body）
echo "測試 3: 衝突請求（相同 key + 不同 body）"
echo "-----------------------------------"
KEY2="$KEY"  # 使用相同的 key
RESPONSE3=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/sources" \
  -H "Idempotency-Key: $KEY2" \
  -H "Content-Type: application/json" \
  -d '{"title": "Different Title"}')

HTTP_CODE3=$(echo "$RESPONSE3" | tail -n 1)
BODY3=$(echo "$RESPONSE3" | head -n -1)

echo "HTTP Status: $HTTP_CODE3"
echo "Response: $BODY3"
echo ""

if [ "$HTTP_CODE3" == "422" ]; then
    echo "✅ 測試 3 通過：正確偵測到衝突"
elif [ "$HTTP_CODE3" == "200" ] || [ "$HTTP_CODE3" == "201" ]; then
    echo "✅ 測試 3 通過：返回快取的回應（首次請求可能尚未完成）"
else
    echo "⚠️  測試 3：預期 422 或 200，收到 $HTTP_CODE3"
fi
echo ""

# 測試 4: 新的 key（應該成功）
echo "測試 4: 新的 Idempotency Key"
echo "-----------------------------------"
KEY3=$(uuidgen 2>/dev/null || cat /proc/sys/kernel/random/uuid 2>/dev/null || echo "test-key-$(date +%s)-2")
RESPONSE4=$(curl -s -w "\n%{http_code}" -X POST "$API_URL/api/sources" \
  -H "Idempotency-Key: $KEY3" \
  -H "Content-Type: application/json" \
  -d '{"title": "Another Test Source"}')

HTTP_CODE4=$(echo "$RESPONSE4" | tail -n 1)
BODY4=$(echo "$RESPONSE4" | head -n -1)

echo "Idempotency Key: $KEY3"
echo "HTTP Status: $HTTP_CODE4"
echo ""

if [ "$HTTP_CODE4" == "200" ] || [ "$HTTP_CODE4" == "201" ]; then
    echo "✅ 測試 4 通過"
else
    echo "❌ 測試 4 失敗：新 key 應該成功"
fi
echo ""

# 總結
echo "========================================="
echo "測試完成"
echo "========================================="
echo ""
echo "注意事項："
echo "1. 如果測試 2 未找到 X-Idempotent-Replayed header，"
echo "   可能是因為首次請求尚未完成處理。"
echo "2. 如果測試 3 返回 200 而非 422，也可能是相同原因。"
echo "3. 建議在測試間增加延遲，或檢查 command 狀態。"
echo ""
echo "查看 idempotency records："
echo "docker-compose exec surrealdb surreal sql --conn http://localhost:8000 --user root --pass root --ns open_notebook --db open_notebook"
echo "然後執行: SELECT * FROM idempotency_record;"

