#!/bin/bash
# Signara - Quick Test Script

echo "=== Signara System Test ==="
echo ""

# Check if server is running
echo "1. Checking backend server..."
if curl -s http://localhost:8001/health >/dev/null 2>&1; then
	echo "   ✅ Backend running at http://localhost:8001"
else
	echo "   ❌ Backend not running. Starting..."
	cd signara/backend
	nohup ./venv/bin/python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8001 >/tmp/signara.log 2>&1 &
	sleep 3
fi

echo ""
echo "2. Testing endpoints..."

# Health check
echo -n "   Health: "
curl -s http://localhost:8001/health | grep -q "healthy" && echo "✅ OK" || echo "❌ FAIL"

# Text to gloss
echo -n "   Text→Gloss: "
result=$(curl -s -X POST http://localhost:8001/convert-text-to-gloss \
	-H "Content-Type: application/json" \
	-d '{"text": "hello thank you"}')
echo "$result" | grep -q "HELLO" && echo "✅ OK" || echo "❌ FAIL"

echo ""
echo "3. System Status"
echo "   - Backend: http://localhost:8001"
echo "   - API Docs: http://localhost:8001/docs"
echo "   - Frontend: Open signara/index.html in browser"

echo ""
echo "=== Test Complete ==="
