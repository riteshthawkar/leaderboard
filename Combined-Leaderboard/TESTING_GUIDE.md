# Quick Testing Guide - Production Fixes

## Pre-Testing Checklist

```bash
# Navigate to project directory
cd Combined-Leaderboard

# Install/update dependencies
pip install -r requirements.txt

# Verify dependencies installed
pip list | grep -E "pydantic|flask-limiter|flask-httpauth|python-json-logger|sqlalchemy|python-magic"
```

## 1. Test Accuracy Calculation Fix

**Purpose:** Verify the weighted accuracy is calculated correctly

```bash
# Create test submission file
cat > test_predictions.json << 'EOF'
{
  "mental_rotation": {
    "image_1": "A",
    "image_2": "B"
  },
  "abstract": {
    "image_3": "C",
    "image_4": "C",
    "image_5": "C"
  }
}
EOF

# Expected: 
# - Task 1: 1/2 correct (50%)
# - Task 2: 3/3 correct (100%)
# - Overall: (1+3)/(2+3) = 80% (NOT 75% which would be the simple average)

python -c "
from backend.scoring.engine import ScoringEngine
from backend.models.submission import BenchmarkType
from pathlib import Path

engine = ScoringEngine(use_ollama=False)
try:
    result = engine.score_submission(
        Path('test_predictions.json'),
        BenchmarkType.MINDS_EYE,
        'test_model'
    )
    print(f'Overall Accuracy: {result.overall_accuracy:.2%}')
    print(f'Total Samples: {result.total_samples}')
    print(f'Correct Samples: {result.correct_samples}')
    print(f'Task Results: {list(result.task_results.keys())}')
except Exception as e:
    print(f'Error: {e}')
"
```

## 2. Test Logging

**Purpose:** Verify logging is working and logs are being created

```bash
# Start the application
python -c "
import logging
from backend.logging_config import logger

logger.info('Test info message')
logger.warning('Test warning message')
logger.error('Test error message')
print('Check logs/ directory for log files')
"

# Verify log files created
ls -lah logs/
cat logs/app.log | tail -10
```

## 3. Test File Upload Security

**Purpose:** Verify file validation is working

```python
# Test file validation
python << 'EOF'
from backend.file_security import FileSecurityValidator
from pathlib import Path
import tempfile

# Test 1: Valid CSV file
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    f.write("image_name,task_name,prediction\n")
    f.write("img_1,task,A\n")
    csv_file = f.name

# Simulate file object
class FakeFile:
    def __init__(self, path):
        self.path = path
    def seek(self, pos, whence=0):
        pass
    def read(self, size=-1):
        with open(self.path, 'rb') as f:
            return f.read(size)
    def tell(self):
        return 0

file_obj = FakeFile(csv_file)
is_valid, error, safe_name = FileSecurityValidator.validate_and_secure_upload(file_obj, "test.csv")
print(f"Valid: {is_valid}, Error: {error}, Safe Name: {safe_name}")

# Test 2: Invalid file type
with tempfile.NamedTemporaryFile(mode='w', suffix='.exe', delete=False) as f:
    f.write("MZ")  # EXE header
    exe_file = f.name

file_obj = FakeFile(exe_file)
is_valid, error, safe_name = FileSecurityValidator.validate_and_secure_upload(file_obj, "virus.exe")
print(f"Valid (should be False): {is_valid}, Error: {error}")

# Cleanup
Path(csv_file).unlink()
Path(exe_file).unlink()
EOF
```

## 4. Test Input Validation (Pydantic)

**Purpose:** Verify input validation prevents invalid requests

```python
python << 'EOF'
from backend.validators import SubmissionRequest, BenchmarkEnum

# Test 1: Valid request
try:
    req = SubmissionRequest(
        model_name="GPT-4V",
        benchmark="minds_eye",
        task_name="mental_rotation"
    )
    print(f"✓ Valid request accepted: {req.model_name}")
except Exception as e:
    print(f"✗ Valid request rejected: {e}")

# Test 2: Invalid benchmark
try:
    req = SubmissionRequest(
        model_name="Test",
        benchmark="invalid_benchmark"
    )
    print(f"✗ Invalid benchmark accepted (BAD)")
except Exception as e:
    print(f"✓ Invalid benchmark rejected: {e}")

# Test 3: Invalid model name (too long)
try:
    req = SubmissionRequest(
        model_name="A" * 300,
        benchmark="minds_eye"
    )
    print(f"✗ Too-long name accepted (BAD)")
except Exception as e:
    print(f"✓ Too-long name rejected: {e}")

# Test 4: Invalid model name (special chars)
try:
    req = SubmissionRequest(
        model_name="Model<script>alert('xss')</script>",
        benchmark="minds_eye"
    )
    print(f"✗ XSS payload accepted (BAD)")
except Exception as e:
    print(f"✓ XSS payload rejected: {e}")
EOF
```

## 5. Test Rate Limiting

**Purpose:** Verify rate limiting is configured

```bash
# Start server in background
python -c "
from backend.web.app import app
print('Starting test server on http://localhost:5000')
print('Press Ctrl+C to stop')
app.run(debug=True, use_reloader=False)
" &
SERVER_PID=$!

# Wait for server to start
sleep 2

# Test rate limiting with rapid requests
for i in {1..15}; do
    curl -s http://localhost:5000/api/leaderboard > /dev/null
    echo "Request $i"
done

# After 10+ requests, should get 429 Too Many Requests
curl -v http://localhost:5000/api/leaderboard 2>&1 | grep "429\|200"

# Kill server
kill $SERVER_PID
```

## 6. Test Health Check Endpoint

**Purpose:** Verify health check is working

```bash
# Start server
python -c "from backend.web.app import app; app.run(debug=False, port=5000)" &
SERVER_PID=$!

sleep 2

# Test health endpoint
curl -s http://localhost:5000/api/health | python -m json.tool

# Should return:
# {
#     "status": "healthy",
#     "timestamp": "2024-06-16T...",
#     "components": {
#         "database": "healthy",
#         "ground_truth": "healthy"
#     }
# }

kill $SERVER_PID
```

## 7. Test Frontend XSS Prevention

**Purpose:** Verify frontend properly escapes HTML

```bash
# Create test HTML file
cat > test_xss.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <script src="frontend/static/js/main.js"></script>
</head>
<body>
    <script>
        // Test escapeHtml function
        console.log("Test 1 - Normal text:", escapeHtml("Hello World"));
        console.log("Test 2 - HTML chars:", escapeHtml("<script>alert('xss')</script>"));
        console.log("Test 3 - Quotes:", escapeHtml('It\'s "quoted"'));
        console.log("Test 4 - Ampersand:", escapeHtml("A & B"));
        
        // All should be properly escaped
    </script>
</body>
</html>
EOF

# Open in browser and check console
echo "Open test_xss.html in browser and check DevTools console"
```

## 8. Test Error Handling

**Purpose:** Verify errors are logged and handled gracefully

```bash
# Create test with invalid ground truth
python << 'EOF'
from backend.data_handlers.ground_truth import DoYouSeeMeHandler
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Try to load non-existent task
try:
    handler = DoYouSeeMeHandler()
    result = handler.load_ground_truth("nonexistent_task")
except Exception as e:
    print(f"✓ Error handled gracefully: {type(e).__name__}")
    print(f"  Message: {e}")

# Check logs for error entries
print("\nCheck logs/app.log for error messages")
EOF
```

## 9. Test Database Operations

**Purpose:** Verify database functionality

```bash
python << 'EOF'
from backend.database import Database
from datetime import datetime
import tempfile

# Create temp database
with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
    db_path = f.name

db = Database(f"sqlite:///{db_path}")

# Test 1: Create tables
print("Creating tables...")
db.create_tables()
print("✓ Tables created")

# Test 2: Add submission
print("\nAdding test submission...")
submission_data = {
    "submission_id": "test-123",
    "model_name": "GPT-4V",
    "benchmark": "minds_eye",
    "overall_accuracy": 0.85,
    "total_samples": 100,
    "correct_samples": 85,
    "submitted_at": datetime.now(),
    "task_results": {"mental_rotation": 0.90},
    "metadata": {}
}
db.add_submission(submission_data)
print("✓ Submission added")

# Test 3: Retrieve submission
print("\nRetrieving submission...")
retrieved = db.get_submission("test-123")
print(f"✓ Retrieved: {retrieved['model_name']} - {retrieved['overall_accuracy']:.2%}")

# Test 4: Get statistics
print("\nGetting statistics...")
stats = db.get_statistics()
print(f"✓ Stats: {stats['total_submissions']} submissions, {stats['unique_models']} unique models")

# Cleanup
import os
os.unlink(db_path)
EOF
```

## 10. End-to-End API Test

**Purpose:** Test complete API flow with authentication

```bash
# Start server
python -c "
import os
os.environ['API_TOKENS'] = 'test-token-12345'
from backend.web.app import app
app.run(debug=False, port=5000)
" &
SERVER_PID=$!

sleep 2

# Test without token (should fail)
echo "Test 1: No token (should fail)"
curl -s -X GET http://localhost:5000/api/leaderboard | python -m json.tool

# Test with invalid token (should fail)
echo -e "\nTest 2: Invalid token (should fail)"
curl -s -H "Authorization: Bearer invalid-token" \
     -X POST http://localhost:5000/api/submit \
     -F "file=@test_predictions.json" \
     -F "model_name=Test" \
     -F "benchmark=minds_eye" | python -m json.tool

# Test health endpoint (no auth required)
echo -e "\nTest 3: Health check (no auth)"
curl -s http://localhost:5000/api/health | python -m json.tool

kill $SERVER_PID
```

## Verification Checklist

After running all tests, verify:

- [ ] Accuracy calculation produces weighted average (not simple mean)
- [ ] Log files created in `logs/` directory
- [ ] File validation rejects invalid files
- [ ] Input validation prevents malformed requests
- [ ] Rate limiting returns 429 after threshold
- [ ] Health endpoint returns valid JSON
- [ ] XSS attempts are properly escaped
- [ ] Errors are logged without exposing details
- [ ] Database operations work correctly
- [ ] API requires authentication token

## Common Issues & Fixes

### Issue: "ModuleNotFoundError: No module named 'pydantic'"
**Fix:** `pip install pydantic==2.5.0`

### Issue: "ModuleNotFoundError: No module named 'flask_limiter'"
**Fix:** `pip install flask-limiter==3.5.0`

### Issue: "ValueError: could not convert string to float"
**Fix:** Ensure accuracy values are between 0.0 and 1.0

### Issue: "FileNotFoundError" for ground truth
**Fix:** Verify DO_YOU_SEE_ME_ROOT and MINDS_EYE_ROOT paths in .env

### Issue: Logging not working
**Fix:** Ensure `logs/` directory exists and is writable

## Next Steps

Once all tests pass:

1. [ ] Run full test suite: `python test_system.py`
2. [ ] Load test with real data
3. [ ] Security audit
4. [ ] Deploy to staging
5. [ ] Deploy to production

