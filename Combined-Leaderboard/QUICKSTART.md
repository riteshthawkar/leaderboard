# Quick Start Guide - Combined Leaderboard

## 1. Initial Setup

### Step 1: Configure Environment
```bash
cd Combined-Leaderboard
cp .env.example .env
```

Edit `.env` and update these paths:
```env
DO_YOU_SEE_ME_ROOT=../Do-You-See-Me
MINDS_EYE_ROOT=../Mind-s-Eye
```

### Step 2: Launch the Application

**Windows:**
```bash
run.bat
```

**Linux/Mac:**
```bash
bash run.sh
```

This will:
- Create virtual environment
- Install dependencies
- Start Flask server on http://localhost:5000

## 2. Access the Web Interface

Open your browser to **http://localhost:5000**

### Dashboard Features
- **Statistics Cards**: Total submissions, unique models, average accuracy, best accuracy
- **Submit Section**: Upload predictions, select benchmark and task
- **Leaderboard**: View rankings with filtering options
- **About Section**: Information about benchmarks and submission formats

## 3. Submit Predictions

### Using the Web Form

1. Enter **Model Name** (e.g., "GPT-4V", "Qwen-VL")
2. Select **Benchmark**: 
   - `minds_eye`: For reasoning tasks
   - `do_you_see_me`: For perception tasks
3. Choose **Task** (optional): Filter to specific task
4. Select **Prediction File**: CSV or JSON format
5. Click **Submit Predictions**

### CSV Format Example
```csv
image_name,task_name,prediction
image_0.png,mental_rotation,A
image_1.png,mental_rotation,C
image_2.png,abstract,B
```

### JSON Format Example
```json
{
  "mental_rotation": {
    "image_0.png": "A",
    "image_1.png": "C"
  },
  "abstract": {
    "image_2.png": "B"
  }
}
```

## 4. View Results

### On the Web Interface
- Results appear instantly after submission
- Overall accuracy percentage displayed
- Per-task breakdown in leaderboard
- Click on any entry for detailed results

### Using the API

**Get Leaderboard:**
```bash
curl http://localhost:5000/api/leaderboard?benchmark=minds_eye&limit=10
```

**Submit Programmatically:**
```bash
curl -X POST http://localhost:5000/api/submit \
  -F "file=@predictions.csv" \
  -F "model_name=MyModel" \
  -F "benchmark=minds_eye"
```

**Get Submission Details:**
```bash
curl http://localhost:5000/api/submission/{submission_id}
```

## 5. Available Tasks

### Mind's-Eye Tasks
- slippage
- abstract
- mental_rotation
- mental_composition
- paper_folding
- dynamic_isomorph
- symmetric_isomorph
- hierarchial_isomorph

### Do-You-See-Me Tasks
- visual_spatial (2D & 3D)
- letter_disambiguation (2D & 3D)
- visual_form_constancy (2D & 3D)
- visual_figure_ground (2D only)
- visual_closure (2D only)
- shape_discrimination (2D & 3D)
- shape_color_discrimination (2D & 3D)

## 6. Troubleshooting

### Ground truth files not found
Check `.env` paths:
```bash
# Verify directories exist
ls ../Do-You-See-Me/2D_DoYouSeeMe/
ls ../Mind-s-Eye/data/
```

### Port already in use
Change port in Flask configuration:
```bash
FLASK_ENV=development FLASK_DEBUG=true python -m flask run --port 5001
```

### Import errors
Ensure virtual environment is activated:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

Then reinstall dependencies:
```bash
pip install -r requirements.txt
```

## 7. Run Tests

Validate system installation:
```bash
python test_system.py
```

Expected output:
```
✓ Option extraction tests passed
✓ Answer comparison tests passed
✓ Ground truth loading test passed
✓ CSV parsing test passed
✓ JSON parsing test passed
✓ Scoring engine initialized successfully
✓ Leaderboard manager test passed
✓ All tests passed!
```

## 8. Example Workflow

```bash
# 1. Prepare predictions in CSV
echo "image_name,task_name,prediction
img_0.png,mental_rotation,A
img_1.png,mental_rotation,B" > predictions.csv

# 2. Submit via API
curl -X POST http://localhost:5000/api/submit \
  -F "file=@predictions.csv" \
  -F "model_name=TestModel" \
  -F "benchmark=minds_eye"

# Response:
# {
#   "success": true,
#   "submission_id": "abc123...",
#   "overall_accuracy": 0.75,
#   "task_results": {...}
# }

# 3. View leaderboard
curl http://localhost:5000/api/leaderboard

# 4. Get submission details
curl http://localhost:5000/api/submission/abc123...
```

## 9. Advanced Configuration

### Enable Ollama for Better Option Extraction
```env
MINDS_EYE_OPTION_EXTRACTOR_MODEL=gemma3:4b
MINDS_EYE_OLLAMA_BASE_URL=http://localhost:11434
```

Then ensure Ollama is running:
```bash
ollama serve
```

### Production Deployment
For production use:
1. Set `FLASK_DEBUG=false` in `.env`
2. Use a production WSGI server (Gunicorn, uWSGI)
3. Consider adding authentication
4. Migrate to database backend (SQLite/PostgreSQL)

## 10. Need Help?

See [README.md](README.md) for:
- Full API documentation
- Detailed architecture
- Benchmark descriptions
- Performance notes
- Future enhancements

---

**Happy Benchmarking! 🎯**
