# Combined Leaderboard - Vision Benchmark Evaluation System

A comprehensive web-based leaderboard system for evaluating Vision Language Models (VLMs) on two major benchmarks: **Mind's-Eye** (reasoning tasks) and **Do-You-See-Me** (perceptual tasks).

## Features

✅ **Dual Benchmark Support**
- **Mind's-Eye**: 8 cognitive reasoning tasks (mental rotation, abstract reasoning, paper folding, etc.)
- **Do-You-See-Me**: 7 visual perception dimensions (shape/color discrimination, form constancy, etc.)

✅ **Instant Evaluation**
- Upload predictions in CSV or JSON format
- Automatic scoring against ground truth
- Per-task and overall accuracy calculation
- Detailed results with answer comparisons

✅ **Smart Answer Extraction**
- Handles multiple answer formats (options A-F, numeric, text)
- Option extraction from verbose responses
- Case-insensitive text comparison
- Optional Ollama integration for advanced normalization

✅ **Leaderboard Rankings**
- Global rankings by overall accuracy
- Per-task leaderboards
- Model submission history
- Detailed statistics

✅ **Web Interface**
- Clean, responsive design
- Real-time leaderboard updates
- Submission form with validation
- Task filtering and sorting
- Submission details view

## Project Structure

```
Combined-Leaderboard/
├── backend/
│   ├── config.py                 # Configuration and paths
│   ├── leaderboard_manager.py    # Leaderboard storage/retrieval
│   ├── models/
│   │   └── submission.py         # Data models
│   ├── data_handlers/
│   │   ├── ground_truth.py       # Ground truth loading
│   │   └── submission.py         # Prediction parsing
│   ├── utils/
│   │   └── answer_extractor.py   # Answer normalization
│   ├── scoring/
│   │   └── engine.py             # Main scoring engine
│   └── web/
│       └── app.py                # Flask web application
├── frontend/
│   ├── templates/
│   │   └── index.html            # Main webpage
│   └── static/
│       ├── css/
│       │   └── style.css         # Styling
│       └── js/
│           └── main.js           # Frontend logic
├── uploads/                      # User submissions
├── results/                      # Evaluation results
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Installation

### Prerequisites
- Python 3.8+
- Git
- The Mind's-Eye and Do-You-See-Me repositories

### Setup

1. **Clone the repository** (or copy to your workspace)
```bash
cd Combined-Leaderboard
```

2. **Create a Python virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your paths and settings
```

5. **Run the application**
```bash
cd backend/web
python -m flask run
```

The leaderboard will be available at `http://localhost:5000`

## Configuration

### Environment Variables

Create a `.env` file with:

```env
# Repository paths
DO_YOU_SEE_ME_ROOT=/path/to/Do-You-See-Me
MINDS_EYE_ROOT=/path/to/Mind-s-Eye

# Flask configuration
FLASK_ENV=development
FLASK_DEBUG=False

# Ollama configuration (optional)
MINDS_EYE_OPTION_EXTRACTOR_MODEL=gemma3:4b
MINDS_EYE_OLLAMA_BASE_URL=http://localhost:11434

# Upload settings
MAX_UPLOAD_SIZE=50  # MB
```

## Usage

### Submission Format

#### CSV Format
```csv
image_name,task_name,prediction
image_0.png,mental_rotation,A
image_1.png,mental_rotation,C
image_2.png,abstract,B
```

#### JSON Format
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

Or alternative format:
```json
[
  {"image_name": "image_0.png", "task_name": "mental_rotation", "prediction": "A"},
  {"image_name": "image_1.png", "task_name": "mental_rotation", "prediction": "C"}
]
```

### API Endpoints

#### Submit Predictions
```bash
POST /api/submit
Content-Type: multipart/form-data

- file: prediction file (CSV or JSON)
- model_name: name of model
- benchmark: 'do_you_see_me' or 'minds_eye'
- task_name: (optional) specific task
```

#### Get Leaderboard
```bash
GET /api/leaderboard?benchmark=minds_eye&limit=25&task=mental_rotation
```

#### Get Submission Details
```bash
GET /api/submission/{submission_id}
```

#### Get Model Submissions
```bash
GET /api/model/{model_name}
```

#### List Available Tasks
```bash
GET /api/tasks?benchmark=minds_eye
```

#### Get Statistics
```bash
GET /api/statistics?benchmark=do_you_see_me
```

## Benchmarks

### Mind's-Eye Tasks

1. **Slippage** - Concept violation detection
2. **Abstract Reasoning** - Odd-one-out analogies
3. **Mental Rotation** - 3D object rotation selection
4. **Mental Composition** - 2D net to 3D matching
5. **Paper Folding** - Hole pattern prediction
6. **Dynamic Isomorphism** - Temporal transformation
7. **Symmetric Structures** - Symmetry violation detection
8. **Hierarchical Isomorphism** - Recursive structure violation

### Do-You-See-Me Dimensions

1. **Shape Discrimination** (2D & 3D)
2. **Shape-Color Discrimination** (2D & 3D)
3. **Visual Form Constancy** (2D & 3D)
4. **Letter Disambiguation** (2D & 3D)
5. **Visual Figure-Ground** (2D only)
6. **Visual Closure** (2D only)
7. **Visual Spatial** (2D & 3D)

## Answer Extraction Strategy

The system uses a multi-strategy approach for comparing answers:

1. **Option Extraction** (for A-F answers)
   - Regex pattern matching: `(A)`, `Option B`, etc.
   - Optional Ollama-based normalization

2. **Numeric Comparison** (for numeric answers)
   - Extract first number from text
   - Direct comparison

3. **Text Matching** (fallback)
   - Case-insensitive comparison
   - Whitespace normalization

## Scoring

- **Metric**: Exact match (1.0 = correct, 0.0 = incorrect)
- **Per-task accuracy**: Mean score across all samples
- **Overall accuracy**: Mean across all tasks
- **Standard deviation**: Calculated per task

## Development

### Adding a New Benchmark

1. Create ground truth loader in `data_handlers/ground_truth.py`
2. Add benchmark enum to `models/submission.py`
3. Update `config.py` with task definitions
4. Implement scoring method in `scoring/engine.py`

### Testing

```bash
cd backend
python -c "from scoring.engine import ScoringEngine; e = ScoringEngine(); print('Engine loaded')"
```

## Troubleshooting

### Ground truth files not found
- Verify `DO_YOU_SEE_ME_ROOT` and `MINDS_EYE_ROOT` environment variables
- Check that `dataset_info.csv` and `annotations.json` exist in the expected paths

### Predictions not matching
- Verify CSV/JSON format matches expected structure
- Check image names match ground truth exactly
- Review comparison reasoning in results

### Ollama errors
- Ensure Ollama is running: `ollama serve`
- Pull required model: `ollama pull gemma3:4b`
- Check `MINDS_EYE_OLLAMA_BASE_URL` is correct

## Performance Notes

- Ground truth files are cached on first load
- Leaderboard is stored in JSON (consider database for scale)
- Submission processing is sequential (suitable for < 1000 submissions/day)
- Max upload size: 50 MB (configurable)

## Future Enhancements

- [ ] Database backend (SQLite/PostgreSQL)
- [ ] Batch evaluation with progress tracking
- [ ] Model comparison tools
- [ ] Difficulty-based rankings
- [ ] Advanced filtering (by date, accuracy range, etc.)
- [ ] Export functionality (CSV, PDF reports)
- [ ] User authentication and model management
- [ ] Automated benchmark refresh

## Citation

If you use this leaderboard system, please cite:

```bibtex
@misc{combined_leaderboard_2024,
  title={Combined Vision Benchmark Leaderboard},
  author={Your Name},
  year={2024}
}
```

Also cite the original benchmarks:

- **Do-You-See-Me**: [Original Paper](link-to-paper)
- **Mind's-Eye**: [Original Paper](link-to-paper)

## License

See LICENSE file in parent directory

## Support

For issues, questions, or contributions, please open an issue or contact the development team.
