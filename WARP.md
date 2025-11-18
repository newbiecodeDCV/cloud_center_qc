# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a call center QA (Quality Assurance) system that evaluates customer service call recordings. It analyzes both **communication skills** and **sales skills** using audio processing, speech recognition, and LLM-based evaluation.

## Core Commands

### Development Setup
```bash
# Install dependencies
pip install -r requirements.txt

# For sales-specific features
pip install -r requirements_sales.txt
```

### Running the Applications

**API Server** (FastAPI backend):
```bash
python api.py --host 0.0.0.0 --port 8000 \
  --gpt_model gpt-4.1-mini \
  --csv_path src/qa_sales/modules/databases/salescript.tsv \
  --eval_prompt_template src/qa_sales/modules/prompt_templates/evaluate_script.txt \
  --preprocess_prompt_template src/qa_sales/modules/prompt_templates/preprocess.txt \
  --classify_prompt_template src/qa_sales/modules/prompt_templates/classify_utterances.txt \
  --db_path src/qa_sales/modules/databases/salescript_db
```

**Gradio Demo** (Web UI):
```bash
python app.py --server_name 0.0.0.0
```

**Single Call Evaluation** (Testing):
```bash
python scripts/evaluate_single_call.py <path_to_audio_file.wav>
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/audio_processing/test_analysis.py

# Run with verbose output
pytest -v
```

### Code Quality
```bash
# Format and lint code
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Docker
```bash
# Build Docker image
cd docker && ./1-build_image.sh

# Run container
./2-run_container.sh

# Add sudo permissions (if needed)
./3-add_sudo_for_container.sh
```

## Architecture

### High-Level Flow

The system follows a two-stage evaluation pipeline:

1. **Communication Skills Evaluation** (`src/qa_communicate/`)
   - Audio → Features Extraction (acoustic + dialogue API)
   - Features → LLM Evaluation (scoring communication quality)

2. **Sales Skills Evaluation** (`src/qa_sales/`)
   - Audio → Dialogue API → Speaker Identification
   - Sales Utterances → Criteria Classification (using vector DB)
   - Classified Utterances → Per-Criterion LLM Evaluation

### Component Structure

```
src/
├── main_evaluator.py          # Orchestrates full evaluation pipeline
├── qa_communicate/            # Communication skills evaluation
│   ├── audio_processing/      # Feature extraction (acoustics + dialogue)
│   │   ├── analysis.py        # Acoustic analyzer (SPM, pitch, volume, silence)
│   │   ├── dialogue.py        # Dialogue API integration
│   │   └── qa.py              # Client for communication QA
│   ├── evaluation/
│   │   └── evaluator.py       # LLM-based communication evaluation
│   ├── prompt/
│   │   └── prompts.py         # Prompt templates for communication scoring
│   └── core/
│       ├── utils.py           # Task ID generation, file utils
│       └── langfuse_config.py # Observability with Langfuse
├── qa_sales/                  # Sales skills evaluation
│   └── modules/
│       ├── dialogue_processor.py   # Speaker role identification
│       ├── database.py            # ChromaDB for criteria matching
│       ├── evaluators.py          # Per-criterion LLM evaluation
│       ├── qa_evaluators.py       # Sales pipeline orchestrator
│       └── databases/
│           └── salescript.tsv     # Sales criteria database (watched for changes)
└── utils/
    ├── llm_service.py         # Unified LLM client (OpenAI)
    └── file_handlers.py       # CSV file watcher for hot-reloading criteria
```

### Key Design Patterns

**Two-Evaluator Pattern**: 
- `QAMainEvaluator` orchestrates both communication and sales evaluations
- Each returns structured scores that are combined into a final report

**Async Pipeline**: 
- All API calls (dialogue, LLM) are async
- Uses `asyncio` throughout for concurrent processing

**ChromaDB for Criteria Matching**:
- Sales criteria stored in `salescript.tsv` are embedded into ChromaDB
- Utterances are classified to criteria using semantic search
- `CSVWatcher` auto-rebuilds DB when TSV changes

**Langfuse Observability**:
- Traces entire evaluation as a tree: Root → Communication/Sales → Sub-steps
- Each LLM call is logged with inputs/outputs
- Conditional: only enabled if `LANGFUSE_PUBLIC_KEY` is set in `.env`

### Audio Processing Details

**AudioSegment**: 
- Represents a dialogue turn with speaker, text, timestamps
- Includes corruption detection (short duration + many words = likely API error)

**AcousticAnalyzer**:
- Calculates: SPM (syllables per minute), volume (dB), pitch (Hz), silence ratio
- Filters filler words (à, ừm, vậy, etc.) before calculating SPM
- Uses `librosa.pyin` for pitch extraction

**MetadataCalculator**:
- Computes call-level stats: total duration, speaker turns, sales/customer time ratio

### Sales Evaluation Pipeline

1. **Get Dialogue**: Call external dialogue API to get transcript + speaker IDs
2. **Preprocess**: Use LLM to identify which speaker is Sales vs. Customer
3. **Classify**: For each Sales utterance, find relevant criteria using ChromaDB similarity search
4. **Evaluate**: For each criterion, LLM judges if sales script was followed (binary: đạt/không đạt)
5. **Aggregate**: Sum scores across criteria

### API Design

**POST /** (FastAPI endpoint):
- Accepts: `file` (audio bytes or URL) + optional `task_id`
- Authentication: JWT token or private token
- Returns: `task_id` immediately (202 Accepted)
- Processing happens in background task
- Results saved to `tasks_YYYY-MM/{task_id}_done.json`

**Polling Pattern**:
- Client polls `/` with same `task_id`
- Server checks for `_done.json` or `_running.json`
- Returns 200 (done), 202 (running), or starts new task

## Environment Variables

Required in `.env`:
```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Optional: Langfuse observability
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Evaluation Criteria

### Communication Skills (4 criteria)
1. **Chào xưng danh** (Greeting/Identification): 0 or 1 point
2. **Kỹ năng nói** (Speaking skills): 0 or 1 point
3. **Kỹ năng nghe** (Listening skills): 0 or 1 point
4. **Thái độ** (Attitude): 0 or 1 point

Final communication score: `0.2 * (criteria_1 + criteria_2) + 0.8 * (criteria_3 + criteria_4)`

### Sales Skills (variable criteria)
- Defined in `src/qa_sales/modules/databases/salescript.tsv`
- Each criterion: criteria_id, criteria_name, description, examples
- Scores summed across all criteria (each worth defined points)

### Error Levels (Mức lỗi)
- **M1**: Minor issues (disfluency, monotone voice, speaking too fast/quiet)
- **M2**: Moderate issues (inappropriate tone, not following call protocols)
- **M3**: Major issues (unprofessional behavior, missing customer needs)

## Testing Notes

- Tests use `pytest` with mocked audio data
- `AcousticAnalyzer` tests focus on SPM calculation and filler word filtering
- `AudioSegment` tests validate corruption detection logic
- No integration tests with external APIs (dialogue API is mocked)

## File Organization

- **Task Results**: Saved to `tasks_YYYY-MM/` (auto-created by month)
- **Logs**: `.log` files (ignored by git)
- **Audio**: `.wav` files (ignored by git)
- **ChromaDB**: `src/qa_sales/modules/databases/salescript_db/` (ignored by git, rebuilt from TSV)

## Important Notes

- The dialogue API endpoint is external (not in this repo) - calls are made to a speech recognition service
- Task IDs are generated as MD5 hashes of audio bytes (for idempotency)
- JWT tokens use secret `"datamining_vcc"` and HS256 algorithm
- Private token is hardcoded in `api.py` for development (should be moved to env vars in production)
- Gradio app assumes proxy path: `https://speech.aiservice.vn/asr/cloud_qa_demo`
