# Abduction Demo: Bridging Explanations and Logics

A demonstration of multimodal AI pipeline combining Large Language Models (LLMs), Vision-Language Models (VLMs), and Probabilistic Logic Programming for explainable image classification.

## Overview

This repository contains the reference implementation for the AISoLA 2025 paper:

**"Bridging Explanations and Logics: Opportunities and Multimodal Language Models"**

**Authors:**

- Nicolas Sebastian Schuler (nicolas.schuler@student.kit.edu)
- Vincenzo Scotti (vincenzo.scotti@kit.edu)
- Matteo Camilli (matteo.camilli@polimi.it)
- Raffaela Mirandola (raffaela.mirandola@kit.edu)

The demo showcases a novel approach to explainable AI that bridges natural language reasoning with formal probabilistic logic to classify animals (cats vs. dogs) in images.

## Architecture

The system implements a 5-stage pipeline that transforms natural language reasoning into formal logical inference:

```
+-------------+     +-------------+     +-------------+     +-------------+     +-------------+
|   Stage 1   |---->|   Stage 2   |---->|   Stage 3   |---->|   Stage 4   |---->|   Stage 5   |
|  Reasoning  |     |   Coding    |     |  Feature    |     |  Grounding  |     |   Logic     |
|             |     |             |     | Extraction  |     |             |     | Execution   |
+-------------+     +-------------+     +-------------+     +-------------+     +-------------+
   LLM-based        LLM-to-ProbLog      LLM-based VLM       VLM-based Image     ProbLog
   Analysis         Translation          Parser              Feature Detection    Inference
```

### Stage 1: Reasoning

- **Input**: Scientific query about cat vs. dog classification
- **Process**: LLM generates comparative analysis of biological characteristics
- **Output**: Structured reasoning document (`reasoning.md`)
- **Model**: Qwen3-30B

### Stage 2: Coding

- **Input**: Reasoning description from Stage 1
- **Process**: LLM translates natural language into ProbLog program
- **Output**: Probabilistic logic program (`coding.pl`)
- **Model**: Gemini (configurable)

### Stage 3: Feature Extraction

- **Input**: ProbLog program from Stage 2
- **Process**: LLM extracts feature names (atoms) from the program
- **Output**: List of feature identifiers (`feature-list.txt`)
- **Model**: Qwen3-Coder-30B with DSPy

### Stage 4: Grounding

- **Input**: Feature list + input image
- **Process**: Vision-Language Model detects features and assigns confidence scores
- **Output**: Feature probability mappings (`grounding.json`)
- **Model**: Qwen3-VL-8B with DSPy

### Stage 5: Logic Execution

- **Input**: ProbLog program + grounded evidence
- **Process**: Probabilistic inference using ProbLog engine
- **Output**: Classification probabilities (P(cat), P(dog))
- **Engine**: ProbLog 2.2.7

## Installation

### Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Local LLM server (LM Studio, Ollama, or similar) running on `http://127.0.0.1:1234/v1`

### Setup

1. **Clone the repository:**

   ```bash
   git clone git@github.com:NicolasSchuler/abduction-demo.git
   cd abduction-demo
   ```

2. **Install dependencies using uv:**

   ```bash
   uv sync
   ```

   Alternatively, with pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up local LLM server:**
   - Install [LM Studio](https://lmstudio.ai/) or [Ollama](https://ollama.ai/)
   - Load the required models:
     - `qwen/qwen3-30b-a3b-2507` (reasoning)
     - `qwen/qwen3-coder-30b` (feature extraction)
     - `qwen/qwen3-vl-8b` (vision grounding)
   - Start the server on `http://127.0.0.1:1234/v1`

4. **Prepare data:**
   - Place labeled images in `data/images/`
   - Place corresponding labels in `data/labels/`
   - Ensure `data/notes.json` contains category mappings

## Usage

### Basic Usage

Run the demo in testing mode (uses cached results):

```bash
python main_dspy.py
```

### Command-Line Options

```bash
python main_dspy.py [OPTIONS]

Options:
  --mode {testing,full}     Operation mode (default: testing)
  --image-index INDEX       Image to process (default: 1)
  --output-dir DIR          Results directory (default: result-prompts/)
  --log-level LEVEL         Logging level (default: INFO)
  --llm-base-url URL        LLM server URL
  --show-config             Print configuration and exit
```

### Examples

```bash
# Run in testing mode with default settings
python main_dspy.py

# Run full pipeline on image 0 with debug logging
python main_dspy.py --mode full --image-index 0 --log-level DEBUG

# Show current configuration
python main_dspy.py --show-config

# Use different LLM server
python main_dspy.py --llm-base-url http://192.168.1.10:8080/v1
```

### Full Pipeline Mode

To run the complete pipeline with live LLM inference:

```bash
# Using command-line argument
python main_dspy.py --mode full

# Using environment variable
export TESTING=0
python main_dspy.py
```

### Output

The script outputs:

- **Console**: Classification probabilities

  ```
  2025-10-20 17:35:46 - INFO - Cat Probability: 0.8734
  2025-10-20 17:35:46 - INFO - Dog Probability: 0.1266
  ```

- **Files** (in `result-prompts/`):
  - `reasoning.md`: Natural language analysis
  - `coding.pl`: ProbLog program
  - `feature-list.txt`: Extracted features
  - `grounding.json`: Feature probabilities from image

## Project Structure

```
abduction-demo/
├── main_dspy.py              # Main pipeline orchestration
├── src/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration management
│   ├── data.py               # Image and label loading utilities
│   ├── img_utils.py          # Image processing and visualization
│   ├── validation.py         # Validation and sanitization utilities
│   └── test_pipeline.py      # Test suite
├── data/
│   ├── images/               # Input images (.jpg)
│   ├── labels/               # Segmentation labels (.txt)
│   ├── notes.json            # Category ID mappings
│   └── classes.txt           # Class names
├── result-prompts/           # Cached pipeline outputs
│   ├── reasoning.md
│   ├── coding.pl
│   ├── feature-list.txt
│   └── grounding.json
├── .archive/                 # Previous implementations
├── pyproject.toml            # Project configuration
├── ENHANCEMENTS.md           # Summary of enhancements
└── README.md                 # This file
```

## Key Dependencies

- **langchain** (0.3.27+): LLM orchestration framework
- **dspy-ai**: Structured prompting and LLM programming
- **problog** (2.2.7+): Probabilistic logic programming engine
- **opencv-python** (4.11+): Image processing
- **numpy** (2.3+): Numerical computations

See `pyproject.toml` for complete dependency list.

## Data Format

### Images

- Format: JPEG (.jpg)
- Location: `data/images/`
- Naming: Arbitrary (matched with label files by stem)

### Labels

- Format: YOLO-style segmentation format
- Location: `data/labels/`
- Format per line: `<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>`
- Coordinates: Normalized (0.0-1.0)

### Category Mapping (`data/notes.json`)

```json
{
  "categories": [
    { "id": 0, "name": "cat" },
    { "id": 1, "name": "dog" }
  ]
}
```

## Configuration

Configuration is centralized in `src/config.py` and can be overridden via:

- Environment variables (e.g., `export TESTING=0`)
- Command-line arguments (e.g., `--mode full`)

Key configuration parameters:

- `TESTING`: Set to `1` for cached results, `0` for live inference
- `IMAGE_INDEX`: Which image to process from the dataset
- `EPSILON_PROB`: Minimum probability value to avoid numerical issues (default: 0.0001)
- `LLM_BASE_URL`: LLM server endpoint
- Model identifiers for each pipeline stage

View configuration:

```bash
python -m src.config
# or
python main_dspy.py --show-config
```

### Environment Variables

```bash
export TESTING=0                              # Run full pipeline
export IMAGE_INDEX=0                          # Process first image
export LOG_LEVEL=DEBUG                        # Verbose logging
export LLM_BASE_URL=http://localhost:8080/v1  # Custom LLM server
```

## Development

### Code Style

- Formatter: Ruff (line length: 120)
- Type checker: Pyright
- Python version: 3.13

### Running Tests

```bash
# Run all tests
pytest src/test_pipeline.py -v

# Run with coverage
pytest src/test_pipeline.py --cov=src

# Run specific test class
pytest src/test_pipeline.py::TestValidation -v
```

### Running Type Checks

```bash
pyright
```

### Running Linter

```bash
ruff check .
```

### Running Formatter

```bash
ruff format .
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{schuler2025bridging,
  title={Bridging Explanations and Logics: Opportunities and Multimodal Language Models},
  author={Schuler, Nicolas Sebastian and Scotti, Vincenzo and Camilli, Matteo and Mirandola, Raffaela},
  booktitle={International Symposium on Leveraging Applications of Formal Methods (AISoLA)},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This research was conducted at:

- Karlsruhe Institute of Technology (KIT), Germany
- Politecnico di Milano, Italy

## Keywords

Explainable AI (XAI), Multimodal Language Models (MLM), Large Language Models (LLM), Logical Programming, Probabilistic Reasoning, ProbLog, Vision-Language Models

## Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: Data directory not found`

- **Solution**: Ensure `data/images/` and `data/labels/` directories exist and contain files

**Issue**: `Connection refused` when running full pipeline

- **Solution**: Start your local LLM server and verify it's running on the configured URL

**Issue**: `NotImplementedError: Gemini model is not implemented`

- **Solution**: Run in testing mode (`--mode testing`) or implement the Gemini API in `main_dspy.py`

**Issue**: Tests fail with import errors

- **Solution**: Run tests from project root: `pytest src/test_pipeline.py -v`

### Getting Help

- Check the [ENHANCEMENTS.md](ENHANCEMENTS.md) file for detailed enhancement documentation
- Review `src/test_pipeline.py` for usage examples
- Run with `--log-level DEBUG` for detailed output
- Verify configuration with `python -m src.config`

## Contributing

This is a research demonstration repository. For questions or collaboration inquiries, please contact the authors listed above.
