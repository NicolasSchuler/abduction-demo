# Abduction Demo: Bridging Explanations and Logics

A reference implementation demonstrating **intersymbolic explainability**—a novel approach that bridges subsymbolic neural perception with symbolic probabilistic reasoning to produce human-interpretable explanations for image classification decisions.

## Abstract

This repository accompanies the AISoLA 2025 paper:

> **["Bridging Explanations and Logics: Opportunities and Multimodal Language Models"](https://doi.org/10.5445/IR/1000184717)**
>
> Nicolas Sebastian Schuler¹, Vincenzo Scotti¹, Matteo Camilli², Raffaela Mirandola¹
>
> ¹Karlsruhe Institute of Technology (KIT), Germany
> ²Politecnico di Milano, Italy

Modern AI systems often lack transparency in their decision-making processes. This work presents an **Intersymbolic Explainability Pipeline** that combines Convolutional Neural Networks (CNNs), Explainable AI (XAI) techniques, Vision-Language Models (VLMs), and Probabilistic Logic Programming (ProbLog) to generate natural language explanations grounded in formal logical reasoning.

The system transforms the opaque prediction "it is a cat" into the interpretable explanation: *"It is a **cat** because it has **pointy ears**"*—where both the classification and its justification emerge from verifiable logical inference.

## Pipeline Overview

![Intersymbolic Explainability Pipeline](docs/static/images/xai_abduction_overview.png)

**Figure 1.** The Intersymbolic Explainability Pipeline architecture. The system operates in four phases: (1) **Phase 0** constructs an ontology and knowledge base encoding domain expertise about distinguishing features; (2) **Phase 1** performs subsymbolic perception through CNN prediction and XAI-based explanation generation; (3) **Phase 2** translates visual evidence into symbolic representations; (4) **Phase 3** applies abductive reasoning to derive explanations grounded in formal logic. The output provides both a classification decision and a human-interpretable justification.

## Methodology

The pipeline integrates multiple AI paradigms through a four-phase architecture, each phase bridging the gap between neural perception and symbolic reasoning.

### Phase 0: Ontology and Knowledge Base Construction

Before processing any image, the system constructs a domain-specific knowledge base that encodes expert knowledge about distinguishing features between classes.

| Stage | Process | Input | Output | Model |
|-------|---------|-------|--------|-------|
| **1. Reasoning** | LLM generates comparative analysis of biological characteristics | Scientific query | `reasoning.md` | Qwen3-30B |
| **2. Coding** | LLM translates natural language into ProbLog rules | Reasoning document | `coding.pl` | Gemini |
| **3. Feature Extraction** | LLM extracts observable feature atoms from logic program | ProbLog program | `feature-list.txt` | Qwen3-Coder-30B |

This phase is executed once per domain and produces reusable artifacts that encode the logical structure for classification.

### Phase 1: Subsymbolic Perception

Given an input image, the system performs neural network-based classification with explainability.

| Stage | Process | Input | Output |
|-------|---------|-------|--------|
| **Prediction** | CNN binary classification | Raw image | Class prediction + confidence |
| **Explanation Generation** | XAI attribution maps (Grad-CAM, SHAP, etc.) | Image + CNN activations | Saliency heatmaps |

The optional CNN+XAI preprocessing module (see [Preprocessing Pipeline](#preprocessing-pipeline)) generates enhanced images that visually highlight decision-relevant regions, improving downstream VLM feature detection.

### Phase 2: Intersymbolic Translation

The visual evidence from Phase 1 is translated into symbolic form suitable for logical reasoning.

| Stage | Process | Input | Output | Model |
|-------|---------|-------|--------|-------|
| **4. Grounding** | VLM detects features and assigns confidence scores | Feature list + image | `grounding.json` | Qwen3-VL-8B |

Feature names undergo bidirectional transformation for natural language processing: `pointed_ears` → `pointed ears` (for VLM) → `pointed_ears` (for ProbLog). This stage supports two modes:

- **Standard mode**: Detects all features present in the image
- **Highlighted-only mode** (`--highlighted-only`): Only considers red-highlighted regions, assigning 0.0 to non-highlighted features

### Phase 3: Abductive Reasoning

The grounded evidence is combined with the knowledge base for probabilistic logical inference.

| Stage | Process | Input | Output | Engine |
|-------|---------|-------|--------|--------|
| **5. Logic Execution** | Probabilistic inference | ProbLog program + evidence | P(cat), P(dog) | ProbLog 2.2.7 |

The ProbLog engine computes posterior probabilities for each class given the observed features, producing both a classification and a logical trace explaining which features contributed to the decision.

## Demo Videos

See the system in action:

- [**Kitty with highlighting** (VLM respects highlighting)](https://asciinema.org/a/kK4DzK42LhfPiLWwLsov5H1QD) — The `--highlighted-only` mode where the VLM correctly identifies only red-highlighted features
- [**Raw kitty** (no highlighting)](https://asciinema.org/a/L6XpAwddCe5zaRdzGORKrGuvN) — Standard processing without feature highlighting
- [**Kitty with highlighting** (VLM not respecting highlighting)](https://asciinema.org/a/ALCFuopk6n5Jyb51jYTqBcMZ7) — Example where VLM processes all features despite highlighting

## Preprocessing Pipeline

The optional CNN+XAI preprocessing module enhances input images by highlighting decision-relevant regions before VLM grounding. This improves feature detection accuracy by directing the VLM's attention to salient areas.

### Architecture

```
Input Image → CNN Classifier → XAI Explainer → Image Enhancer → Enhanced Image
                  ↓                  ↓               ↓
            Prediction +      Attribution      Visualization
            Confidence         Heatmap          Overlays
```

### Components

| Component | Description | Implementation |
|-----------|-------------|----------------|
| **CNN Classifier** | Binary classification (cat vs. dog) using transfer learning | EfficientNet-B0 with ImageNet weights |
| **XAI Explainer** | Generates attribution maps identifying influential image regions | Grad-CAM, Integrated Gradients, SHAP, Saliency, Layer-CAM, Occlusion |
| **Image Enhancer** | Creates visualization overlays highlighting salient regions | Heatmap overlay, spotlight, blur background, desaturation |

### XAI Methods

The system supports multiple explainability techniques, each offering different perspectives on model decision-making:

- **Grad-CAM**: Gradient-weighted Class Activation Mapping for localized explanations
- **Integrated Gradients**: Attribution method satisfying sensitivity and implementation invariance
- **SHAP**: Shapley Additive Explanations for feature importance
- **Saliency Maps**: Gradient-based pixel importance
- **Layer-CAM**: Layer-wise relevance propagation
- **Occlusion Sensitivity**: Systematic region masking analysis

### Enhancement Styles

Generated visualizations can be customized through configuration:

| Style | Description |
|-------|-------------|
| `heatmap_overlay` | Semi-transparent attribution heatmap over original image |
| `spotlight_heatmap` | Darkened background with illuminated salient regions |
| `composite_overlay` | Combined multi-method attribution visualization |
| `color_overlay` | Class-specific colormap (warm for cat, cool for dog) |
| `blur_background` | Sharp foreground with Gaussian-blurred background |
| `desaturate_background` | Color-preserved foreground with grayscale background |

### Configuration

Enable preprocessing via environment variables or `src/config.py`:

```bash
export ENABLE_CNN_PREPROCESSING=1
export XAI_METHODS="grad_cam,integrated_grad,shap"
export ENHANCEMENT_STYLES="heatmap_overlay,spotlight_heatmap"
```

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

   Alternatively, with pip (generate requirements.txt first if needed):

   ```bash
   pip install -e .
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

### Using Custom Images

You can process custom images using the `--image-path` flag. This is particularly useful in partial mode for testing grounding on new images:

```bash
# Process a custom image with the same feature set
python main_dspy.py --mode partial --image-path /path/to/your/image.jpg

# Process custom image with highlighted-only mode
python main_dspy.py --mode partial --image-path ./kitty_highlighted.webp --highlighted-only
```

**Note**: Custom images will have their label set to "unknown" since they're not part of the labeled dataset. The `--image-path` flag overrides `--image-index`.

### Command-Line Options

```bash
python main_dspy.py [OPTIONS]

Options:
  --mode {testing,full,partial}  Operation mode (default: testing)
  --image-index INDEX            Image to process from dataset (default: 1)
  --image-path PATH              Path to custom image file (overrides --image-index)
  --highlighted-only             Only consider red highlighted features in grounding
  --output-dir DIR               Results directory (default: result-prompts/)
  --log-level LEVEL              Logging level (default: INFO)
  --llm-base-url URL             LLM server URL
  --show-config                  Print configuration and exit
```

### Operation Modes

1. **Testing Mode** (`--mode testing`): Uses cached results from all pipeline stages
2. **Full Mode** (`--mode full`): Runs the complete 5-stage pipeline with live LLM inference
3. **Partial Mode** (`--mode partial`): Starts after the coding step, using cached reasoning, coding, and feature extraction results. Only runs grounding (on potentially new images) and logic execution. Ideal for testing different images with the same feature set.

### Examples

```bash
# Run in testing mode with default settings
python main_dspy.py

# Run full pipeline on image 0 with debug logging
python main_dspy.py --mode full --image-index 0 --log-level DEBUG

# Run partial pipeline with custom image
python main_dspy.py --mode partial --image-path ./my_cat.jpg

# Run with highlighted features only (red-highlighted regions)
python main_dspy.py --mode partial --image-path ./kitty_highlighted.webp --highlighted-only

# Process image from dataset with highlighted features
python main_dspy.py --mode partial --image-index 1 --highlighted-only

# Show current configuration
python main_dspy.py --show-config

# Use different LLM server
python main_dspy.py --llm-base-url http://192.168.1.10:8080/v1
```

### Pipeline Modes

**Full Pipeline Mode**: Run the complete pipeline with live LLM inference:

```bash
# Using command-line argument
python main_dspy.py --mode full

# Using environment variable
export TESTING=0
python main_dspy.py
```

**Partial Pipeline Mode**: Start after the coding step, using cached reasoning, coding, and feature extraction results:

```bash
# Run partial pipeline (requires cached reasoning.md, coding.pl, and feature-list.txt files)
python main_dspy.py --mode partial

# Test grounding with custom image
python main_dspy.py --mode partial --image-path ./custom_image.jpg

# Test with highlighted features only
python main_dspy.py --mode partial --highlighted-only
```

This mode is particularly useful when:

- You want to test grounding on different images with the same feature set
- You want to experiment with `--highlighted-only` mode
- The reasoning, coding, and feature extraction are stable but you need to update the vision processing
- You're developing or debugging the grounding and logic execution stages

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
├── main_dspy.py                  # Main pipeline orchestration
├── run_experiment.py             # Batch experiment runner
├── src/
│   ├── __init__.py               # Package initialization
│   ├── config.py                 # Centralized configuration management
│   ├── data.py                   # Image and label loading utilities
│   ├── img_utils.py              # Image encoding and visualization
│   ├── validation.py             # Validation and sanitization utilities
│   ├── test_pipeline.py          # Test suite
│   └── preprocessing/            # CNN + XAI preprocessing module
│       ├── __init__.py           # Subpackage initialization
│       ├── cnn_classifier.py     # EfficientNet-B0 binary classifier
│       ├── xai_explainer.py      # XAI methods (Grad-CAM, SHAP, etc.)
│       ├── image_enhancer.py     # Saliency-based image enhancement
│       └── preprocessing_pipeline.py  # Preprocessing orchestration
├── data/
│   ├── images/                   # Input images (.jpg)
│   ├── labels/                   # YOLO-format segmentation labels
│   ├── notes.json                # Category ID mappings
│   ├── classes.txt               # Class names
│   └── paper/                    # Paper figures and assets
│       └── xai_abduction_overview.pdf
├── result-prompts/               # Cached pipeline outputs
│   ├── reasoning.md              # Phase 0, Stage 1 output
│   ├── coding.pl                 # Phase 0, Stage 2 output
│   ├── feature-list.txt          # Phase 0, Stage 3 output
│   ├── grounding.json            # Phase 2, Stage 4 output
│   ├── cnn_metadata/             # CNN classification metadata
│   ├── xai_explanations/         # XAI visualization outputs
│   └── enhanced_images/          # Enhanced image visualizations
├── pyproject.toml                # Project configuration
├── ENHANCEMENTS.md               # Enhancement documentation
└── README.md                     # This file
```

## Key Dependencies

### Core Pipeline

| Package | Version | Purpose |
|---------|---------|---------|
| **langchain** | ≥0.3.27 | LLM orchestration framework |
| **dspy-ai** | latest | Structured prompting and LLM programming |
| **problog** | ≥2.2.7 | Probabilistic logic programming engine |
| **opencv-python** | ≥4.11 | Image processing and visualization |
| **numpy** | ≥2.3 | Numerical computations |

### CNN + XAI Preprocessing

| Package | Version | Purpose |
|---------|---------|---------|
| **torch** | ≥2.1 | Deep learning framework |
| **torchvision** | ≥0.16 | Pre-trained models (EfficientNet-B0) |
| **captum** | ≥0.6 | Model interpretability (Integrated Gradients) |
| **pytorch-grad-cam** | ≥1.4 | Grad-CAM and Layer-CAM implementations |
| **shap** | ≥0.42 | SHAP explanations |

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

Configuration is centralized in `src/config.py` and can be overridden via environment variables or command-line arguments.

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TESTING` | `1` | Operation mode: `1` = cached results, `0` = live inference |
| `IMAGE_INDEX` | `1` | Image index to process from dataset |
| `EPSILON_PROB` | `0.0001` | Minimum probability for numerical stability |
| `LLM_BASE_URL` | `http://127.0.0.1:1234/v1` | Local LLM server endpoint |

### CNN + XAI Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ENABLE_CNN_PREPROCESSING` | `1` | Enable/disable CNN+XAI preprocessing |
| `CNN_DEVICE` | `auto` | Compute device: `auto`, `cuda`, `cpu`, `mps` |
| `CNN_CONFIDENCE_THRESHOLD` | `0.7` | Minimum confidence for classification |
| `XAI_METHODS` | `grad_cam,integrated_grad` | Comma-separated XAI methods |
| `ENHANCEMENT_STYLES` | `heatmap_overlay,...` | Comma-separated enhancement styles |

### Viewing Configuration

```bash
python -m src.config
# or
python main_dspy.py --show-config
```

### Environment Variables Example

```bash
export TESTING=0                              # Run full pipeline
export IMAGE_INDEX=0                          # Process first image
export LOG_LEVEL=DEBUG                        # Verbose logging
export LLM_BASE_URL=http://localhost:8080/v1  # Custom LLM server
export ENABLE_CNN_PREPROCESSING=1             # Enable preprocessing
export XAI_METHODS="grad_cam,shap"            # XAI methods to use
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

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `FileNotFoundError: Data directory not found` | Ensure `data/images/` and `data/labels/` directories exist with files |
| `Connection refused` | Start local LLM server on configured URL |
| `NotImplementedError: Gemini model` | Use `--mode testing` or `--mode partial` with cached results |
| `Cached reasoning file not found` | Run `--mode full` first to generate cached files |
| Tests fail with import errors | Run from project root: `pytest src/test_pipeline.py -v` |

### Diagnostic Commands

```bash
python -m src.config              # Verify configuration
python main_dspy.py --show-config # Display current settings
python main_dspy.py --log-level DEBUG  # Verbose output
```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@inproceedings{schuler2025bridging,
  title     = {Bridging Explanations and Logics: Opportunities and Multimodal Language Models},
  author    = {Schuler, Nicolas Sebastian and Scotti, Vincenzo and Camilli, Matteo and Mirandola, Raffaela},
  booktitle = {International Symposium on Leveraging Applications of Formal Methods (AISoLA)},
  year      = {2025},
  doi       = {10.5445/IR/1000184717}
}
```

## Acknowledgments

This research was conducted at:

- **Karlsruhe Institute of Technology (KIT)**, Germany
- **Politecnico di Milano**, Italy

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

This is a research demonstration repository accompanying a peer-reviewed publication. For questions, collaboration inquiries, or to report issues, please contact the authors or open an issue on GitHub.

---

**Keywords**: Explainable AI (XAI) · Intersymbolic AI · Multimodal Language Models · Probabilistic Logic Programming · ProbLog · Vision-Language Models · Neuro-Symbolic AI · Abductive Reasoning
