"""
Configuration module for the Abduction Demo pipeline.

This module centralizes all configuration parameters including model endpoints,
file paths, processing parameters, and runtime options. Values can be overridden
via environment variables.
"""

import logging
import os
from pathlib import Path
from urllib.parse import urlparse

# Initialize module logger early so later warnings (e.g., missing model file) work reliably
logger = logging.getLogger(__name__)

try:
    import cv2
except ImportError:
    cv2 = None
    logger.warning("OpenCV not available, some colormap features will be disabled")


# =============================================================================
# Safe Environment Variable Parsing Helpers
# =============================================================================


def _safe_int(env_var: str, default: int) -> int:
    """Safely parse an integer environment variable with fallback to default."""
    value = os.getenv(env_var, str(default))
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid integer for {env_var}: '{value}', using default: {default}")
        return default


def _safe_float(env_var: str, default: float) -> float:
    """Safely parse a float environment variable with fallback to default."""
    value = os.getenv(env_var, str(default))
    try:
        return float(value)
    except ValueError:
        logger.warning(f"Invalid float for {env_var}: '{value}', using default: {default}")
        return default


def _safe_list(env_var: str, default: str) -> list:
    """Safely parse a comma-separated list, stripping whitespace from each item."""
    value = os.getenv(env_var, default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_url(url: str, env_var: str) -> str:
    """Validate URL format, returning original if valid or warning if malformed."""
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            logger.warning(f"Malformed URL for {env_var}: '{url}' - missing scheme or host")
        return url
    except Exception as e:
        logger.warning(f"Invalid URL for {env_var}: '{url}' - {e}")
        return url


# =============================================================================
# Valid Values Constants (used in validation)
# =============================================================================

VALID_XAI_METHODS = frozenset(["grad_cam", "integrated_grad", "shap", "saliency", "layer_cam", "occlusion"])

VALID_ENHANCEMENT_STYLES = frozenset([
    "heatmap_overlay",
    "spotlight_heatmap",
    "composite_overlay",
    "color_overlay",
    "blur_background",
    "desaturate_background",
])

VALID_CNN_DEVICES = frozenset(["auto", "cuda", "cpu", "mps"])

# =============================================================================
# Runtime Configuration
# =============================================================================

# Operation mode: 1 = use cached results, 0 = run full pipeline with LLM inference
TESTING = _safe_int("TESTING", 1)
if TESTING not in (0, 1):
    logger.warning(f"TESTING must be 0 or 1, got {TESTING}. Defaulting to 1.")
    TESTING = 1

# Image index to process from loaded dataset
IMAGE_INDEX = _safe_int("IMAGE_INDEX", 1)

# =============================================================================
# Model Configuration
# =============================================================================

# Base URL for local LLM server (LM Studio, Ollama, etc.)
_llm_url = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_BASE_URL = _validate_url(_llm_url, "LLM_BASE_URL")

# API key for LLM server (empty string if not required)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# Model identifiers
MODEL_REASONING = os.getenv("MODEL_REASONING", "qwen/qwen3-30b-a3b-2507")
MODEL_CODING = os.getenv("MODEL_CODING", "gemini")  # Placeholder - needs implementation
MODEL_FEATURE_EXTRACTION = os.getenv("MODEL_FEATURE_EXTRACTION", "huggingface/qwen/qwen3-coder-30b")
MODEL_GROUNDING = os.getenv("MODEL_GROUNDING", "openai/qwen3-vl:235b-a22b-instruct")

# =============================================================================
# File Paths
# =============================================================================

# Root directory (project root - one level up from src/)
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Data directories
DATA_DIR = ROOT_DIR / "data"
DATA_IMAGES_DIR = DATA_DIR / "images"
DATA_LABELS_DIR = DATA_DIR / "labels"
DATA_CATEGORIES_FILE = DATA_DIR / "notes.json"

# Result output directory
RESULTS_DIR = ROOT_DIR / "result-prompts"

# Cached result files
CACHED_REASONING_FILE = RESULTS_DIR / "reasoning.md"
CACHED_CODING_FILE = RESULTS_DIR / "coding.pl"
CACHED_FEATURES_FILE = RESULTS_DIR / "feature-list.txt"
CACHED_GROUNDING_FILE = RESULTS_DIR / "grounding.json"

# =============================================================================
# Processing Parameters
# =============================================================================

# Minimum probability value to avoid numerical issues in ProbLog
EPSILON_PROB = _safe_float("EPSILON_PROB", 0.0001)

# Image processing parameters for grounding
IMAGE_MAX_WIDTH = _safe_int("IMAGE_MAX_WIDTH", 512)
IMAGE_MAX_HEIGHT = _safe_int("IMAGE_MAX_HEIGHT", 512)
IMAGE_QUALITY = _safe_int("IMAGE_QUALITY", 80)

# =============================================================================
# Logging Configuration
# =============================================================================

# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# =============================================================================
# Prompt Templates
# =============================================================================

# System role for reasoning stage
REASONING_SYSTEM_ROLE = """You are a scientific expert in the classification of whether an animal is a cat or a dog. If tasked to answer questions, you shall ADHERE TO scientific facts, THINK STEP-BY-STEP, and explain your decision-making process. Focus on 'WHY' something is done, especially for complex logic, rather than 'WHAT' is done. Your answer SHOULD BE concise and direct, but still exhaustive, and avoid conversational fillers. Format your answer appropriately for better understanding. DO NOT rely on fillers like 'e.g.,', instead spell it out and try to be as complete as possible. Your reasoning MAY NOT be mutually exclusive. This is OK."""

# Question for reasoning stage
REASONING_QUESTION = """I want you to do a comparative analysis of cats and dogs. Your analysis must use the inherent traits and biological characteristics of each species. You should list each of these characteristics so that an informed decision can be made about whether a given animal depicted in an image is a cat or a dog. Please provide a detailed analysis, focusing on traits and characteristics that can be extracted from a given image. For formatting please use a list."""

# System role for coding stage
CODING_SYSTEM_ROLE = """You are an expert Problog programmer with extended knowledge in reasoning and probabilities. Given instructions and a description, you write a correct logical Problog program that expresses the given question with probabilistic theory in mind. You SHALL format your answer so that it can be directly used as an input for a Problog interpreter. DO NOT incorporate example facts or queries into the knowledge base; these will be added later by the user. If necessary, add comments to your program to provide explanations to the user. You should follow roughly the following form:
    1. Core Causal Model (LOGIC)
    2. Knowledge Base (P(Feature | Animal))
    3. Observation Model (PER-OBSERVATION)
"""

# Instruction for coding stage
CODING_INSTRUCTION = "Write a logical program for the following description:"

# =============================================================================
# Feature Detection Parameters
# =============================================================================

# True Positive Rate and False Positive Rate calculation
# For X% confidence on POSITIVE sighting: TPR = X/100, FPR = (100-X)/100
# For Y% confidence on NEGATIVE sighting: TPR = (100-Y)/100, FPR = (100-Y)/100

# =============================================================================
# Validation Thresholds
# =============================================================================

# Minimum number of features expected from extraction
MIN_FEATURES = _safe_int("MIN_FEATURES", 1)

# Maximum number of features to process
MAX_FEATURES = _safe_int("MAX_FEATURES", 100)

# Valid probability range for grounding results
MIN_PROBABILITY = 0.0
MAX_PROBABILITY = 1.0

# =============================================================================
# CNN + XAI Configuration
# =============================================================================

# Enable/disable CNN preprocessing
ENABLE_CNN_PREPROCESSING = _safe_int("ENABLE_CNN_PREPROCESSING", 1)
if ENABLE_CNN_PREPROCESSING not in (0, 1):
    logger.warning(f"ENABLE_CNN_PREPROCESSING must be 0 or 1, got {ENABLE_CNN_PREPROCESSING}. Defaulting to 1.")
    ENABLE_CNN_PREPROCESSING = 1

# CNN Model Configuration
CNN_MODEL_PATH = os.getenv("CNN_MODEL_PATH", "")  # Empty string uses ImageNet weights
CNN_INPUT_SIZE = (224, 224)
CNN_BATCH_SIZE = _safe_int("CNN_BATCH_SIZE", 1)
CNN_DEVICE = os.getenv("CNN_DEVICE", "auto")  # auto, cuda, cpu, mps
if CNN_DEVICE not in VALID_CNN_DEVICES:
    logger.warning(f"CNN_DEVICE must be one of {list(VALID_CNN_DEVICES)}, got '{CNN_DEVICE}'. Defaulting to 'auto'.")
    CNN_DEVICE = "auto"

# CNN confidence thresholds
CNN_CONFIDENCE_THRESHOLD = _safe_float("CNN_CONFIDENCE_THRESHOLD", 0.7)

# XAI Methods Configuration (with whitespace stripping)
XAI_METHODS = _safe_list("XAI_METHODS", "grad_cam,integrated_grad")
XAI_SAVE_EXPLANATIONS = _safe_int("XAI_SAVE_EXPLANATIONS", 1)
XAI_OUTPUT_DIR = RESULTS_DIR / "xai_explanations"
XAI_OVERLAY_ALPHA = _safe_float("XAI_OVERLAY_ALPHA", 0.6)

# XAI thresholds and parameters
XAI_ATTRIBUTION_THRESHOLD = _safe_float("XAI_ATTRIBUTION_THRESHOLD", 0.2)
XAI_CAM_THRESHOLD = _safe_float("XAI_CAM_THRESHOLD", 0.3)
XAI_INTEGRATED_GRAD_STEPS = _safe_int("XAI_INTEGRATED_GRAD_STEPS", 50)

# Image Enhancement Configuration (with whitespace stripping)
ENHANCEMENT_SAVE_IMAGES = _safe_int("ENHANCEMENT_SAVE_IMAGES", 1)
ENHANCEMENT_OUTPUT_DIR = RESULTS_DIR / "enhanced_images"
ENHANCEMENT_STYLES = _safe_list(
    "ENHANCEMENT_STYLES",
    "heatmap_overlay,spotlight_heatmap,composite_overlay,color_overlay,blur_background,desaturate_background",
)

# Enhancement overlay parameters
ENHANCEMENT_FOREGROUND_ALPHA = _safe_float("ENHANCEMENT_FOREGROUND_ALPHA", 0.6)
ENHANCEMENT_BACKGROUND_ALPHA = _safe_float("ENHANCEMENT_BACKGROUND_ALPHA", 0.3)
ENHANCEMENT_BLUR_INTENSITY = (35, 35)
ENHANCEMENT_DARKNESS_FACTOR = _safe_float("ENHANCEMENT_DARKNESS_FACTOR", 0.4)

# Preprocessing integration options
USE_ENHANCED_IMAGE_IN_PIPELINE = _safe_int("USE_ENHANCED_IMAGE_IN_PIPELINE", 1)
SAVE_CNN_METADATA = _safe_int("SAVE_CNN_METADATA", 1)
CNN_METADATA_OUTPUT_DIR = RESULTS_DIR / "cnn_metadata"

# Colormap preferences for different classes (guarded for cv2 availability)
if cv2 is not None:
    XAI_CAT_COLORMAP = cv2.COLORMAP_HOT
    XAI_DOG_COLORMAP = cv2.COLORMAP_COOL
else:
    # Fallback to integer codes (OpenCV internally maps these; kept for downstream use)
    XAI_CAT_COLORMAP = 2  # HOT
    XAI_DOG_COLORMAP = 8  # COOL

# =============================================================================
# Helper Functions
# =============================================================================


def validate_config():
    """
    Validate configuration values and check required resources.

    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If required directories don't exist
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}")

    if not DATA_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Images directory not found: {DATA_IMAGES_DIR}")

    if not DATA_LABELS_DIR.exists():
        raise FileNotFoundError(f"Labels directory not found: {DATA_LABELS_DIR}")

    if not DATA_CATEGORIES_FILE.exists():
        raise FileNotFoundError(f"Categories file not found: {DATA_CATEGORIES_FILE}")

    # TESTING should already be validated to 0 or 1 at module load time
    if TESTING == 1:
        # In testing mode, verify cached files exist
        if not CACHED_REASONING_FILE.exists():
            raise FileNotFoundError(f"Cached reasoning file not found: {CACHED_REASONING_FILE}")
        if not CACHED_CODING_FILE.exists():
            raise FileNotFoundError(f"Cached coding file not found: {CACHED_CODING_FILE}")
        if not CACHED_FEATURES_FILE.exists():
            raise FileNotFoundError(f"Cached features file not found: {CACHED_FEATURES_FILE}")
        if not CACHED_GROUNDING_FILE.exists():
            raise FileNotFoundError(f"Cached grounding file not found: {CACHED_GROUNDING_FILE}")

    if EPSILON_PROB <= 0 or EPSILON_PROB >= 0.5:
        raise ValueError(f"EPSILON_PROB must be in range (0, 0.5), got: {EPSILON_PROB}")

    if IMAGE_QUALITY < 1 or IMAGE_QUALITY > 100:
        raise ValueError(f"IMAGE_QUALITY must be in range [1, 100], got: {IMAGE_QUALITY}")

    # Validate CNN/XAI configuration (most already validated at module load time)
    if CNN_BATCH_SIZE < 1:
        raise ValueError(f"CNN_BATCH_SIZE must be >= 1, got: {CNN_BATCH_SIZE}")

    if not (0.0 <= CNN_CONFIDENCE_THRESHOLD <= 1.0):
        raise ValueError(f"CNN_CONFIDENCE_THRESHOLD must be in range [0.0, 1.0], got: {CNN_CONFIDENCE_THRESHOLD}")

    # Removed validation for unused CNN_CAT_PROB_THRESHOLD

    if not (0.0 <= XAI_OVERLAY_ALPHA <= 1.0):
        raise ValueError(f"XAI_OVERLAY_ALPHA must be in range [0.0, 1.0], got: {XAI_OVERLAY_ALPHA}")

    if not (0.0 <= XAI_ATTRIBUTION_THRESHOLD <= 1.0):
        raise ValueError(f"XAI_ATTRIBUTION_THRESHOLD must be in range [0.0, 1.0], got: {XAI_ATTRIBUTION_THRESHOLD}")

    if not (0.0 <= XAI_CAM_THRESHOLD <= 1.0):
        raise ValueError(f"XAI_CAM_THRESHOLD must be in range [0.0, 1.0], got: {XAI_CAM_THRESHOLD}")

    if XAI_INTEGRATED_GRAD_STEPS < 1:
        raise ValueError(f"XAI_INTEGRATED_GRAD_STEPS must be >= 1, got: {XAI_INTEGRATED_GRAD_STEPS}")

    if not (0.0 <= ENHANCEMENT_FOREGROUND_ALPHA <= 1.0):
        raise ValueError(
            f"ENHANCEMENT_FOREGROUND_ALPHA must be in range [0.0, 1.0], got: {ENHANCEMENT_FOREGROUND_ALPHA}"
        )

    if not (0.0 <= ENHANCEMENT_BACKGROUND_ALPHA <= 1.0):
        raise ValueError(
            f"ENHANCEMENT_BACKGROUND_ALPHA must be in range [0.0, 1.0], got: {ENHANCEMENT_BACKGROUND_ALPHA}"
        )

    if not (0.0 <= ENHANCEMENT_DARKNESS_FACTOR <= 1.0):
        raise ValueError(f"ENHANCEMENT_DARKNESS_FACTOR must be in range [0.0, 1.0], got: {ENHANCEMENT_DARKNESS_FACTOR}")

    # Validate XAI methods using module-level constant
    for method in XAI_METHODS:
        if method not in VALID_XAI_METHODS:
            raise ValueError(f"Invalid XAI method: '{method}'. Valid methods: {list(VALID_XAI_METHODS)}")

    # Validate enhancement styles using module-level constant
    for style in ENHANCEMENT_STYLES:
        if style not in VALID_ENHANCEMENT_STYLES:
            raise ValueError(f"Invalid enhancement style: '{style}'. Valid styles: {list(VALID_ENHANCEMENT_STYLES)}")

    # Check CNN model file if specified
    if CNN_MODEL_PATH and CNN_MODEL_PATH.strip():
        model_path = Path(CNN_MODEL_PATH)
        if not model_path.exists():
            logger.warning(f"CNN model file not found: {model_path}. Using ImageNet weights instead.")


def print_config():
    """Print current configuration for debugging."""
    print("=" * 80)
    print("Abduction Demo Configuration")
    print("=" * 80)
    print(f"Mode: {'TESTING (cached)' if TESTING else 'FULL PIPELINE'}")
    print(f"Image Index: {IMAGE_INDEX}")
    print(f"LLM Base URL: {LLM_BASE_URL}")
    print("Models:")
    print(f"  - Reasoning: {MODEL_REASONING}")
    print(f"  - Coding: {MODEL_CODING}")
    print(f"  - Feature Extraction: {MODEL_FEATURE_EXTRACTION}")
    print(f"  - Grounding: {MODEL_GROUNDING}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Epsilon: {EPSILON_PROB}")
    print(f"Image Processing: {IMAGE_MAX_WIDTH}x{IMAGE_MAX_HEIGHT} @ {IMAGE_QUALITY}% quality")
    print(f"Log Level: {LOG_LEVEL}")
    print()

    # CNN + XAI Configuration
    print("CNN + XAI Configuration:")
    print(f"CNN Preprocessing: {'ENABLED' if ENABLE_CNN_PREPROCESSING else 'DISABLED'}")
    if ENABLE_CNN_PREPROCESSING:
        print(f"CNN Model: {'Custom: ' + CNN_MODEL_PATH if CNN_MODEL_PATH else 'ImageNet-pretrained ResNet-50'}")
        print(f"CNN Device: {CNN_DEVICE}")
        print(f"CNN Confidence Threshold: {CNN_CONFIDENCE_THRESHOLD}")
        print(f"XAI Methods: {', '.join(XAI_METHODS)}")
        print(f"Enhancement Styles: {', '.join(ENHANCEMENT_STYLES)}")
        print(f"Use Enhanced Images: {'YES' if USE_ENHANCED_IMAGE_IN_PIPELINE else 'NO'}")
    print("=" * 80)


if __name__ == "__main__":
    # When run directly, validate and print config
    try:
        validate_config()
        print_config()
        print("\n✓ Configuration is valid!")
    except (ValueError, FileNotFoundError) as e:
        print(f"\n✗ Configuration error: {e}")
        exit(1)
