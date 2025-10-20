"""
Configuration module for the Abduction Demo pipeline.

This module centralizes all configuration parameters including model endpoints,
file paths, processing parameters, and runtime options. Values can be overridden
via environment variables.
"""

import os
from pathlib import Path

# =============================================================================
# Runtime Configuration
# =============================================================================

# Operation mode: 1 = use cached results, 0 = run full pipeline with LLM inference
TESTING = int(os.getenv("TESTING", "1"))

# Image index to process from loaded dataset
IMAGE_INDEX = int(os.getenv("IMAGE_INDEX", "1"))

# =============================================================================
# Model Configuration
# =============================================================================

# Base URL for local LLM server (LM Studio, Ollama, etc.)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")

# API key for LLM server (empty string if not required)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# Model identifiers
MODEL_REASONING = os.getenv("MODEL_REASONING", "qwen/qwen3-30b-a3b-2507")
MODEL_CODING = os.getenv("MODEL_CODING", "gemini")  # Placeholder - needs implementation
MODEL_FEATURE_EXTRACTION = os.getenv("MODEL_FEATURE_EXTRACTION", "huggingface/qwen/qwen3-coder-30b")
MODEL_GROUNDING = os.getenv("MODEL_GROUNDING", "huggingface/qwen/qwen3-vl-8b")

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
EPSILON_PROB = float(os.getenv("EPSILON_PROB", "0.0001"))

# Image processing parameters for grounding
IMAGE_MAX_WIDTH = int(os.getenv("IMAGE_MAX_WIDTH", "512"))
IMAGE_MAX_HEIGHT = int(os.getenv("IMAGE_MAX_HEIGHT", "512"))
IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", "80"))

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
MIN_FEATURES = int(os.getenv("MIN_FEATURES", "1"))

# Maximum number of features to process
MAX_FEATURES = int(os.getenv("MAX_FEATURES", "100"))

# Valid probability range for grounding results
MIN_PROBABILITY = 0.0
MAX_PROBABILITY = 1.0

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


def print_config():
    """Print current configuration for debugging."""
    print("=" * 80)
    print("Abduction Demo Configuration")
    print("=" * 80)
    print(f"Mode: {'TESTING (cached)' if TESTING else 'FULL PIPELINE'}")
    print(f"Image Index: {IMAGE_INDEX}")
    print(f"LLM Base URL: {LLM_BASE_URL}")
    print(f"Models:")
    print(f"  - Reasoning: {MODEL_REASONING}")
    print(f"  - Coding: {MODEL_CODING}")
    print(f"  - Feature Extraction: {MODEL_FEATURE_EXTRACTION}")
    print(f"  - Grounding: {MODEL_GROUNDING}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Epsilon: {EPSILON_PROB}")
    print(f"Image Processing: {IMAGE_MAX_WIDTH}x{IMAGE_MAX_HEIGHT} @ {IMAGE_QUALITY}% quality")
    print(f"Log Level: {LOG_LEVEL}")
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
