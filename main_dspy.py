import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import dspy
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from problog import get_evaluatable
from problog.program import PrologString

from src import config
from src.data import load_data
from src.img_utils import encode_base64_resized
from src.validation import (
    ValidationError,
    sanitize_grounding_results,
    validate_feature_list,
    validate_grounding_results,
    validate_problog_program,
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
)
logger = logging.getLogger(__name__)

# Import CNN + XAI preprocessing pipeline (EfficientNet-based)
try:
    from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

    CNN_PREPROCESSING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CNN + XAI preprocessing not available: {e}")
    CNN_PREPROCESSING_AVAILABLE = False


@dataclass
class RuntimeConfig:
    """
    Runtime configuration for pipeline execution.

    This dataclass holds all runtime parameters that can be set from command-line
    arguments, keeping the config module immutable after import.

    Attributes:
        mode: Operation mode ('testing', 'partial', or 'full')
        image_index: Index of image to process from dataset
        output_dir: Directory to save results
        log_level: Logging level
        llm_base_url: Base URL for LLM server
        image_path: Optional custom image path (overrides image_index)
        highlighted_only: Only consider highlighted features in grounding
        disable_cnn: Disable CNN preprocessing
        disable_xai: Disable XAI explanations
        disable_enhancement: Disable image enhancement
        use_original_image: Use original image instead of enhanced
        force_preprocessing: Force preprocessing even in testing mode
    """

    mode: str
    image_index: int
    output_dir: Path
    log_level: str
    llm_base_url: str
    llm_api_key: str
    model_reasoning: str
    model_coding: str
    model_feature_extraction: str
    model_grounding: str
    image_path: Optional[Path] = None
    highlighted_only: bool = False
    disable_cnn: bool = False
    disable_xai: bool = False
    disable_enhancement: bool = False
    use_original_image: bool = False
    force_preprocessing: bool = False


def cleanup_temp_files(temp_paths: list[Path]) -> None:
    """
    Clean up temporary files created during pipeline execution.

    Args:
        temp_paths: List of Path objects to temporary files to delete
    """
    for path in temp_paths:
        try:
            if path.exists():
                path.unlink()
                logger.debug(f"Cleaned up temp file: {path}")
        except OSError as e:
            logger.warning(f"Failed to clean up temp file {path}: {e}")


def log_cnn_xai_context(preprocessing_result, cnn_classification, stage_label: str):
    """
    Consolidated logging for CNN/XAI context to avoid duplication.
    """
    if not cnn_classification and not (preprocessing_result and preprocessing_result.get("explanations")):
        return
    logger.info("-" * 80)
    logger.info(f"CNN/XAI Context ({stage_label})")
    if cnn_classification:
        logger.info(
            "CNN Baseline: %s (confidence: %.3f)",
            cnn_classification["predicted_class"],
            cnn_classification["confidence"],
        )
        logger.info("CNN Probabilities: %s", cnn_classification["probabilities"])
    if preprocessing_result and preprocessing_result.get("explanations"):
        xai_methods_used = [
            m
            for m in preprocessing_result["explanations"].keys()
            if m not in {"image_name", "target_class", "target_index", "original_image", "input_tensor"}
        ]
        if xai_methods_used:
            logger.info("XAI Methods Applied: %s", xai_methods_used)


def build_enhanced_temp_path(stem: str) -> Path:
    """
    Build (and ensure) a temp path for an enhanced image tied to the current image stem.
    """
    tmp_dir = config.RESULTS_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    return tmp_dir / f"{stem}_enhanced.png"


def reasoning():
    """
    Stage 1: Generate scientific reasoning about cat vs. dog classification.

    Uses an LLM to perform comparative biological analysis of cats and dogs,
    identifying key distinguishing characteristics that can be detected in images.

    Returns:
        str: Detailed comparative analysis in markdown format, listing physical
             characteristics like skull morphology, ear shape, eye position, etc.

    Example output structure:
        - Skull Morphology: cats have rounded skulls, dogs have elongated...
        - Ear Shape: cats have pointed upright ears, dogs have variable...

    Note:
        Uses Qwen3-30B model via local LM server (http://127.0.0.1:1234/v1).
        Result is typically cached to result-prompts/reasoning.md.
    """
    logger.info("Stage 1: Generating scientific reasoning about cat vs. dog classification")

    try:
        template_reasoning = ChatPromptTemplate.from_messages([
            ("system", "{role_reasoning}"),
            ("human", "Question: {question_reasoning}"),
        ])
        model_reasoning = ChatOpenAI(
            model=config.MODEL_REASONING,
            base_url=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
        )
        reasoning_chain = template_reasoning | model_reasoning | StrOutputParser()

        logger.debug(f"Using model: {config.MODEL_REASONING} at {config.LLM_BASE_URL}")

        description = reasoning_chain.invoke({
            "role_reasoning": config.REASONING_SYSTEM_ROLE,
            "question_reasoning": config.REASONING_QUESTION,
        })

        if not description or not description.strip():
            raise ValueError("LLM returned empty reasoning description")

        logger.info(f"Generated reasoning description ({len(description)} characters)")
        return description

    except Exception as e:
        logger.error(f"Failed to generate reasoning: {e}")
        raise RuntimeError(f"Reasoning stage failed: {e}") from e


def coding(description: str):
    """
    Stage 2: Translate natural language reasoning into ProbLog program.

    Converts the comparative analysis from Stage 1 into a formal probabilistic
    logic program using ProbLog syntax. The program encodes:
    - Prior probabilities for cat/dog classification
    - Conditional probabilities P(feature | animal_type)
    - Observation model linking features to observations

    Args:
        description: Natural language reasoning from reasoning() function,
                     containing comparative analysis of cat/dog characteristics.

    Returns:
        str: ProbLog program with following structure:
             1. Core Causal Model (priors and rules)
             2. Knowledge Base (conditional probabilities)
             3. Observation Model (TPR/FPR for features)

    Example output:
        ```prolog
        0.5::is_cat; 0.5::is_dog.
        P::feature(F) :- is_cat, prob_cat(F, P).
        prob_cat(small_muzzle, 0.95). prob_dog(small_muzzle, 0.2).
        ...
        ```

    Note:
        Currently uses a placeholder gemini() function. Replace with actual
        Gemini API call or alternative LLM for full functionality.
    """
    logger.info("Stage 2: Translating reasoning into ProbLog program")

    try:
        template_coding = ChatPromptTemplate.from_messages([
            ("system", "{role_coding}"),
            ("human", "Instructions: {instruction}\n Description: {description}"),
        ])

        def gemini():
            """
            Placeholder for Gemini API implementation.

            IMPORTANT: This function is NOT implemented. To use --mode=full, you must either:
            1. Implement this function with a valid LLM API call (see example below), OR
            2. Use --mode=testing or --mode=partial with pre-cached results

            Example implementation using langchain-google-genai:
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model="gemini-1.5-pro",
                    google_api_key=os.getenv("GOOGLE_API_KEY"),
                )

            Or use the local LLM server (like reasoning stage):
                return ChatOpenAI(
                    model=config.MODEL_CODING,
                    base_url=config.LLM_BASE_URL,
                    api_key=config.LLM_API_KEY,
                )

            Returns:
                A LangChain chat model compatible with ChatPromptTemplate.

            Raises:
                NotImplementedError: Always raised until this function is implemented.
            """
            raise NotImplementedError(
                "Gemini model is not implemented.\n"
                "To run the coding stage (--mode=full), either:\n"
                "  1. Implement gemini() with Google's Gemini API or alternative LLM\n"
                "  2. Use --mode=testing or --mode=partial with pre-cached results\n"
                f"See config.MODEL_CODING (current: {config.MODEL_CODING}) for expected model."
            )

        model_coding = gemini()
        coding_chain = template_coding | model_coding | StrOutputParser()

        coding_description = coding_chain.invoke({
            "role_coding": config.CODING_SYSTEM_ROLE,
            "instruction": config.CODING_INSTRUCTION,
            "description": description,
        })

        # Validate the generated program
        validate_problog_program(coding_description)

        logger.info(f"Generated ProbLog program ({len(coding_description)} characters)")
        return coding_description

    except NotImplementedError:
        # Re-raise NotImplementedError for gemini
        raise
    except Exception as e:
        logger.error(f"Failed to generate ProbLog program: {e}")
        raise RuntimeError(f"Coding stage failed: {e}") from e


def extract_feature_list(problog_program: str) -> list[str]:
    """
    Stage 3: Extract feature names (atoms) from ProbLog program.

    Parses the ProbLog program to identify all feature atoms that need to be
    grounded from visual observations. Features are the predicates used in
    prob_cat() and prob_dog() facts, such as 'small_muzzle', 'pointed_ears', etc.

    Args:
        problog_program: ProbLog program string from coding() function containing
                         prob_cat/prob_dog predicates with feature names.

    Returns:
        List of feature name strings to be grounded in images.

    Example:
        >>> program = "prob_cat(small_muzzle, 0.95). prob_dog(small_muzzle, 0.2)."
        >>> extract_feature_list(program)
        ['small_muzzle', 'pointed_ears', 'vertical_pupils', ...]

    Note:
        Uses Qwen3-Coder-30B via DSPy's structured prompting for reliable parsing.
    """
    logger.info("Stage 3: Extracting feature names from ProbLog program")

    try:
        lm = dspy.LM(
            config.MODEL_FEATURE_EXTRACTION, api_base=config.LLM_BASE_URL, api_key=config.LLM_API_KEY, cache=False
        )
        dspy.configure(lm=lm)

        logger.debug(f"Using model: {config.MODEL_FEATURE_EXTRACTION}")

        class FeatureNames(dspy.Signature):
            """Extracting Feature names from a Problog program."""

            problog_program: str = dspy.InputField(desc="Problog program")
            features: list[str] = dspy.OutputField(desc="List of Feature names (atoms) of the Problog program")

        prompt = dspy.ChainOfThought(FeatureNames)
        features = prompt(problog_program=problog_program).features

        # Validate extracted features
        validate_feature_list(features)

        # Transform underscores to spaces
        features = [feature.replace("_", " ") for feature in features]

        logger.info(f"Extracted {len(features)} features from ProbLog program")
        logger.debug(f"Features: {features[:5]}..." if len(features) > 5 else f"Features: {features}")

        return features

    except ValidationError as e:
        logger.error(f"Feature list validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to extract features: {e}")
        raise RuntimeError(f"Feature extraction stage failed: {e}") from e


def grounding(feature_list: list[str], image_path: Path, highlighted_only: bool = False):
    """
    Stage 4: Ground features in image using Vision-Language Model.

    Analyzes an input image to detect the presence of each feature from the
    feature list, assigning a probability score (0.0-1.0) for each feature.
    This bridges symbolic reasoning with visual perception.

    Args:
        feature_list: List of feature names to detect (e.g., ['small_muzzle',
                      'pointed_ears', 'vertical_pupils']).
        image_path: Path to the image file to analyze.
        highlighted_only: If True, only consider features that are highlighted/emphasized
                          in the image. If False, consider all features present regardless
                          of highlighting. Defaults to False.

    Returns:
        Dictionary mapping feature names to probability scores (0.0 to 1.0).

    Example:
        >>> features = ['small_muzzle', 'pointed_ears']
        >>> grounding(features, Path('data/images/cat.jpg'))
        {'small_muzzle': 0.92, 'pointed_ears': 0.88}

    Note:
        - Uses Qwen3-VL-8B multimodal model via DSPy
        - Image is automatically resized to 512x512 with 80% quality
        - Requires local LM server with vision model support
    """
    logger.info(f"Stage 4: Grounding {len(feature_list)} features in image")
    logger.debug(f"Image path: {image_path}")

    try:
        # Validate inputs
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        if highlighted_only:
            grounding_desc = """A json, mapping feature names to feature being FULLY HIGHLIGHTED BY THE HEATMAP present in the input image.
        ONLY FULLY HIGHLIGHTED FEATURES BY THE HEATMAP ARE CONSIDERED! FEATURES THAT ARE NOT FULLY HIGHLIGHTED BY THE HEATMAP HAVE A '0.0' PROBABILITY! If it is too ambigious, omit the feature or give it a '0.0' propability.
        """
            image_desc = "Image to be analyzed for red highlighted features."
        else:
            grounding_desc = """A json, mapping feature names to the probability of the feature being present AND recognized in the input image.
            Example:
                {'feature_A': 0.9, 'feature_B': 0.1}
            """
            image_desc = "Image to be analyzed for present features"

        print(config.MODEL_GROUNDING)
        print(config.LLM_BASE_URL)
        print(config.LLM_API_KEY)
        lm = dspy.LM(
            config.MODEL_GROUNDING,
            api_base=config.LLM_BASE_URL,
            api_key=config.LLM_API_KEY,
            cache=False,
            max_tokens=None,
        )
        dspy.configure(lm=lm)

        logger.debug(f"Using model: {config.MODEL_GROUNDING}")

        class Grounding(dspy.Signature):
            """Matches features from a given list to the given image and returns the propability of the features being correctly detected."""

            feature_list: list[str] = dspy.InputField(desc="List of possible features present in the image")
            image: dspy.Image = dspy.InputField(desc=image_desc)
            grounding: dict[str, float] = dspy.OutputField(desc=grounding_desc)

        prompt = dspy.ChainOfThought(Grounding)
        result = prompt(
            feature_list=feature_list,
            image=dspy.Image(
                url=f"data:image/jpg;base64,{encode_base64_resized(image_path, max_width=config.IMAGE_MAX_WIDTH, max_height=config.IMAGE_MAX_HEIGHT, quality=config.IMAGE_QUALITY)}"
            ),
        ).grounding

        # Validate and sanitize grounding results
        validate_grounding_results(result)
        result = sanitize_grounding_results(result)

        # Transform spaces back to underscores for ProbLog program compatibility
        result = {feature.replace(" ", "_"): prob for feature, prob in result.items()}

        logger.info(f"Grounding completed: detected {len(result)} feature probabilities")
        # Log confidence statistics
        if result:
            avg_conf = sum(result.values()) / len(result)
            logger.debug(f"Average confidence: {avg_conf:.3f}")

        return result

    except FileNotFoundError:
        raise
    except ValidationError as e:
        logger.error(f"Grounding validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to ground features: {e}")
        raise RuntimeError(f"Grounding stage failed: {e}") from e


def display_grounding_results(grounding_res: dict[str, float]) -> None:
    """
    Display grounding results in a fancy, visually appealing format.

    This function emphasizes the VLM's output by grouping features by confidence
    level and displaying them with visual indicators.

    Args:
        grounding_res: Dictionary mapping feature names to probability scores (0.0-1.0)
    """
    if not grounding_res:
        logger.warning("No grounding results to display")
        return

    # Sort features by probability (descending)
    sorted_features = sorted(grounding_res.items(), key=lambda x: x[1], reverse=True)

    # Group features by confidence level
    high_conf = [(f, p) for f, p in sorted_features if p >= 0.8]
    medium_conf = [(f, p) for f, p in sorted_features if 0.5 <= p < 0.8]
    low_conf = [(f, p) for f, p in sorted_features if p < 0.5]

    # Helper function to create visual bar
    def create_bar(probability: float, width: int = 40) -> str:
        filled = int(probability * width)
        empty = width - filled
        return "█" * filled + "░" * empty

    # Helper function to get confidence emoji
    def get_emoji(probability: float) -> str:
        if probability >= 0.9:
            return "🔥"
        elif probability >= 0.8:
            return "✓"
        elif probability >= 0.5:
            return "◆"
        else:
            return "·"

    # Display header
    logger.info("")
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 22 + "🤖 VISION-LANGUAGE MODEL OUTPUT 🤖" + " " * 22 + "║")
    logger.info("║" + " " * 26 + "Stage 4: Feature Grounding" + " " * 26 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")
    logger.info(f"Model: {config.MODEL_GROUNDING}")
    logger.info(f"Total features analyzed: {len(grounding_res)}")
    logger.info("")

    # Display HIGH confidence features
    if high_conf:
        logger.info("━" * 80)
        logger.info(f"🔥 HIGH CONFIDENCE DETECTIONS (≥0.80) - {len(high_conf)} features")
        logger.info("━" * 80)
        for feature, prob in high_conf:
            emoji = get_emoji(prob)
            bar = create_bar(prob)
            # Convert underscores to spaces for display
            display_name = feature.replace("_", " ").title()
            logger.info(f"{emoji} {display_name:<35} {prob:.3f} │{bar}│")
        logger.info("")

    # Display MEDIUM confidence features
    if medium_conf:
        logger.info("─" * 80)
        logger.info(f"◆ MEDIUM CONFIDENCE DETECTIONS (0.50-0.79) - {len(medium_conf)} features")
        logger.info("─" * 80)
        for feature, prob in medium_conf:
            emoji = get_emoji(prob)
            bar = create_bar(prob)
            display_name = feature.replace("_", " ").title()
            logger.info(f"{emoji} {display_name:<35} {prob:.3f} │{bar}│")
        logger.info("")

    # Display LOW confidence features (collapsed/summary)
    if low_conf:
        logger.info("·" * 80)
        logger.info(f"· LOW CONFIDENCE DETECTIONS (<0.50) - {len(low_conf)} features")
        logger.info("·" * 80)
        # Show top 5 low confidence features
        for feature, prob in low_conf[:5]:
            emoji = get_emoji(prob)
            bar = create_bar(prob)
            display_name = feature.replace("_", " ").title()
            logger.info(f"{emoji} {display_name:<35} {prob:.3f} │{bar}│")
        if len(low_conf) > 5:
            logger.info(f"  ... and {len(low_conf) - 5} more low-confidence features")
        logger.info("")

    logger.info("=" * 80)
    logger.info("")


def execute_logic_program(problog_program: str, grounding: dict[str, float]) -> tuple[float, float]:
    """
    Execute a ProbLog program with evidence from grounding results.

    This function takes the base ProbLog program and adds evidence based on the
    grounding results (feature probabilities extracted from images). It then runs
    ProbLog inference to determine the probability of the animal being a cat or dog.

    Args:
        problog_program: Base ProbLog program as string containing the knowledge base
        grounding: Dictionary mapping feature names to probabilities (0.0 to 1.0)

    Returns:
        Tuple of (cat_probability, dog_probability)

    Example:
        >>> grounding = {"small_muzzle": 0.9, "retractable_claws": 0.8}
        >>> p_cat, p_dog = execute_logic_program(program, grounding)
        >>> print(f"Cat: {p_cat:.3f}, Dog: {p_dog:.3f}")
    """
    # Remove the example evidence section from the base program
    lines = problog_program.split("\n")
    filtered_lines = []
    skip_section = False

    for line in lines:
        # Skip the example evidence section that starts with "% --- Example Scenario:"
        if "% --- Example Scenario:" in line:
            skip_section = True
            continue
        elif (
            line.startswith("% =============================================================================")
            and skip_section
        ):
            # Stop skipping when we reach the query section
            if "5. QUERY" in line:
                skip_section = False
            continue
        elif skip_section and (
            line.startswith("confidence(")
            or line.startswith("evidence(")
            or line.startswith("% confidence(")
            or line.startswith("% evidence(")
        ):
            # Skip example confidence and evidence statements (both active and commented)
            continue
        elif skip_section and line.strip().startswith("%"):
            # Skip comments in the example section
            continue
        elif skip_section and not line.strip():
            # Skip empty lines in the example section
            continue
        # Also filter out commented queries since we add them back
        elif line.strip().startswith("% query("):
            continue
        else:
            skip_section = False
            filtered_lines.append(line)

    # Reconstruct the program without example evidence
    base_program = "\n".join(filtered_lines)

    # Add evidence based on grounding results
    evidence_lines = []
    evidence_lines.append("\n% =============================================================================")
    evidence_lines.append("% ==  4. EVIDENCE FROM GROUNDING")
    evidence_lines.append("% =============================================================================")

    for feature_name, probability in grounding.items():
        # Clamp probability to avoid numerical issues with extreme values
        prob = max(config.EPSILON_PROB, min(1.0 - config.EPSILON_PROB, probability))

        # Set up observation model parameters
        # TPR (True Positive Rate): probability of reporting the feature when it's present
        # FPR (False Positive Rate): probability of reporting the feature when it's absent
        tpr = prob  # High confidence means high TPR
        fpr = 1.0 - prob  # High confidence means low FPR

        evidence_lines.append(f"confidence(grounding_observer, {feature_name}, {tpr:.6f}, {fpr:.6f}).")
        evidence_lines.append(f"evidence(report(grounding_observer, {feature_name}), true).")

    # Add queries section
    evidence_lines.append("\n% =============================================================================")
    evidence_lines.append("% ==  5. QUERY")
    evidence_lines.append("% =============================================================================")
    evidence_lines.append("% Asks the program for the final probability of each hypothesis.")
    evidence_lines.append("query(is_cat).")
    evidence_lines.append("query(is_dog).")

    # Combine base program with evidence
    full_program = base_program + "\n".join(evidence_lines)

    # Execute ProbLog inference
    logger.info("Stage 5: Executing ProbLog inference")
    logger.debug(f"Running inference with {len(grounding)} evidence items")

    try:
        # Create ProbLog program from string
        problog_model = PrologString(full_program)

        # Get evaluatable model for inference
        evaluatable = get_evaluatable().create_from(problog_model)

        # Evaluate all queries in the program
        results = evaluatable.evaluate()

        # Extract probabilities for cat and dog classifications
        cat_prob = 0.0
        dog_prob = 0.0

        for query, probability in results.items():
            query_str = str(query)
            if "is_cat" in query_str:
                cat_prob = float(probability)
            elif "is_dog" in query_str:
                dog_prob = float(probability)

        logger.info(f"Inference complete: P(cat)={cat_prob:.4f}, P(dog)={dog_prob:.4f}")
        return cat_prob, dog_prob

    except Exception as e:
        logger.error(f"Error executing ProbLog program: {e}", exc_info=True)
        logger.warning("Returning default equal probabilities (0.5, 0.5)")
        # Return default equal probabilities in case of error
        return 0.5, 0.5


def display_classification_results(
    image_name: str,
    actual_label: str,
    p_cat: float,
    p_dog: float,
    preprocessing_result: dict | None,
    cnn_classification: dict | None,
) -> str:
    """
    Display final classification results in a formatted output.

    Args:
        image_name: Name of the processed image file
        actual_label: Ground truth label (or "unknown")
        p_cat: Probability of cat classification
        p_dog: Probability of dog classification
        preprocessing_result: Results from CNN+XAI preprocessing (for context)
        cnn_classification: CNN classification result (for comparison)

    Returns:
        Predicted class ("cat" or "dog")
    """
    logger.info("=" * 80)
    logger.info("CLASSIFICATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Image: {image_name}")
    logger.info(f"Actual Label: {actual_label}")
    logger.info(f"Cat Probability: {p_cat:.4f}")
    logger.info(f"Dog Probability: {p_dog:.4f}")

    predicted = "cat" if p_cat > p_dog else "dog"
    logger.info(f"Predicted (ProbLog Inference): {predicted}")

    if actual_label != "unknown":
        logger.info(f"Correct: {'YES' if predicted == actual_label else 'NO'}")

    # Comparative summary with CNN baseline (if available)
    log_cnn_xai_context(preprocessing_result, cnn_classification, "final")
    if cnn_classification:
        agreement = predicted == cnn_classification.get("predicted_class")
        logger.info(f"Model Agreement: {'AGREEMENT' if agreement else 'DISAGREEMENT'}")

    logger.info("=" * 80)
    logger.info("Abduction Demo Complete")
    logger.info("=" * 80)

    return predicted


def run_testing_mode() -> tuple[str, str, list[str], dict[str, float]]:
    """
    Execute testing mode: load all results from cached files.

    Returns:
        Tuple of (reasoning_description, problog_program, feature_list, grounding_res)

    Raises:
        FileNotFoundError: If cached files are missing
        ValidationError: If cached data is invalid
    """
    logger.info("Running in TESTING mode (using cached results)")

    reasoning_description = config.CACHED_REASONING_FILE.read_text()
    problog_program = config.CACHED_CODING_FILE.read_text()
    feature_list = [line.strip() for line in config.CACHED_FEATURES_FILE.read_text().splitlines()]
    grounding_res = json.loads(config.CACHED_GROUNDING_FILE.read_text())

    # Validate loaded data
    validate_problog_program(problog_program)
    validate_feature_list(feature_list)
    validate_grounding_results(grounding_res)
    grounding_res = sanitize_grounding_results(grounding_res)

    # Transform underscores to spaces in feature names
    feature_list = [feature.replace("_", " ") for feature in feature_list]

    logger.debug("Loaded and validated all cached results")
    return reasoning_description, problog_program, feature_list, grounding_res


def run_partial_mode(
    pipeline_image_path: Path,
    highlighted_only: bool,
    preprocessing_result: dict | None,
    cnn_classification: dict | None,
) -> tuple[str, str, list[str], dict[str, float]]:
    """
    Execute partial mode: load cached reasoning/coding, run live grounding.

    Args:
        pipeline_image_path: Path to image for grounding (original or enhanced)
        highlighted_only: Only consider highlighted features
        preprocessing_result: Results from CNN+XAI preprocessing (for logging)
        cnn_classification: CNN classification result (for logging)

    Returns:
        Tuple of (reasoning_description, problog_program, feature_list, grounding_res)

    Raises:
        FileNotFoundError: If cached files are missing
        ValidationError: If validation fails
        RuntimeError: If grounding fails
    """
    logger.info("Running in PARTIAL mode (starting after coding step)")

    # Ensure output directory exists
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load cached reasoning, coding, and features
    reasoning_description = config.CACHED_REASONING_FILE.read_text()
    problog_program = config.CACHED_CODING_FILE.read_text()
    feature_list = [line.strip() for line in config.CACHED_FEATURES_FILE.read_text().splitlines()]

    # Validate loaded cached data
    validate_problog_program(problog_program)
    validate_feature_list(feature_list)

    # Transform underscores to spaces in feature names
    feature_list = [feature.replace("_", " ") for feature in feature_list]

    logger.debug("Loaded and validated cached reasoning, coding, and features")

    # Run grounding on the image (original or enhanced)
    grounding_res = grounding(
        feature_list=feature_list, image_path=pipeline_image_path, highlighted_only=highlighted_only
    )

    # Display fancy grounding results
    display_grounding_results(grounding_res)
    log_cnn_xai_context(preprocessing_result, cnn_classification, "partial grounding")

    config.CACHED_GROUNDING_FILE.write_text(json.dumps(grounding_res, indent=2))
    logger.debug(f"Saved grounding results to {config.CACHED_GROUNDING_FILE}")

    return reasoning_description, problog_program, feature_list, grounding_res


def run_full_mode(
    pipeline_image_path: Path,
    highlighted_only: bool,
    preprocessing_result: dict | None,
    cnn_classification: dict | None,
) -> tuple[str, str, list[str], dict[str, float]]:
    """
    Execute full mode: run complete pipeline with live LLM inference.

    Args:
        pipeline_image_path: Path to image for grounding (original or enhanced)
        highlighted_only: Only consider highlighted features
        preprocessing_result: Results from CNN+XAI preprocessing (for logging)
        cnn_classification: CNN classification result (for logging)

    Returns:
        Tuple of (reasoning_description, problog_program, feature_list, grounding_res)

    Raises:
        NotImplementedError: If Gemini API is not implemented
        ValidationError: If validation fails
        RuntimeError: If any pipeline stage fails
    """
    logger.info("Running in FULL PIPELINE mode (live LLM inference)")

    # Ensure output directory exists
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    reasoning_description = reasoning()
    config.CACHED_REASONING_FILE.write_text(reasoning_description)
    logger.debug(f"Saved reasoning to {config.CACHED_REASONING_FILE}")

    # ONLY FOR DEMONSTRATION PURPOSES
    problog_program = coding(reasoning_description)
    config.CACHED_CODING_FILE.write_text(problog_program)
    logger.debug(f"Saved ProbLog program to {config.CACHED_CODING_FILE}")

    feature_list = extract_feature_list(problog_program=problog_program)
    config.CACHED_FEATURES_FILE.write_text("\n".join(feature_list))
    logger.debug(f"Saved feature list to {config.CACHED_FEATURES_FILE}")

    grounding_res = grounding(
        feature_list=feature_list, image_path=pipeline_image_path, highlighted_only=highlighted_only
    )

    # Display fancy grounding results
    display_grounding_results(grounding_res)
    log_cnn_xai_context(preprocessing_result, cnn_classification, "full grounding")

    config.CACHED_GROUNDING_FILE.write_text(json.dumps(grounding_res, indent=2))
    logger.debug(f"Saved grounding results to {config.CACHED_GROUNDING_FILE}")

    return reasoning_description, problog_program, feature_list, grounding_res


def create_runtime_config(args) -> RuntimeConfig:
    """
    Convert parsed arguments to RuntimeConfig dataclass.

    Args:
        args: Parsed command-line arguments from argparse

    Returns:
        RuntimeConfig instance with all runtime parameters
    """
    return RuntimeConfig(
        mode=args.mode,
        image_index=args.image_index,
        output_dir=args.output_dir,
        log_level=args.log_level,
        llm_base_url=args.llm_base_url,
        llm_api_key=args.llm_api_key,
        model_reasoning=args.model_reasoning,
        model_coding=args.model_coding,
        model_feature_extraction=args.model_feature_extraction,
        model_grounding=args.model_grounding,
        image_path=args.image_path,
        highlighted_only=args.highlighted_only,
        disable_cnn=args.disable_cnn,
        disable_xai=args.disable_xai,
        disable_enhancement=args.disable_enhancement,
        use_original_image=args.use_original_image,
        force_preprocessing=args.force_preprocessing,
    )


def main(runtime_config: RuntimeConfig):
    """
    Main pipeline orchestration function.

    Executes the complete 5-stage pipeline:
    1. Reasoning: Generate comparative analysis (or load cached)
    2. Coding: Translate to ProbLog program (or load cached)
    3. Feature Extraction: Extract feature names (or load cached)
    4. Grounding: Detect features in image (or load cached)
    5. Logic Execution: Run ProbLog inference to classify animal

    The pipeline can run in three modes:
    - testing: Uses cached results from result-prompts/ directory
    - full: Runs complete pipeline with live LLM inference
    - partial: Starts after coding step, uses cached reasoning and coding results

    Args:
        runtime_config: Runtime configuration dataclass with all execution parameters.

    Returns:
        Tuple of (cat_probability, dog_probability)

    Note:
        Uses runtime_config.image_index to select image from dataset,
        or runtime_config.image_path for custom images.
    """
    logger.info("=" * 80)
    logger.info("Starting Abduction Demo")
    logger.info("=" * 80)

    # Validate configuration
    try:
        config.validate_config()
        logger.info("Configuration validated successfully")
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Configuration validation failed: {e}")
        raise

    # Handle custom image path or load from dataset
    if runtime_config.image_path is not None:
        # Use custom image path
        logger.info(f"Using custom image path: {runtime_config.image_path}")
        if not runtime_config.image_path.exists():
            logger.error(f"Custom image file not found: {runtime_config.image_path}")
            raise FileNotFoundError(f"Image file not found: {runtime_config.image_path}")

        # Create a simple object to hold image info (without labels)
        class CustomImage:
            def __init__(self, image_path: Path):
                self.image_path = image_path
                self.cl = "unknown"  # No label available for custom images

        labeled_image = CustomImage(runtime_config.image_path)
        logger.info(f"Processing custom image: {labeled_image.image_path.name}")
    else:
        # Load data from predefined dataset
        logger.info("Loading labeled images from data directory")
        labeled_images = load_data()
        logger.info(f"Loaded {len(labeled_images)} images")

        # Select image to process
        if runtime_config.image_index >= len(labeled_images):
            logger.error(f"IMAGE_INDEX {runtime_config.image_index} out of range (0-{len(labeled_images) - 1})")
            raise ValueError(f"IMAGE_INDEX must be in range 0-{len(labeled_images) - 1}")

        labeled_image = labeled_images[runtime_config.image_index]
        logger.info(
            f"Processing image {runtime_config.image_index}: {labeled_image.image_path.name} (actual label: {labeled_image.cl})"
        )

    # CNN + XAI Preprocessing Stage (conditionally enabled; skipped in testing mode unless --force-preprocessing is used)
    preprocessing_result = None
    pipeline_image_path = labeled_image.image_path  # Default to original image
    cnn_classification = None  # Persist CNN classification for final reporting
    temp_files: list[Path] = []  # Track temp files for cleanup

    run_preprocessing = (
        CNN_PREPROCESSING_AVAILABLE
        and config.ENABLE_CNN_PREPROCESSING
        and (runtime_config.mode != "testing" or runtime_config.force_preprocessing)
        and not runtime_config.disable_cnn
    )

    if run_preprocessing:
        logger.info("=" * 80)
        logger.info("CNN + XAI PREPROCESSING STAGE")
        logger.info("=" * 80)

        try:
            # Initialize preprocessing pipeline with command-line options (use config.RESULTS_DIR for consistency)
            preprocessing_pipeline = PreprocessingPipeline(
                enable_cnn=not runtime_config.disable_cnn,
                enable_xai=not runtime_config.disable_xai,
                enable_enhancement=not runtime_config.disable_enhancement,
            )

            # Process the image
            preprocessing_result = preprocessing_pipeline.process_image(
                labeled_image.image_path, save_prefix=labeled_image.image_path.stem
            )

            # Display preprocessing results (with null-safe access)
            if preprocessing_result and preprocessing_result.get("classification"):
                cls_result = preprocessing_result["classification"]
                cnn_classification = cls_result  # persist for final printing
                predicted_class = cls_result.get("predicted_class", "unknown")
                confidence = cls_result.get("confidence", 0.0)
                probabilities = cls_result.get("probabilities", {})
                logger.info(f"CNN Classification: {predicted_class} (confidence: {confidence:.3f})")
                logger.info(f"Probabilities: {probabilities}")

            if preprocessing_result and preprocessing_result.get("explanations"):
                methods = list(preprocessing_result["explanations"].keys())
                logger.info(f"XAI explanations generated: {methods}")

            if preprocessing_result and preprocessing_result.get("enhanced_images"):
                styles = list(preprocessing_result["enhanced_images"].keys())
                logger.info(f"Image enhancements generated: {styles}")

            # Use enhanced image for pipeline if configured (with explicit null checks)
            if (
                config.USE_ENHANCED_IMAGE_IN_PIPELINE
                and preprocessing_result is not None
                and preprocessing_result.get("pipeline_image") is not None
            ):
                enhanced_image = preprocessing_result["pipeline_image"]
                temp_path = build_enhanced_temp_path(labeled_image.image_path.stem)
                enhanced_image.save(temp_path)
                pipeline_image_path = temp_path
                temp_files.append(temp_path)  # Track for cleanup
                logger.info(f"Using enhanced image for pipeline: {temp_path}")

        except Exception as e:
            logger.error(f"CNN + XAI preprocessing failed: {e}")
            logger.info("Continuing with original image for pipeline")

    else:
        logger.info(
            "Skipping CNN + XAI preprocessing (disabled/unavailable or suppressed in testing mode). "
            "Use --force-preprocessing to run it in testing mode."
        )

    try:
        # Mode-specific execution using extracted handlers
        if runtime_config.mode == "testing":
            try:
                _, problog_program, _, grounding_res = run_testing_mode()
            except FileNotFoundError as e:
                logger.error(f"Cached file not found: {e}")
                logger.error("Please run with --mode=full first to generate cached results, or check your file paths")
                raise
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse cached JSON file: {e}")
                raise
            except ValidationError as e:
                logger.error(f"Cached data validation failed: {e}")
                raise

        elif runtime_config.mode == "partial":
            try:
                _, problog_program, _, grounding_res = run_partial_mode(
                    pipeline_image_path=pipeline_image_path,
                    highlighted_only=runtime_config.highlighted_only,
                    preprocessing_result=preprocessing_result,
                    cnn_classification=cnn_classification,
                )
            except FileNotFoundError as e:
                logger.error(f"Cached file not found: {e}")
                logger.error("Please run with --mode=full first to generate reasoning, coding, and feature results")
                raise
            except (RuntimeError, ValidationError) as e:
                logger.error(f"Pipeline failed: {e}")
                raise

        else:  # runtime_config.mode == "full"
            try:
                _, problog_program, _, grounding_res = run_full_mode(
                    pipeline_image_path=pipeline_image_path,
                    highlighted_only=runtime_config.highlighted_only,
                    preprocessing_result=preprocessing_result,
                    cnn_classification=cnn_classification,
                )
            except NotImplementedError as e:
                logger.error(str(e))
                logger.error("Cannot continue without implementing the Gemini API")
                raise
            except (RuntimeError, ValidationError) as e:
                logger.error(f"Pipeline failed: {e}")
                raise

        # Execute final inference
        p_cat, p_dog = execute_logic_program(problog_program, grounding_res)

        # Display results using extracted function
        display_classification_results(
            image_name=labeled_image.image_path.name,
            actual_label=labeled_image.cl,
            p_cat=p_cat,
            p_dog=p_dog,
            preprocessing_result=preprocessing_result,
            cnn_classification=cnn_classification,
        )

        return p_cat, p_dog

    finally:
        # Clean up temporary files
        cleanup_temp_files(temp_files)


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="Abduction Demo: Bridging Explanations and Logics with Multimodal LLMs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["testing", "full", "partial"],
        default="testing" if config.TESTING == 1 else "full",
        help="Operation mode: 'testing' uses cached results, 'full' runs complete pipeline, 'partial' starts after coding step",
    )

    parser.add_argument(
        "--image-index",
        type=int,
        default=config.IMAGE_INDEX,
        help="Index of image to process from dataset (0-based)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.RESULTS_DIR,
        help="Directory to save results (also used by CNN/XAI preprocessing)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=config.LOG_LEVEL,
        help="Logging level",
    )

    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=config.LLM_BASE_URL,
        help="Base URL for LLM server",
    )

    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=config.LLM_API_KEY,
        help="API key for LLM server",
    )

    parser.add_argument(
        "--model-reasoning",
        type=str,
        default=config.MODEL_REASONING,
        help="Model for reasoning stage",
    )

    parser.add_argument(
        "--model-coding",
        type=str,
        default=config.MODEL_CODING,
        help="Model for coding stage",
    )

    parser.add_argument(
        "--model-feature-extraction",
        type=str,
        default=config.MODEL_FEATURE_EXTRACTION,
        help="Model for feature extraction stage",
    )

    parser.add_argument(
        "--model-grounding",
        type=str,
        default=config.MODEL_GROUNDING,
        help="Model for grounding stage",
    )

    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print configuration and exit",
    )

    parser.add_argument(
        "--highlighted-only",
        action="store_true",
        default=False,
        help="Only consider highlighted features in grounding (default: consider all features)",
    )

    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Path to a custom image file to process (overrides --image-index)",
    )

    # CNN + XAI preprocessing arguments
    parser.add_argument(
        "--disable-cnn",
        action="store_true",
        default=False,
        help="Disable CNN + XAI preprocessing (default: enabled if available)",
    )

    parser.add_argument(
        "--disable-xai",
        action="store_true",
        default=False,
        help="Disable XAI explanations (default: enabled if CNN is enabled)",
    )

    parser.add_argument(
        "--disable-enhancement",
        action="store_true",
        default=False,
        help="Disable image enhancement (default: enabled if XAI is enabled)",
    )

    parser.add_argument(
        "--use-original-image",
        action="store_true",
        default=False,
        help="Use original image in pipeline instead of enhanced image (default: use enhanced if available)",
    )

    parser.add_argument(
        "--force-preprocessing",
        action="store_true",
        default=False,
        help="Force running CNN+XAI preprocessing in testing mode (normally skipped when --mode=testing)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Show config if requested (before creating RuntimeConfig)
    if args.show_config:
        config.print_config()
        exit(0)

    # Create RuntimeConfig from args (no config mutation needed)
    runtime_config = create_runtime_config(args)

    # Update logging level based on runtime config
    logging.getLogger().setLevel(getattr(logging, runtime_config.log_level))

    # Update essential config paths that affect caching (minimal mutation)
    # Note: These are necessary for cached file paths to work correctly
    config.RESULTS_DIR = runtime_config.output_dir

    # Update LLM and Model configuration from runtime arguments
    config.LLM_BASE_URL = runtime_config.llm_base_url
    config.LLM_API_KEY = runtime_config.llm_api_key
    config.MODEL_REASONING = runtime_config.model_reasoning
    config.MODEL_CODING = runtime_config.model_coding
    config.MODEL_FEATURE_EXTRACTION = runtime_config.model_feature_extraction
    config.MODEL_GROUNDING = runtime_config.model_grounding

    # Update CNN settings for the preprocessing pipeline
    if runtime_config.disable_cnn:
        config.ENABLE_CNN_PREPROCESSING = 0
    if runtime_config.use_original_image:
        config.USE_ENHANCED_IMAGE_IN_PIPELINE = 0

    if runtime_config.force_preprocessing and runtime_config.mode == "testing":
        logger.info("Force preprocessing enabled in testing mode.")

    # Run main pipeline
    try:
        main(runtime_config)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)
