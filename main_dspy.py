import argparse
import json
import logging
from pathlib import Path

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

            TODO: Implement actual Gemini API call. For now, this will cause an error
            if coding stage is run without cached results.
            """
            logger.error("Gemini API not implemented. Please use TESTING=1 mode or implement this function.")
            raise NotImplementedError(
                "Gemini model is not implemented. "
                "Please run in TESTING mode (TESTING=1) or implement the gemini() function "
                "with a valid LLM API call (e.g., Google's Gemini API or alternative)."
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
        lm = dspy.LM(config.MODEL_FEATURE_EXTRACTION, api_base=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
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
            grounding_desc = "A json mapping feature names to the probability of the feature being present in the input image. Example: {'feature_A': 0.9, 'feature_B': 0.1}"
        else:
            grounding_desc = "A json mapping feature names to the probability of the feature being present and highlighted in the input image. Example: {'feature_A': 0.9, 'feature_B': 0.1}"

        lm = dspy.LM(config.MODEL_GROUNDING, api_base=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)
        dspy.configure(lm=lm)

        logger.debug(f"Using model: {config.MODEL_GROUNDING}")

        class Grounding(dspy.Signature):
            feature_list: list[str] = dspy.InputField(desc="List of possible features present in the image")
            image: dspy.Image = dspy.InputField(desc="Image to be analyzed for present features")
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


def main(args=None):
    """
    Main pipeline orchestration function.

    Executes the complete 5-stage pipeline:
    1. Reasoning: Generate comparative analysis (or load cached)
    2. Coding: Translate to ProbLog program (or load cached)
    3. Feature Extraction: Extract feature names (or load cached)
    4. Grounding: Detect features in image (or load cached)
    5. Logic Execution: Run ProbLog inference to classify animal

    The pipeline can run in two modes:
    - TESTING=1: Uses cached results from result-prompts/ directory
    - TESTING=0: Runs full pipeline with live LLM inference

    Args:
        args: Command-line arguments namespace. If None, will parse arguments.

    Output:
        Prints classification probabilities to console:
        - Cat Probability: 0.0-1.0
        - Dog Probability: 0.0-1.0

    Note:
        Currently processes labeled_images[1]. Modify index to classify
        different images from the dataset.
    """
    if args is None:
        args = parse_args()
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

    # Load data
    logger.info("Loading labeled images from data directory")
    labeled_images = load_data()
    logger.info(f"Loaded {len(labeled_images)} images")

    # Select image to process
    if args.image_index >= len(labeled_images):
        logger.error(f"IMAGE_INDEX {args.image_index} out of range (0-{len(labeled_images) - 1})")
        raise ValueError(f"IMAGE_INDEX must be in range 0-{len(labeled_images) - 1}")

    labeled_image = labeled_images[args.image_index]
    logger.info(
        f"Processing image {args.image_index}: {labeled_image.image_path.name} (actual label: {labeled_image.cl})"
    )

    if config.TESTING == 1:
        logger.info("Running in TESTING mode (using cached results)")
        try:
            reasoning_description = config.CACHED_REASONING_FILE.open("r").read()
            problog_program = config.CACHED_CODING_FILE.open("r").read()
            feature_list = [line.strip() for line in config.CACHED_FEATURES_FILE.open("r").readlines()]
            grounding_res = json.loads(config.CACHED_GROUNDING_FILE.open("r").read())

            # Validate loaded data
            validate_problog_program(problog_program)
            validate_feature_list(feature_list)
            validate_grounding_results(grounding_res)
            grounding_res = sanitize_grounding_results(grounding_res)

            logger.debug("Loaded and validated all cached results")
        except FileNotFoundError as e:
            logger.error(f"Cached file not found: {e}")
            logger.error("Please run with TESTING=0 first to generate cached results, or check your file paths")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse cached JSON file: {e}")
            raise
        except ValidationError as e:
            logger.error(f"Cached data validation failed: {e}")
            raise
    else:
        logger.info("Running in FULL PIPELINE mode (live LLM inference)")

        try:
            # Ensure output directory exists
            config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

            reasoning_description = reasoning()
            with open(config.CACHED_REASONING_FILE, "w") as f:
                f.write(reasoning_description)
            logger.debug(f"Saved reasoning to {config.CACHED_REASONING_FILE}")

            # ONLY FOR DEMONSTRATION PURPOSES
            problog_program = coding(reasoning_description)
            with open(config.CACHED_CODING_FILE, "w") as f:
                f.write(problog_program)
            logger.debug(f"Saved ProbLog program to {config.CACHED_CODING_FILE}")

            feature_list = extract_feature_list(problog_program=problog_program)
            with open(file=config.CACHED_FEATURES_FILE, mode="w") as f:
                f.write("\n".join(feature_list))
            logger.debug(f"Saved feature list to {config.CACHED_FEATURES_FILE}")

            grounding_res = grounding(
                feature_list=feature_list, image_path=labeled_image.image_path, highlighted_only=args.highlighted_only
            )
            with open(file=config.CACHED_GROUNDING_FILE, mode="w") as f:
                json.dump(grounding_res, f, indent=2)
            logger.debug(f"Saved grounding results to {config.CACHED_GROUNDING_FILE}")

        except NotImplementedError as e:
            logger.error(str(e))
            logger.error("Cannot continue without implementing the Gemini API")
            raise
        except (RuntimeError, ValidationError) as e:
            logger.error(f"Pipeline failed: {e}")
            raise

    # Execute final inference
    p_cat, p_dog = execute_logic_program(problog_program, grounding_res)

    # Display results
    logger.info("=" * 80)
    logger.info("CLASSIFICATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Image: {labeled_image.image_path.name}")
    logger.info(f"Actual Label: {labeled_image.cl}")
    logger.info(f"Cat Probability: {p_cat:.4f}")
    logger.info(f"Dog Probability: {p_dog:.4f}")
    predicted = "cat" if p_cat > p_dog else "dog"
    logger.info(f"Predicted: {predicted}")
    logger.info(f"Correct: {'✓' if predicted == labeled_image.cl else '✗'}")
    logger.info("=" * 80)
    logger.info("Abduction Demo Complete")
    logger.info("=" * 80)

    return p_cat, p_dog


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
        choices=["testing", "full"],
        default="testing" if config.TESTING == 1 else "full",
        help="Operation mode: 'testing' uses cached results, 'full' runs complete pipeline",
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
        help="Directory to save results",
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

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Update config based on arguments
    if args.mode == "testing":
        config.TESTING = 1
    else:
        config.TESTING = 0

    config.IMAGE_INDEX = args.image_index
    config.RESULTS_DIR = args.output_dir
    config.LOG_LEVEL = args.log_level
    config.LLM_BASE_URL = args.llm_base_url

    # Update logging level
    logging.getLogger().setLevel(getattr(logging, config.LOG_LEVEL))

    # Show config if requested
    if args.show_config:
        config.print_config()
        exit(0)

    # Run main pipeline
    try:
        main(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit(1)
