"""
Validation utilities for the Abduction Demo pipeline.

Provides validation functions for data structures, probabilities,
and pipeline outputs.
"""

import logging
import re
from typing import Any

from src import config

logger = logging.getLogger(__name__)

# Threshold for logging probability clamping (values changed by more than this will be logged)
CLAMP_LOG_THRESHOLD = 0.001

# Regex patterns for ProbLog predicate detection (more robust than substring matching)
PROBLOG_PREDICATE_PATTERNS = {
    "is_cat": re.compile(r'\bis_cat\s*[(:.]'),
    "is_dog": re.compile(r'\bis_dog\s*[(:.]'),
    "prob_cat": re.compile(r'\bprob_cat\s*[(:.]'),
    "prob_dog": re.compile(r'\bprob_dog\s*[(:.]'),
}


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


def validate_probability(value: float, name: str = "probability") -> None:
    """
    Validate that a value is a valid probability (0.0 to 1.0).

    Args:
        value: The probability value to validate
        name: Name of the variable for error messages

    Raises:
        ValidationError: If value is not in [0.0, 1.0]
        TypeError: If value is not a number
    """
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")

    if not (config.MIN_PROBABILITY <= value <= config.MAX_PROBABILITY):
        raise ValidationError(
            f"{name} must be in range [{config.MIN_PROBABILITY}, {config.MAX_PROBABILITY}], got {value}"
        )


def validate_grounding_results(grounding: dict[str, float], *, strict: bool = False) -> None:
    """
    Validate grounding results from vision model.

    Checks that:
    - Result is a dictionary
    - All values are valid probabilities
    - Dictionary is not empty

    Args:
        grounding: Dictionary mapping feature names to probabilities
        strict: If True, raise on invalid probabilities; if False (default), log warnings only.
                Use strict=True when you want to ensure all values are valid before proceeding.
                Use strict=False when you plan to sanitize values with sanitize_grounding_results().

    Raises:
        ValidationError: If validation fails (always for structural issues, conditionally for value issues)
        TypeError: If types are incorrect
    """
    if not isinstance(grounding, dict):
        raise TypeError(f"Grounding results must be a dictionary, got {type(grounding).__name__}")

    if not grounding:
        raise ValidationError("Grounding results cannot be empty")

    for feature_name, probability in grounding.items():
        if not isinstance(feature_name, str):
            raise TypeError(f"Feature names must be strings, got {type(feature_name).__name__} for {feature_name}")

        try:
            validate_probability(probability, f"Probability for feature '{feature_name}'")
        except (ValidationError, TypeError) as e:
            if strict:
                raise  # Re-raise in strict mode
            logger.warning(f"Invalid probability for feature '{feature_name}': {e}")
            logger.warning("Value will be clamped when sanitize_grounding_results() is called")

    logger.info(f"Validated {len(grounding)} grounding results")


def validate_feature_list(features: list[str]) -> None:
    """
    Validate feature list extracted from ProbLog program.

    Checks that:
    - Result is a list
    - List is not empty
    - Contains only strings
    - Within size limits

    Args:
        features: List of feature names

    Raises:
        ValidationError: If validation fails
        TypeError: If types are incorrect
    """
    if not isinstance(features, list):
        raise TypeError(f"Feature list must be a list, got {type(features).__name__}")

    if not features:
        raise ValidationError("Feature list cannot be empty")

    if len(features) < config.MIN_FEATURES:
        raise ValidationError(f"Feature list must contain at least {config.MIN_FEATURES} features, got {len(features)}")

    if len(features) > config.MAX_FEATURES:
        logger.warning(f"Feature list contains {len(features)} features, exceeding recommended maximum of {config.MAX_FEATURES}")

    for i, feature in enumerate(features):
        if not isinstance(feature, str):
            raise TypeError(f"Feature at index {i} must be a string, got {type(feature).__name__}")

        if not feature.strip():
            raise ValidationError(f"Feature at index {i} is empty or whitespace")

    logger.info(f"Validated {len(features)} features")


def validate_problog_program(program: str) -> None:
    """
    Validate ProbLog program structure.

    Performs basic checks to ensure program is well-formed:
    - Not empty
    - Contains required sections (markers)
    - Contains query predicates

    Uses regex patterns for more robust predicate detection, avoiding false positives
    from comments or string literals.

    Args:
        program: ProbLog program as string

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(program, str):
        raise TypeError(f"ProbLog program must be a string, got {type(program).__name__}")

    if not program.strip():
        raise ValidationError("ProbLog program cannot be empty")

    # Check for essential components using regex patterns for robustness
    # This avoids false positives from comments or unrelated strings
    if not PROBLOG_PREDICATE_PATTERNS["is_cat"].search(program) or \
       not PROBLOG_PREDICATE_PATTERNS["is_dog"].search(program):
        logger.warning("ProbLog program may be malformed: missing 'is_cat' or 'is_dog' predicates")

    if not PROBLOG_PREDICATE_PATTERNS["prob_cat"].search(program) or \
       not PROBLOG_PREDICATE_PATTERNS["prob_dog"].search(program):
        logger.warning("ProbLog program may be malformed: missing 'prob_cat' or 'prob_dog' facts")

    # Validate syntax by attempting to parse with ProbLog
    try:
        from problog.program import PrologString
        PrologString(program)
    except ImportError:
        logger.warning("ProbLog library not available for syntax validation")
    except Exception as e:
        raise ValidationError(f"Invalid ProbLog syntax: {e}")

    logger.info(f"Validated ProbLog program ({len(program)} characters)")


def clamp_probability(value: float) -> float:
    """
    Clamp a probability value to valid range [EPSILON_PROB, 1.0 - EPSILON_PROB].

    Args:
        value: Probability value to clamp

    Returns:
        Clamped probability value

    Example:
        >>> clamp_probability(-0.1)
        0.0001
        >>> clamp_probability(1.5)
        0.9999
    """
    return max(config.EPSILON_PROB, min(1.0 - config.EPSILON_PROB, float(value)))


def sanitize_grounding_results(grounding: dict[str, Any]) -> dict[str, float]:
    """
    Sanitize grounding results by clamping probabilities to valid range.

    Args:
        grounding: Raw grounding results

    Returns:
        Sanitized grounding results with clamped probabilities

    Note:
        Invalid values (non-numeric types) default to 0.5, representing maximum uncertainty
        in a binary classification context. Use validate_grounding_results(strict=True) first
        if you need to detect and handle invalid values differently.

    Example:
        >>> sanitize_grounding_results({'feature_a': 1.2, 'feature_b': -0.1})
        {'feature_a': 0.9999, 'feature_b': 0.0001}
    """
    # Default probability for invalid values: 0.5 represents maximum uncertainty
    # (equally likely to be present or absent) in binary classification
    DEFAULT_UNCERTAIN_PROBABILITY = 0.5

    sanitized = {}

    for feature_name, value in grounding.items():
        try:
            prob = float(value)
            clamped = clamp_probability(prob)

            if abs(prob - clamped) > CLAMP_LOG_THRESHOLD:
                logger.debug(f"Clamped probability for '{feature_name}': {prob:.4f} -> {clamped:.4f}")

            sanitized[feature_name] = clamped

        except (ValueError, TypeError) as e:
            logger.warning(
                f"Invalid probability value for '{feature_name}': {value} ({e}). "
                f"Using {DEFAULT_UNCERTAIN_PROBABILITY} (maximum uncertainty)."
            )
            sanitized[feature_name] = DEFAULT_UNCERTAIN_PROBABILITY

    return sanitized
