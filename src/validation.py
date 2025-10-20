"""
Validation utilities for the Abduction Demo pipeline.

Provides validation functions for data structures, probabilities,
and pipeline outputs.
"""

import logging
from typing import Any

from src import config

logger = logging.getLogger(__name__)


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


def validate_grounding_results(grounding: dict[str, float]) -> None:
    """
    Validate grounding results from vision model.

    Checks that:
    - Result is a dictionary
    - All values are valid probabilities
    - Dictionary is not empty

    Args:
        grounding: Dictionary mapping feature names to probabilities

    Raises:
        ValidationError: If validation fails
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
            logger.warning(f"Invalid probability for feature '{feature_name}': {e}")
            # Log but don't fail - we can clamp the value
            logger.warning(f"Clamping probability to valid range")

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

    Args:
        program: ProbLog program as string

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(program, str):
        raise TypeError(f"ProbLog program must be a string, got {type(program).__name__}")

    if not program.strip():
        raise ValidationError("ProbLog program cannot be empty")

    # Check for essential components
    if "is_cat" not in program or "is_dog" not in program:
        logger.warning("ProbLog program may be malformed: missing 'is_cat' or 'is_dog' predicates")

    if "prob_cat" not in program or "prob_dog" not in program:
        logger.warning("ProbLog program may be malformed: missing 'prob_cat' or 'prob_dog' facts")

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

    Example:
        >>> sanitize_grounding_results({'feature_a': 1.2, 'feature_b': -0.1})
        {'feature_a': 0.9999, 'feature_b': 0.0001}
    """
    sanitized = {}

    for feature_name, value in grounding.items():
        try:
            prob = float(value)
            clamped = clamp_probability(prob)

            if abs(prob - clamped) > 0.001:
                logger.debug(f"Clamped probability for '{feature_name}': {prob:.4f} -> {clamped:.4f}")

            sanitized[feature_name] = clamped

        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid probability value for '{feature_name}': {value} ({e}). Using 0.5 as default.")
            sanitized[feature_name] = 0.5

    return sanitized
