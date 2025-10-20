"""
Test suite for the Abduction Demo pipeline.

Tests cover data loading, validation, and key pipeline components.
"""

import json
from pathlib import Path

import pytest

from src import config
from src.data import LabeledImage, load_data
from src.validation import (
    ValidationError,
    clamp_probability,
    sanitize_grounding_results,
    validate_feature_list,
    validate_grounding_results,
    validate_problog_program,
)


class TestDataLoading:
    """Tests for data loading functionality."""

    def test_load_data_returns_list(self):
        """Test that load_data returns a list."""
        images = load_data()
        assert isinstance(images, list), "load_data should return a list"

    def test_load_data_non_empty(self):
        """Test that load_data returns at least one image."""
        images = load_data()
        assert len(images) > 0, "Data directory should contain at least one image"

    def test_labeled_image_has_classification(self):
        """Test that each LabeledImage has a classification."""
        images = load_data()
        for img in images:
            assert isinstance(img, LabeledImage)
            assert hasattr(img, "cl")
            assert img.cl in ["cat", "dog"], f"Classification should be 'cat' or 'dog', got {img.cl}"

    def test_labeled_image_has_path(self):
        """Test that each LabeledImage has a valid path."""
        images = load_data()
        for img in images:
            assert hasattr(img, "image_path")
            assert img.image_path.exists(), f"Image file should exist: {img.image_path}"


class TestValidation:
    """Tests for validation utilities."""

    def test_validate_grounding_valid(self):
        """Test validation passes for valid grounding results."""
        valid_grounding = {"feature_a": 0.5, "feature_b": 0.9, "feature_c": 0.1}
        # Should not raise
        validate_grounding_results(valid_grounding)

    def test_validate_grounding_empty_raises(self):
        """Test validation fails for empty grounding results."""
        with pytest.raises(ValidationError):
            validate_grounding_results({})

    def test_validate_grounding_non_dict_raises(self):
        """Test validation fails for non-dictionary grounding."""
        with pytest.raises(TypeError):
            validate_grounding_results([0.5, 0.9])  # type: ignore

    def test_validate_feature_list_valid(self):
        """Test validation passes for valid feature list."""
        valid_features = ["feature_a", "feature_b", "feature_c"]
        # Should not raise
        validate_feature_list(valid_features)

    def test_validate_feature_list_empty_raises(self):
        """Test validation fails for empty feature list."""
        with pytest.raises(ValidationError):
            validate_feature_list([])

    def test_validate_feature_list_non_list_raises(self):
        """Test validation fails for non-list features."""
        with pytest.raises(TypeError):
            validate_feature_list("feature_a")  # type: ignore

    def test_validate_problog_program_valid(self):
        """Test validation passes for valid ProbLog program."""
        valid_program = """
        0.5::is_cat; 0.5::is_dog.
        P::feature(F) :- is_cat, prob_cat(F, P).
        prob_cat(small_muzzle, 0.9). prob_dog(small_muzzle, 0.1).
        """
        # Should not raise
        validate_problog_program(valid_program)

    def test_validate_problog_program_empty_raises(self):
        """Test validation fails for empty program."""
        with pytest.raises(ValidationError):
            validate_problog_program("")


class TestProbabilityClamping:
    """Tests for probability clamping and sanitization."""

    def test_clamp_probability_in_range(self):
        """Test that valid probabilities are unchanged."""
        assert clamp_probability(0.5) == 0.5
        assert clamp_probability(0.1) == 0.1
        assert clamp_probability(0.9) == 0.9

    def test_clamp_probability_below_epsilon(self):
        """Test that probabilities below EPSILON are clamped."""
        result = clamp_probability(0.0)
        assert result == config.EPSILON_PROB

        result = clamp_probability(-0.5)
        assert result == config.EPSILON_PROB

    def test_clamp_probability_above_one(self):
        """Test that probabilities above 1.0 are clamped."""
        result = clamp_probability(1.0)
        assert result == 1.0 - config.EPSILON_PROB

        result = clamp_probability(1.5)
        assert result == 1.0 - config.EPSILON_PROB

    def test_sanitize_grounding_results(self):
        """Test sanitization of grounding results."""
        raw_grounding = {
            "feature_a": 0.5,  # Valid
            "feature_b": 1.5,  # Above 1.0
            "feature_c": -0.1,  # Below 0.0
            "feature_d": "invalid",  # Invalid type - should default to 0.5
        }

        sanitized = sanitize_grounding_results(raw_grounding)

        assert sanitized["feature_a"] == 0.5
        assert sanitized["feature_b"] == 1.0 - config.EPSILON_PROB
        assert sanitized["feature_c"] == config.EPSILON_PROB
        assert sanitized["feature_d"] == 0.5  # Default for invalid


class TestConfiguration:
    """Tests for configuration validation."""

    def test_config_has_required_attributes(self):
        """Test that config has all required attributes."""
        required_attrs = [
            "TESTING",
            "IMAGE_INDEX",
            "LLM_BASE_URL",
            "EPSILON_PROB",
            "DATA_DIR",
            "RESULTS_DIR",
        ]

        for attr in required_attrs:
            assert hasattr(config, attr), f"Config should have attribute '{attr}'"

    def test_epsilon_prob_in_valid_range(self):
        """Test that EPSILON_PROB is in valid range."""
        assert 0 < config.EPSILON_PROB < 0.5, "EPSILON_PROB should be in range (0, 0.5)"

    def test_data_directories_exist(self):
        """Test that required data directories exist."""
        assert config.DATA_DIR.exists(), f"Data directory should exist: {config.DATA_DIR}"
        assert config.DATA_IMAGES_DIR.exists(), f"Images directory should exist: {config.DATA_IMAGES_DIR}"
        assert config.DATA_LABELS_DIR.exists(), f"Labels directory should exist: {config.DATA_LABELS_DIR}"

    def test_validate_config_testing_mode(self):
        """Test configuration validation in testing mode."""
        # This should pass if cached files exist
        if config.TESTING == 1:
            # Should not raise if cached files exist
            config.validate_config()


class TestCachedResults:
    """Tests for cached result files (when in testing mode)."""

    @pytest.mark.skipif(config.TESTING != 1, reason="Only applicable in testing mode")
    def test_cached_files_exist(self):
        """Test that all cached result files exist in testing mode."""
        assert config.CACHED_REASONING_FILE.exists(), "Cached reasoning file should exist"
        assert config.CACHED_CODING_FILE.exists(), "Cached coding file should exist"
        assert config.CACHED_FEATURES_FILE.exists(), "Cached features file should exist"
        assert config.CACHED_GROUNDING_FILE.exists(), "Cached grounding file should exist"

    @pytest.mark.skipif(config.TESTING != 1, reason="Only applicable in testing mode")
    def test_cached_grounding_valid_json(self):
        """Test that cached grounding file contains valid JSON."""
        with open(config.CACHED_GROUNDING_FILE, "r") as f:
            data = json.load(f)
            assert isinstance(data, dict), "Grounding file should contain a dictionary"

            # Validate structure
            validate_grounding_results(data)

    @pytest.mark.skipif(config.TESTING != 1, reason="Only applicable in testing mode")
    def test_cached_features_non_empty(self):
        """Test that cached features file is non-empty."""
        with open(config.CACHED_FEATURES_FILE, "r") as f:
            features = [line.strip() for line in f.readlines()]
            assert len(features) > 0, "Features file should contain at least one feature"

            # Validate structure
            validate_feature_list(features)

    @pytest.mark.skipif(config.TESTING != 1, reason="Only applicable in testing mode")
    def test_cached_problog_valid(self):
        """Test that cached ProbLog program is valid."""
        with open(config.CACHED_CODING_FILE, "r") as f:
            program = f.read()
            # Should not raise
            validate_problog_program(program)


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
