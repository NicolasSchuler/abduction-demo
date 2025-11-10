"""
CNN + XAI Preprocessing Module for Abduction Demo

This module provides CNN-based image classification and explainable AI (XAI) methods
to enhance the abduction demo pipeline with robust visual analysis capabilities.

Features:
- EfficientNet-B0 CNN classifier for cat vs dog classification
- Multiple XAI methods (configurable): Grad-CAM, Grad-CAM++, Score-CAM, Integrated Gradients, SHAP
- Image enhancement with attribution overlays
- Integration with existing pipeline infrastructure
"""

from .cnn_classifier import EfficientNetCatDogClassifier
from .image_enhancer import ImageEnhancer
from .preprocessing_pipeline import PreprocessingPipeline
from .xai_explainer import XAIExplainer

__all__ = [
    "EfficientNetCatDogClassifier",
    "XAIExplainer",
    "ImageEnhancer",
    "PreprocessingPipeline",
]

__version__ = "1.0.0"
