"""
CNN + XAI Preprocessing Pipeline

This module orchestrates the complete CNN + XAI preprocessing pipeline, combining
classification, explanation generation, and image enhancement into a unified
workflow that can be easily integrated into the existing abduction demo pipeline.

Features:
- Complete CNN + XAI preprocessing workflow
- Automatic method selection and configuration
- Metadata generation and saving
- Integration-ready output format
- Configurable processing options
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from src import config

from .cnn_classifier import EfficientNetCatDogClassifier
from .image_enhancer import ImageEnhancer
from .xai_explainer import XAIExplainer

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Complete CNN + XAI preprocessing pipeline.

    This class orchestrates the entire preprocessing workflow:
    1. CNN classification of the input image
    2. XAI explanation generation
    3. Image enhancement with attribution overlays
    4. Metadata collection and saving
    """

    def __init__(
        self,
        enable_cnn: Optional[bool] = None,
        enable_xai: Optional[bool] = None,
        enable_enhancement: Optional[bool] = None,
        save_outputs: Optional[bool] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the preprocessing pipeline.

        Args:
            enable_cnn: Whether to enable CNN classification
            enable_xai: Whether to enable XAI explanations
            enable_enhancement: Whether to enable image enhancement
            save_outputs: Whether to save intermediate outputs
            output_dir: Directory to save outputs
        """
        # Configuration
        self.enable_cnn = enable_cnn if enable_cnn is not None else bool(config.ENABLE_CNN_PREPROCESSING)
        self.enable_xai = enable_xai if enable_xai is not None else self.enable_cnn  # XAI requires CNN
        self.enable_enhancement = (
            enable_enhancement if enable_enhancement is not None else self.enable_xai
        )  # Enhancement requires XAI
        self.save_outputs = save_outputs if save_outputs is not None else config.ENABLE_CNN_PREPROCESSING

        # Set output directory
        if output_dir is None:
            self.output_dir = config.RESULTS_DIR / "cnn_preprocessing"
        else:
            self.output_dir = Path(output_dir)

        if self.save_outputs:
            # Ensure root output directory (still used for cached pipeline items)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            # Use central config-managed directories (avoid per-pipeline subfolder divergence)
            config.ENHANCEMENT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            config.XAI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            config.CNN_METADATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._initialize_components()

        logger.info("Initialized preprocessing pipeline:")
        logger.info(f"  CNN: {'ENABLED' if self.enable_cnn else 'DISABLED'}")
        logger.info(f"  XAI: {'ENABLED' if self.enable_xai else 'DISABLED'}")
        logger.info(f"  Enhancement: {'ENABLED' if self.enable_enhancement else 'DISABLED'}")
        logger.info(f"  Save outputs: {'YES' if self.save_outputs else 'NO'}")
        logger.info(f"  Output directory: {self.output_dir}")

    def _initialize_components(self):
        """Initialize pipeline components based on configuration."""
        self.cnn_classifier = None
        self.xai_explainer = None
        self.image_enhancer = None

        if self.enable_cnn:
            try:
                # Initialize CNN classifier
                device = None if config.CNN_DEVICE == "auto" else config.CNN_DEVICE
                model_path = config.CNN_MODEL_PATH if config.CNN_MODEL_PATH.strip() else None

                self.cnn_classifier = EfficientNetCatDogClassifier(model_path=model_path, device=device)

                logger.info("CNN classifier initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize CNN classifier: {e}")
                self.enable_cnn = False
                self.enable_xai = False
                self.enable_enhancement = False

        if self.enable_xai and self.cnn_classifier:
            try:
                # Initialize XAI explainer (use centralized config directory)
                xai_methods = [method.strip() for method in config.XAI_METHODS if method.strip()]
                xai_output_dir = config.XAI_OUTPUT_DIR if self.save_outputs else None

                self.xai_explainer = XAIExplainer(
                    classifier=self.cnn_classifier,
                    methods=xai_methods,
                    save_explanations=config.XAI_SAVE_EXPLANATIONS,
                    output_dir=xai_output_dir,
                )

                logger.info(f"XAI explainer initialized with methods: {xai_methods}")

            except Exception as e:
                logger.error(f"Failed to initialize XAI explainer: {e}")
                self.enable_xai = False
                self.enable_enhancement = False

        if self.enable_enhancement:
            try:
                # Initialize image enhancer (use centralized config directory)
                enhancement_styles = [style.strip() for style in config.ENHANCEMENT_STYLES if style.strip()]
                enhancement_output_dir = config.ENHANCEMENT_OUTPUT_DIR if self.save_outputs else None

                self.image_enhancer = ImageEnhancer(
                    save_enhanced=config.ENHANCEMENT_SAVE_IMAGES,
                    output_dir=enhancement_output_dir,
                    enhancement_styles=enhancement_styles,
                )

                logger.info(f"Image enhancer initialized with styles: {enhancement_styles}")

            except Exception as e:
                logger.error(f"Failed to initialize image enhancer: {e}")
                self.enable_enhancement = False

    def process_image(self, image: Union[Image.Image, str, Path], save_prefix: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline.

        Args:
            image: PIL Image, or path to image file
            save_prefix: Prefix for saved files

        Returns:
            Dictionary containing all processing results
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            original_image = Image.open(image_path).convert("RGB")
            image_name = image_path.stem
        else:
            original_image = image
            image_name = save_prefix or "processed_image"

        logger.info(f"Processing image: {image_name}")

        # Initialize result dictionary
        result = {
            "image_name": image_name,
            "original_image": original_image,
            "processing_enabled": {
                "cnn": self.enable_cnn,
                "xai": self.enable_xai,
                "enhancement": self.enable_enhancement,
            },
        }

        # Step 1: CNN Classification
        if self.enable_cnn and self.cnn_classifier:
            try:
                logger.info("Step 1: CNN Classification...")
                classification_result = self.cnn_classifier.predict(
                    original_image, return_features=True, return_raw_logits=True
                )

                result["classification"] = classification_result
                predicted_class = classification_result["predicted_class"]
                confidence = classification_result["confidence"]

                logger.info(f"  Predicted: {predicted_class} (confidence: {confidence:.3f})")

                # Check confidence threshold
                if confidence < config.CNN_CONFIDENCE_THRESHOLD:
                    logger.warning(f"Low confidence prediction: {confidence:.3f} < {config.CNN_CONFIDENCE_THRESHOLD}")

            except Exception as e:
                logger.error(f"CNN classification failed: {e}")
                result["classification"] = None
                predicted_class = None
        else:
            logger.info("CNN classification disabled")
            result["classification"] = None
            predicted_class = None

        # Step 2: XAI Explanations
        if self.enable_xai and self.xai_explainer and predicted_class:
            try:
                logger.info("Step 2: XAI Explanations...")
                explanations = self.xai_explainer.explain_image(
                    original_image,
                    target_class=predicted_class,
                    target_index=classification_result.get("xai_target_index"),
                )

                result["explanations"] = explanations
                logger.info(f"  Generated explanations: {list(explanations.keys())}")

            except Exception as e:
                logger.error(f"XAI explanation failed: {e}")
                result["explanations"] = None
        else:
            logger.info("XAI explanations disabled")
            result["explanations"] = None

        # Step 3: Image Enhancement
        if self.enable_enhancement and self.image_enhancer and result["explanations"]:
            try:
                logger.info("Step 3: Image Enhancement...")
                enhanced_images = self.image_enhancer.enhance_image(
                    original_image, result["explanations"], save_prefix=image_name
                )

                result["enhanced_images"] = enhanced_images
                logger.info(f"  Generated enhancements: {list(enhanced_images.keys())}")

                # Select primary enhanced image for pipeline use
                if config.USE_ENHANCED_IMAGE_IN_PIPELINE and enhanced_images:
                    primary_style = "heatmap_overlay"  # Default style
                    if primary_style in enhanced_images:
                        result["pipeline_image"] = enhanced_images[primary_style]
                        logger.info(f"  Selected enhanced image for pipeline: {primary_style}")
                    else:
                        # Use first available enhancement
                        first_style = list(enhanced_images.keys())[0]
                        result["pipeline_image"] = enhanced_images[first_style]
                        logger.info(f"  Selected enhanced image for pipeline: {first_style} (fallback)")
                else:
                    result["pipeline_image"] = original_image
                    logger.info("  Using original image for pipeline")

            except Exception as e:
                logger.error(f"Image enhancement failed: {e}")
                result["enhanced_images"] = None
                result["pipeline_image"] = original_image
        else:
            logger.info("Image enhancement disabled")
            result["enhanced_images"] = None
            result["pipeline_image"] = original_image

        # Step 4: Save metadata
        if self.save_outputs:
            try:
                self._save_metadata(result, image_name)
                logger.info("Metadata saved successfully")
            except Exception as e:
                logger.error(f"Failed to save metadata: {e}")

        logger.info(f"Processing completed for {image_name}")
        return result

    def _save_metadata(self, result: Dict[str, Any], image_name: str):
        """Save processing metadata to JSON file."""
        if not config.SAVE_CNN_METADATA:
            return

        metadata = {
            "image_name": image_name,
            "processing_config": {
                "cnn_enabled": self.enable_cnn,
                "xai_enabled": self.enable_xai,
                "enhancement_enabled": self.enable_enhancement,
                "use_enhanced_image": config.USE_ENHANCED_IMAGE_IN_PIPELINE,
            },
            "classification": None,
            "explanations_summary": None,
            "enhancement_summary": None,
        }

        # Add classification metadata
        if result["classification"]:
            cls_result = result["classification"]
            metadata["classification"] = {
                "predicted_class": cls_result["predicted_class"],
                "confidence": cls_result["confidence"],
                "probabilities": cls_result["probabilities"],
                "xai_target_index": cls_result.get("xai_target_index"),
            }

            if "features" in cls_result:
                metadata["classification"]["features_shape"] = list(cls_result["features"].shape)

        # Add explanations summary
        if result["explanations"]:
            explanations = result["explanations"]
            metadata["explanations_summary"] = {"target_class": explanations.get("target_class"), "methods": []}

            for method_name, explanation in explanations.items():
                if method_name in ["image_name", "target_class", "target_index", "original_image", "input_tensor"]:
                    continue

                if explanation:
                    method_summary = {
                        "method": method_name,
                        "name": explanation.get("method", method_name),
                        "description": explanation.get("description", ""),
                    }

                    # Add statistics if available
                    if "grayscale_cam" in explanation:
                        cam_map = explanation["grayscale_cam"]
                        method_summary["statistics"] = {
                            "mean_activation": float(np.mean(cam_map)),
                            "max_activation": float(np.max(cam_map)),
                            "shape": list(cam_map.shape),
                        }
                    elif "attributions" in explanation or "importance_map" in explanation:
                        attr_key = "attributions" if "attributions" in explanation else "importance_map"
                        attr_map = explanation[attr_key]
                        method_summary["statistics"] = {
                            "mean_importance": float(np.mean(attr_map)),
                            "max_importance": float(np.max(attr_map)),
                            "shape": list(attr_map.shape),
                        }

                    metadata["explanations_summary"]["methods"].append(method_summary)

        # Add enhancement summary
        if result["enhanced_images"]:
            enhanced = result["enhanced_images"]
            metadata["enhancement_summary"] = {"generated_styles": list(enhanced.keys()), "pipeline_image_style": None}

            # Identify which enhanced image is used for pipeline
            for style_name, enhanced_image in enhanced.items():
                if enhanced_image is result.get("pipeline_image"):
                    metadata["enhancement_summary"]["pipeline_image_style"] = style_name
                    break

        # Save metadata
        metadata_path = config.CNN_METADATA_OUTPUT_DIR / f"{image_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Metadata saved to {metadata_path}")

    def process_batch(
        self, images: List[Union[Image.Image, str, Path]], save_prefixes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images through the pipeline.

        Args:
            images: List of PIL Images or image paths
            save_prefixes: List of prefixes for saved files

        Returns:
            List of processing results
        """
        if save_prefixes is None:
            save_prefixes = [f"batch_{i:03d}" for i in range(len(images))]
        elif len(save_prefixes) != len(images):
            raise ValueError("Number of save prefixes must match number of images")

        logger.info(f"Processing batch of {len(images)} images")
        results = []

        for i, (image, prefix) in enumerate(zip(images, save_prefixes)):
            try:
                logger.info(f"Processing batch image {i + 1}/{len(images)}: {prefix}")
                result = self.process_image(image, save_prefix=prefix)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process batch image {i}: {e}")
                results.append(None)

        successful_count = sum(1 for r in results if r is not None)
        logger.info(f"Batch processing completed: {successful_count}/{len(images)} successful")

        return results

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline configuration."""
        info = {
            "enabled_components": {
                "cnn": self.enable_cnn,
                "xai": self.enable_xai,
                "enhancement": self.enable_enhancement,
            },
            "output_directory": str(self.output_dir),
            "save_outputs": self.save_outputs,
        }

        if self.cnn_classifier:
            info["cnn_info"] = self.cnn_classifier.get_model_info()

        if self.xai_explainer:
            info["xai_info"] = self.xai_explainer.get_explainer_info()

        if self.image_enhancer:
            info["enhancement_info"] = self.image_enhancer.get_enhancer_info()

        return info


# Convenience function for quick preprocessing
def preprocess_image(
    image: Union[Image.Image, str, Path],
    enable_cnn: bool = True,
    enable_xai: bool = True,
    enable_enhancement: bool = True,
) -> Dict[str, Any]:
    """
    Quick preprocessing function for single images.

    Args:
        image: PIL Image, or path to image file
        enable_cnn: Whether to enable CNN classification
        enable_xai: Whether to enable XAI explanations
        enable_enhancement: Whether to enable image enhancement

    Returns:
        Processing result dictionary
    """
    pipeline = PreprocessingPipeline(
        enable_cnn=enable_cnn, enable_xai=enable_xai, enable_enhancement=enable_enhancement, save_outputs=False
    )
    return pipeline.process_image(image)


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Test CNN + XAI Preprocessing Pipeline")
    parser.add_argument("image", type=Path, help="Path to test image")
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")
    parser.add_argument("--disable-cnn", action="store_true", help="Disable CNN processing")
    parser.add_argument("--disable-xai", action="store_true", help="Disable XAI processing")
    parser.add_argument("--disable-enhancement", action="store_true", help="Disable image enhancement")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = PreprocessingPipeline(
        enable_cnn=not args.disable_cnn,
        enable_xai=not args.disable_xai,
        enable_enhancement=not args.disable_enhancement,
        output_dir=args.output_dir,
    )

    # Print pipeline info
    info = pipeline.get_pipeline_info()
    print("Pipeline Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    # Test preprocessing
    if args.image.exists():
        result = pipeline.process_image(args.image)

        print(f"Processed image: {result['image_name']}")
        print(f"Classification: {result['classification']['predicted_class'] if result['classification'] else 'None'}")
        print(f"Explanations: {list(result['explanations'].keys()) if result['explanations'] else []}")
        print(f"Enhancements: {list(result['enhanced_images'].keys()) if result['enhanced_images'] else []}")
        print(f"Pipeline image shape: {result['pipeline_image'].size}")
    else:
        print(f"Image not found: {args.image}")
