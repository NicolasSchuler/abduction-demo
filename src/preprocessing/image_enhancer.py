"""
Image Enhancement Pipeline for CNN + XAI Integration

This module creates enhanced images by combining XAI explanations with the original
images using the existing img_utils functions. It generates various visualization
styles to highlight important regions identified by XAI methods.

Features:
- Multiple enhancement styles (heatmap, spotlight, composite)
- Integration with existing img_utils functions
- Configurable overlay parameters
- Automatic image format conversion
- Batch processing capabilities
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import cv2
except Exception:
    cv2 = None
import numpy as np
from PIL import Image

from src import config
from src.img_utils import (
    apply_color_overlay,
    apply_composite_overlay,
    apply_gradient_heatmap_overlay,
    apply_spotlight_heatmap,
    blur_background,
    desaturate_background,
)

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """
    Image enhancement utility for creating visual explanations.

    This class takes XAI explanations and original images to create enhanced
    visualizations that highlight important regions for model predictions.
    """

    def __init__(
        self,
        save_enhanced: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        enhancement_styles: Optional[List[str]] = None,
    ):
        """
        Initialize the image enhancer.

        Args:
            save_enhanced: Whether to save enhanced images
            output_dir: Directory to save enhanced images
            enhancement_styles: List of enhancement styles to generate
        """
        self.save_enhanced = save_enhanced

        # Set output directory
        if output_dir is None:
            self.output_dir = Path("results/enhanced_images")
        else:
            self.output_dir = Path(output_dir)

        if self.save_enhanced:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set enhancement styles
        self.enhancement_styles = enhancement_styles or [
            "heatmap_overlay",
            "spotlight_heatmap",
            "composite_overlay",
            "color_overlay",
            "blur_background",
            "desaturate_background",
        ]

        logger.info(f"Initialized image enhancer with styles: {self.enhancement_styles}")
        logger.info(f"Output directory: {self.output_dir}")

    def enhance_image(
        self,
        original_image: Union[Image.Image, np.ndarray],
        explanations: Dict[str, Any],
        enhancement_styles: Optional[List[str]] = None,
        save_prefix: Optional[str] = None,
    ) -> Dict[str, Union[Image.Image, np.ndarray]]:
        """
        Enhance an image with XAI explanations.

        Args:
            original_image: Original PIL Image or numpy array
            explanations: Dictionary containing XAI explanations
            enhancement_styles: List of enhancement styles to apply
            save_prefix: Prefix for saved filenames

        Returns:
            Dictionary mapping style names to enhanced images
        """
        if enhancement_styles is None:
            enhancement_styles = self.enhancement_styles
        # Graceful fallback: if OpenCV is not available, skip enhancement
        if cv2 is None:
            logger.warning("OpenCV not available; skipping image enhancement.")
            return {}

        # Convert PIL Image to numpy array if needed
        if isinstance(original_image, Image.Image):
            original_array = np.array(original_image)
        else:
            original_array = original_image

        # Ensure BGR format for OpenCV functions
        if len(original_array.shape) == 3 and original_array.shape[2] == 3:
            # Assume RGB if PIL, convert to BGR for OpenCV
            if isinstance(original_image, Image.Image):
                original_bgr = cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR)
            else:
                # Assume already RGB, convert to BGR
                original_bgr = cv2.cvtColor(original_array, cv2.COLOR_RGB2BGR)
        else:
            original_bgr = original_array

        enhanced_images = {}
        image_name = save_prefix or "enhanced"

        # Create masks from explanations
        masks = self._create_masks_from_explanations(explanations)

        # Apply different enhancement styles
        for style in enhancement_styles:
            try:
                enhanced = self._apply_enhancement_style(style, original_bgr, masks, explanations)
                if enhanced is not None:
                    # Convert back to RGB for consistency
                    if len(enhanced.shape) == 3 and enhanced.shape[2] == 3:
                        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
                        enhanced_pil = Image.fromarray(enhanced_rgb)
                    else:
                        enhanced_pil = Image.fromarray(enhanced)

                    enhanced_images[style] = enhanced_pil

                    if self.save_enhanced:
                        self._save_enhanced_image(enhanced_pil, style, image_name)

            except Exception as e:
                logger.error(f"Failed to apply {style} enhancement: {e}")

        return enhanced_images

    def _create_masks_from_explanations(self, explanations: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Create refined foreground masks from XAI explanations with:
        - Configurable thresholds (config.XAI_CAM_THRESHOLD, config.XAI_ATTRIBUTION_THRESHOLD)
        - Morphological cleanup (opening + dilation)
        - Largest connected component retention
        - Optional intersection of CAM and attribution masks to suppress background
        """

        def _morph_clean(binary: np.ndarray, open_ks: int = 3, dilate_ks: int = 3) -> np.ndarray:
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
            kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_ks, dilate_ks))
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
            dilated = cv2.dilate(opened, kernel_dil, iterations=1)
            return dilated

        def _largest_component(binary: np.ndarray) -> np.ndarray:
            """Keep only the largest connected component to reduce background spill."""
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            if num_labels <= 1:  # only background
                return binary
            # Skip label 0 (background); find largest foreground
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.zeros_like(binary)
            mask[labels == largest_label] = 255
            return mask

        cam_thresh = int(getattr(config, "XAI_CAM_THRESHOLD", 0.3) * 255)
        attr_thresh = int(getattr(config, "XAI_ATTRIBUTION_THRESHOLD", 0.2) * 255)

        raw_cam_masks: List[np.ndarray] = []
        raw_attr_masks: List[np.ndarray] = []
        masks: Dict[str, np.ndarray] = {}

        for method_name, explanation in explanations.items():
            if not isinstance(explanation, dict):
                continue

            # Grad-CAM based mask
            if "grayscale_cam" in explanation:
                cam_map = explanation["grayscale_cam"]
                if cam_map.max() > 0:
                    normalized = (cam_map / cam_map.max() * 255).astype(np.uint8)
                    cam_binary = np.where(normalized >= cam_thresh, 255, 0).astype(np.uint8)
                    cam_binary = _morph_clean(cam_binary)
                    cam_binary = _largest_component(cam_binary)
                    masks[f"{method_name}_cam"] = cam_binary
                    raw_cam_masks.append(cam_binary)

            # Attribution-based mask
            attr_key = (
                "attributions"
                if "attributions" in explanation
                else ("importance_map" if "importance_map" in explanation else None)
            )
            if attr_key:
                attr_map = explanation[attr_key]
                if attr_map.max() > 0:
                    normalized_attr = (attr_map / attr_map.max() * 255).astype(np.uint8)
                    attr_binary = np.where(normalized_attr >= attr_thresh, 255, 0).astype(np.uint8)
                    attr_binary = _morph_clean(attr_binary)
                    attr_binary = _largest_component(attr_binary)
                    masks[f"{method_name}_attr"] = attr_binary
                    raw_attr_masks.append(attr_binary)

        # Combine CAM and attribution masks if both available: intersection to tighten foreground
        if raw_cam_masks and raw_attr_masks:
            combined_intersection = raw_cam_masks[0]
            for m in raw_cam_masks[1:]:
                combined_intersection = cv2.bitwise_or(combined_intersection, m)
            attr_union = raw_attr_masks[0]
            for a in raw_attr_masks[1:]:
                attr_union = cv2.bitwise_or(attr_union, a)
            intersection = cv2.bitwise_and(combined_intersection, attr_union)
            intersection = _largest_component(intersection)
            if np.sum(intersection) > 0:
                masks["fused_foreground"] = intersection
        elif raw_cam_masks:
            # Use combined cam union
            cam_union = raw_cam_masks[0]
            for m in raw_cam_masks[1:]:
                cam_union = cv2.bitwise_or(cam_union, m)
            cam_union = _largest_component(cam_union)
            masks["fused_foreground"] = cam_union
        elif raw_attr_masks:
            attr_union = raw_attr_masks[0]
            for a in raw_attr_masks[1:]:
                attr_union = cv2.bitwise_or(attr_union, a)
            attr_union = _largest_component(attr_union)
            masks["fused_foreground"] = attr_union

        # Fallback default mask if still empty
        if not masks:
            h, w = 224, 224
            default_mask = np.zeros((h, w), dtype=np.uint8)
            cy, cx = h // 2, w // 2
            radius = min(h, w) // 4
            y, x = np.ogrid[:h, :w]
            if ((x - cx) ** 2 + (y - cy) ** 2 <= radius**2).any():
                default_mask[((x - cx) ** 2 + (y - cy) ** 2 <= radius**2)] = 255
            masks["default"] = default_mask
            logger.warning("No valid XAI masks produced; using default center mask.")

        return masks

    def _apply_enhancement_style(
        self, style: str, original_bgr: np.ndarray, masks: Dict[str, np.ndarray], explanations: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Apply a specific enhancement style."""
        # Choose the best mask for this style
        mask = self._choose_best_mask(style, masks)
        if mask is None:
            logger.warning(f"No suitable mask found for {style}")
            return None

        # Resize mask to match image dimensions
        if mask.shape[:2] != original_bgr.shape[:2]:
            mask = cv2.resize(mask, (original_bgr.shape[1], original_bgr.shape[0]))

        # Apply enhancement style
        if style == "heatmap_overlay":
            return self._apply_heatmap_overlay(original_bgr, mask, explanations)
        elif style == "spotlight_heatmap":
            return self._apply_spotlight_heatmap(original_bgr, mask, explanations)
        elif style == "composite_overlay":
            return self._apply_composite_overlay(original_bgr, mask, explanations)
        elif style == "color_overlay":
            return self._apply_color_overlay(original_bgr, mask)
        elif style == "blur_background":
            return self._apply_blur_background(original_bgr, mask)
        elif style == "desaturate_background":
            return self._apply_desaturate_background(original_bgr, mask)
        else:
            logger.warning(f"Unknown enhancement style: {style}")
            return None

    def _choose_best_mask(self, style: str, masks: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Choose the most appropriate mask for a given style."""
        if not masks:
            return None

        # Priority order for different explanation types
        priority_order = [
            "grad_cam_cam",  # Prefer Grad-CAM masks
            "integrated_grad_attr",  # Then Integrated Gradients
            "shap_attr",  # Then SHAP
            "saliency_attr",  # Then Saliency
            "default",  # Finally default mask
        ]

        for mask_name in priority_order:
            for available_mask in masks:
                if mask_name in available_mask:
                    return masks[available_mask]

        # Return first available mask if no preferred mask found
        return list(masks.values())[0]

    def _apply_heatmap_overlay(
        self, original_bgr: np.ndarray, mask: np.ndarray, explanations: Dict[str, Any]
    ) -> np.ndarray:
        """Apply gradient heatmap overlay using existing img_utils function."""
        # Choose colormap and alpha based on configuration
        colormap = cv2.COLORMAP_JET
        alpha = config.XAI_OVERLAY_ALPHA

        # Use different colormaps for different classes if specified
        target_class = explanations.get("target_class", "")
        if target_class.lower() == "cat":
            colormap = getattr(config, "XAI_CAT_COLORMAP", cv2.COLORMAP_HOT)
        elif target_class.lower() == "dog":
            colormap = getattr(config, "XAI_DOG_COLORMAP", cv2.COLORMAP_COOL)

        return apply_gradient_heatmap_overlay(original_bgr, mask, colormap, alpha)

    def _apply_spotlight_heatmap(
        self, original_bgr: np.ndarray, mask: np.ndarray, explanations: Dict[str, Any]
    ) -> np.ndarray:
        """Apply spotlight heatmap effect using existing img_utils function."""
        colormap = cv2.COLORMAP_JET
        alpha = config.XAI_OVERLAY_ALPHA
        darkness_factor = config.ENHANCEMENT_DARKNESS_FACTOR

        return apply_spotlight_heatmap(original_bgr, mask, colormap, alpha, darkness_factor)

    def _apply_composite_overlay(
        self, original_bgr: np.ndarray, mask: np.ndarray, explanations: Dict[str, Any]
    ) -> np.ndarray:
        """Apply composite overlay using existing img_utils function."""
        colormap = cv2.COLORMAP_JET
        foreground_alpha = config.ENHANCEMENT_FOREGROUND_ALPHA
        background_alpha = config.ENHANCEMENT_BACKGROUND_ALPHA

        return apply_composite_overlay(original_bgr, mask, colormap, foreground_alpha, background_alpha)

    def _apply_color_overlay(self, original_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply simple color overlay using existing img_utils function."""
        # Choose color based on content (red for high importance regions)
        color = (0, 0, 255)  # Red in BGR
        alpha = config.ENHANCEMENT_FOREGROUND_ALPHA

        return apply_color_overlay(original_bgr, mask, color, alpha)

    def _apply_blur_background(self, original_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply blur background effect using existing img_utils function."""
        blur_intensity = config.ENHANCEMENT_BLUR_INTENSITY
        return blur_background(original_bgr, mask, blur_intensity)

    def _apply_desaturate_background(self, original_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply desaturate background effect using existing img_utils function."""
        return desaturate_background(original_bgr, mask)

    def _save_enhanced_image(self, image: Image.Image, style: str, prefix: str):
        """Save an enhanced image."""
        if not self.save_enhanced:
            return

        filename = f"{prefix}_{style}.png"
        filepath = self.output_dir / filename

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        image.save(filepath, "PNG", quality=95)
        logger.debug(f"Saved enhanced image: {filepath}")

    def enhance_batch(
        self,
        images: List[Union[Image.Image, np.ndarray]],
        explanations_list: List[Dict[str, Any]],
        enhancement_styles: Optional[List[str]] = None,
    ) -> List[Dict[str, Union[Image.Image, np.ndarray]]]:
        """
        Enhance multiple images with their explanations.

        Args:
            images: List of original images
            explanations_list: List of explanation dictionaries
            enhancement_styles: Enhancement styles to apply

        Returns:
            List of enhanced image dictionaries
        """
        if len(images) != len(explanations_list):
            raise ValueError("Number of images must match number of explanations")

        results = []
        for i, (image, explanations) in enumerate(zip(images, explanations_list)):
            try:
                prefix = f"batch_{i:03d}"
                enhanced = self.enhance_image(image, explanations, enhancement_styles, save_prefix=prefix)
                results.append(enhanced)
            except Exception as e:
                logger.error(f"Failed to enhance batch image {i}: {e}")
                results.append({})

        return results

    def create_comparison_grid(
        self,
        original_image: Union[Image.Image, np.ndarray],
        enhanced_images: Dict[str, Union[Image.Image, np.ndarray]],
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """
        Create a comparison grid showing original and enhanced images.

        Args:
            original_image: Original image
            enhanced_images: Dictionary of enhanced images
            grid_size: Size of the grid (rows, cols)

        Returns:
            PIL Image containing the comparison grid
        """
        # Convert all images to PIL format
        if isinstance(original_image, np.ndarray):
            if len(original_image.shape) == 3 and original_image.shape[2] == 3:
                if cv2 is not None:
                    original_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
                else:
                    original_pil = Image.fromarray(original_image)
            else:
                original_pil = Image.fromarray(original_image)
        else:
            original_pil = original_image

        # Prepare images for grid
        grid_images = {"Original": original_pil}
        grid_images.update(enhanced_images)

        # Calculate grid size
        num_images = len(grid_images)
        if grid_size is None:
            cols = min(3, num_images)
            rows = (num_images + cols - 1) // cols
        else:
            rows, cols = grid_size

        # Create grid
        img_width, img_height = original_pil.size
        grid_width = cols * img_width
        grid_height = rows * img_height

        grid = Image.new("RGB", (grid_width, grid_height), color="white")

        # Place images in grid
        for idx, (label, img) in enumerate(grid_images.items()):
            row = idx // cols
            col = idx % cols

            # Resize image to fit grid cell
            img_resized = img.resize((img_width, img_height), Image.Resampling.LANCZOS)

            # Calculate position
            x = col * img_width
            y = row * img_height

            # Paste image
            grid.paste(img_resized, (x, y))

        return grid

    def get_enhancer_info(self) -> Dict[str, Any]:
        """Get information about the enhancer configuration."""
        return {
            "enhancement_styles": self.enhancement_styles,
            "output_directory": str(self.output_dir),
            "save_enhanced": self.save_enhanced,
            "supported_styles": [
                "heatmap_overlay",
                "spotlight_heatmap",
                "composite_overlay",
                "color_overlay",
                "blur_background",
                "desaturate_background",
            ],
        }


# Convenience function for quick enhancement
def enhance_image(
    original_image: Union[Image.Image, np.ndarray], explanations: Dict[str, Any], styles: Optional[List[str]] = None
) -> Dict[str, Union[Image.Image, np.ndarray]]:
    """
    Quick image enhancement function.

    Args:
        original_image: Original image
        explanations: XAI explanations
        styles: Enhancement styles to apply

    Returns:
        Dictionary of enhanced images
    """
    enhancer = ImageEnhancer(save_enhanced=False)
    return enhancer.enhance_image(original_image, explanations, styles)


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Test Image Enhancer")
    parser.add_argument("image", type=Path, help="Path to test image")
    parser.add_argument("--output-dir", type=Path, help="Output directory for enhanced images")
    parser.add_argument(
        "--styles", nargs="+", default=["heatmap_overlay", "spotlight_heatmap"], help="Enhancement styles to apply"
    )

    args = parser.parse_args()

    # Initialize enhancer
    enhancer = ImageEnhancer(output_dir=args.output_dir, enhancement_styles=args.styles)

    # Print enhancer info
    info = enhancer.get_enhancer_info()
    print("Enhancer Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    # Test enhancement (requires dummy explanations)
    if args.image.exists():
        original_image = Image.open(args.image).convert("RGB")

        # Create dummy explanations for testing
        dummy_explanations = {
            "target_class": "cat",
            "grad_cam": {"grayscale_cam": np.random.rand(224, 224) * 0.8 + 0.1},
            "integrated_grad": {"attributions": np.random.rand(224, 224) * 0.7 + 0.15},
        }

        enhanced_images = enhancer.enhance_image(original_image, dummy_explanations, save_prefix="test")

        print(f"Enhanced image with {len(enhanced_images)} styles:")
        for style, img in enhanced_images.items():
            print(f"  {style}: {img.size}")
    else:
        print(f"Image not found: {args.image}")
