import base64
import logging
from pathlib import Path
from typing import Dict, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def apply_color_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Applies a semi-transparent colored overlay to the masked region of an image.

    Args:
        image (np.ndarray): The original BGR image.
        mask (np.ndarray): The single-channel black and white mask.
        color (tuple): The BGR color for the overlay (e.g., (0, 0, 255) for red).
        alpha (float): The transparency of the overlay (0.0 to 1.0).

    Returns:
        np.ndarray: The image with the colored overlay.

    Examples:
        overlay_result = apply_color_overlay(image, mask, color=(255, 100, 0), alpha=0.4)
        cv2.imwrite("image.jpg", overlay_result)
    """
    # Create a colored layer
    # Use mask > 0 for robustness (works with both 0-255 and 0-1 masks)
    overlay = np.zeros_like(image)
    overlay[mask > 0] = color

    # Blend the overlay with the original image
    # result = original_image * (1 - alpha) + overlay * alpha
    highlighted_image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)

    return highlighted_image


def desaturate_background(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Keeps the masked area in color and turns the background to grayscale.

    Args:
        image (np.ndarray): The original BGR image.
        mask (np.ndarray): The single-channel black and white mask.

    Returns:
        np.ndarray: The image with a desaturated background.

    Examples:
        desaturate_result = desaturate_background(image, mask)
        cv2.imwrite("image.jpg", desaturate_result)
    """
    # Create a grayscale version of the image, then convert it back to 3 channels
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    # Use the mask to select the colored foreground
    foreground = cv2.bitwise_and(image, image, mask=mask)

    # Use the inverted mask to select the grayscale background
    inverted_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(background, background, mask=inverted_mask)

    # Combine the colored foreground and grayscale background
    highlighted_image = cv2.add(foreground, background)

    return highlighted_image


def blur_background(
    image: np.ndarray,
    mask: np.ndarray,
    blur_intensity: Tuple[int, int] = (35, 35)
) -> np.ndarray:
    """
    Blurs the background, keeping the masked area in focus.

    Args:
        image (np.ndarray): The original BGR image.
        mask (np.ndarray): The single-channel black and white mask.
        blur_intensity (tuple): The kernel size for Gaussian blur. Must be odd numbers.

    Returns:
        np.ndarray: The image with a blurred background.

    Examples:
        blur_result = blur_background(image, mask, blur_intensity=(51, 51))
        cv2.imwrite("image.jpg", blur_result)

    Raises:
        ValueError: If blur_intensity values are not odd positive integers.
    """
    # Validate blur_intensity - Gaussian blur requires odd kernel sizes
    if blur_intensity[0] % 2 == 0 or blur_intensity[1] % 2 == 0:
        raise ValueError(
            f"blur_intensity must contain odd numbers, got {blur_intensity}. "
            "OpenCV GaussianBlur requires odd kernel sizes."
        )
    if blur_intensity[0] <= 0 or blur_intensity[1] <= 0:
        raise ValueError(f"blur_intensity must contain positive numbers, got {blur_intensity}")

    # Create a blurred version of the image
    blurred_image = cv2.GaussianBlur(image, blur_intensity, 0)

    # Use the mask to select the sharp foreground
    foreground = cv2.bitwise_and(image, image, mask=mask)

    # Use the inverted mask to select the blurred background
    inverted_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(blurred_image, blurred_image, mask=inverted_mask)

    # Combine the sharp foreground and blurred background
    highlighted_image = cv2.add(foreground, background)

    return highlighted_image


def apply_gradient_heatmap_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.6
) -> np.ndarray:
    """
    Applies a semi-transparent gradient heatmap overlay to the masked region.

    Args:
        image (np.ndarray): The original BGR image.
        mask (np.ndarray): The single-channel black and white mask.
        colormap (int): The OpenCV colormap to use (e.g., cv2.COLORMAP_JET).
        alpha (float): The transparency of the heatmap (0.0 to 1.0).

    Returns:
        np.ndarray: The image with the gradient heatmap overlay.

    Examples:
        gradient_heatmap_result = apply_gradient_heatmap_overlay(image, mask, colormap=cv2.COLORMAP_JET, alpha=0.6)
        cv2.imwrite("image.jpg", gradient_heatmap_result)
    """
    # Handle empty or flat masks to avoid NaN in normalization
    if mask is None or mask.sum() == 0:
        logger.warning("Empty mask provided to apply_gradient_heatmap_overlay, returning original image")
        return image.copy()

    # 1. Create a distance transform from the mask.
    # This creates a float32 image where each pixel's value is its distance
    # to the nearest zero-pixel (the edge of the mask).
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # 2. Normalize the distance transform to the 0-255 range.
    # Handle flat masks where max == min to avoid NaN
    if dist_transform.max() == dist_transform.min():
        gradient_mask = np.zeros_like(dist_transform, dtype=np.uint8)
    else:
        normalized_dist = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        gradient_mask = normalized_dist.astype(np.uint8)

    # 3. Apply the colormap to the gradient mask.
    heatmap_color = cv2.applyColorMap(gradient_mask, colormap)

    # 4. Isolate the heatmap to the region of interest using the original mask.
    heatmap_region = cv2.bitwise_and(heatmap_color, heatmap_color, mask=mask)

    # 5. Blend the isolated heatmap with the original image.
    highlighted_image = cv2.addWeighted(image, 1 - alpha, heatmap_region, alpha, 0)

    return highlighted_image


def apply_spotlight_heatmap(
    image: np.ndarray,
    mask: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    alpha: float = 0.6,
    darkness_factor: float = 0.3
) -> np.ndarray:
    """
    Applies a gradient heatmap overlay and darkens the background.

    Args:
        image (np.ndarray): The original BGR image.
        mask (np.ndarray): The single-channel black and white mask.
        colormap (int): The OpenCV colormap to use.
        alpha (float): The transparency of the heatmap.
        darkness_factor (float): How dark the background should be (0.0 to 1.0).

    Returns:
        np.ndarray: The image with the spotlight heatmap effect.

    Examples:
        spotlight_result = apply_spotlight_heatmap(image, mask, darkness_factor=0.2)
        cv2.imwrite("image.jpg", spotlight_result)
    """
    # Handle empty or flat masks to avoid NaN in normalization
    if mask is None or mask.sum() == 0:
        logger.warning("Empty mask provided to apply_spotlight_heatmap, returning original image")
        return image.copy()

    # 1. Create the gradient heatmap region (same as the previous method)
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Handle flat masks where max == min to avoid NaN
    if dist_transform.max() == dist_transform.min():
        gradient_mask = np.zeros_like(dist_transform, dtype=np.uint8)
    else:
        normalized_dist = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        gradient_mask = normalized_dist.astype(np.uint8)
    heatmap_color = cv2.applyColorMap(gradient_mask, colormap)
    heatmap_region = cv2.bitwise_and(heatmap_color, heatmap_color, mask=mask)

    # 2. Isolate the original object and blend it with the heatmap
    foreground_original = cv2.bitwise_and(image, image, mask=mask)
    highlighted_foreground = cv2.addWeighted(foreground_original, 1 - alpha, heatmap_region, alpha, 0)

    # 3. Create the darkened background
    dark_image = (image * darkness_factor).astype(np.uint8)
    inverted_mask = cv2.bitwise_not(mask)
    darkened_background = cv2.bitwise_and(dark_image, dark_image, mask=inverted_mask)

    # 4. Combine the highlighted foreground and darkened background
    final_image = cv2.add(highlighted_foreground, darkened_background)

    return final_image


def apply_composite_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    colormap: int = cv2.COLORMAP_JET,
    foreground_alpha: float = 0.6,
    background_alpha: float = 0.5
) -> np.ndarray:
    """
    Creates a composite image with different overlays for foreground and background.
    - Foreground: Original image + gradient heatmap overlay.
    - Background: Original image + solid color overlay (coldest map color).

    Args:
        image (np.ndarray): The original BGR image.
        mask (np.ndarray): The single-channel black and white mask.
        colormap (int): The OpenCV colormap to use.
        foreground_alpha (float): Transparency of the heatmap on the object.
        background_alpha (float): Transparency of the color on the background.

    Returns:
        np.ndarray: The final composite image.

    Examples:
        composite_result = apply_composite_overlay(
          image, mask, colormap=cv2.COLORMAP_JET, foreground_alpha=0.6, background_alpha=0.5
        )
        cv2.imwrite("image.jpg", composite_result)
    """
    # Handle empty or flat masks to avoid NaN in normalization
    if mask is None or mask.sum() == 0:
        logger.warning("Empty mask provided to apply_composite_overlay, returning original image")
        return image.copy()

    # === Part 1: Create the Highlighted Foreground ===

    # Generate the gradient for the object
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Handle flat masks where max == min to avoid NaN
    if dist_transform.max() == dist_transform.min():
        gradient_mask = np.zeros_like(dist_transform, dtype=np.uint8)
    else:
        normalized_dist = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX)
        gradient_mask = normalized_dist.astype(np.uint8)

    # Isolate the heatmap and original object regions
    heatmap_region = cv2.applyColorMap(gradient_mask, colormap)
    foreground_original = cv2.bitwise_and(image, image, mask=mask)

    # Blend the heatmap onto the original object
    highlighted_foreground = cv2.addWeighted(
        foreground_original, 1 - foreground_alpha, heatmap_region, foreground_alpha, 0
    )
    # Ensure only the foreground is kept after blending
    highlighted_foreground = cv2.bitwise_and(highlighted_foreground, highlighted_foreground, mask=mask)

    # === Part 2: Create the Colored Background ===

    # Programmatically get the "coldest" color from the colormap
    zero_pixel = np.zeros((1, 1), dtype=np.uint8)
    cold_color = cv2.applyColorMap(zero_pixel, colormap)[0][0].tolist()

    # Create a solid color layer and blend it with the full original image
    color_layer = np.full(image.shape, cold_color, dtype=np.uint8)
    blended_background_full = cv2.addWeighted(image, 1 - background_alpha, color_layer, background_alpha, 0)

    # Isolate only the background from this blended result
    inverted_mask = cv2.bitwise_not(mask)
    final_background = cv2.bitwise_and(blended_background_full, blended_background_full, mask=inverted_mask)

    # === Part 3: Combine Foreground and Background ===

    final_image = cv2.add(highlighted_foreground, final_background)

    return final_image


def draw_border(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3
) -> np.ndarray:
    """
    Draws a border around the masked region on the original image.

    Args:
        image (np.ndarray): The original BGR image.
        mask (np.ndarray): The single-channel black and white mask.
        color (tuple): The BGR color for the border.
        thickness (int): The thickness of the border line.

    Returns:
        np.ndarray: The image with the border drawn on it.

    Examples:
        bordered_image = draw_border(
            image,
            mask,
            color=(50, 255, 50),  # A bright green color
            thickness=8,
        )
        cv2.imwrite(".tmp-data/highlight_border.jpg", bordered_image)
    """
    # Create a copy to avoid modifying the original image
    output_image = image.copy()

    # 1. Find the contours of the shape in the mask
    # cv2.RETR_EXTERNAL finds only the outer contours of the shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 2. Draw the found contours on the output image
    # The '-1' argument draws all found contours
    cv2.drawContours(output_image, contours, -1, color, thickness)

    return output_image


def encode_base64_simple(image_path: Path) -> str:
    """
    Simple base64 encoding using direct file reading (preserves original format).

    Args:
        image_path (Path): Path to the image file.

    Returns:
        str: Base64 encoded string of the image file.
    """
    try:
        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
        return image_base64
    except FileNotFoundError:
        raise ValueError(f"Image file not found: {image_path}")
    except Exception as e:
        raise ValueError(f"Error reading image file: {e}")


def encode_base64_resized(image_path: Path, max_width: int = 800, max_height: int = 600, quality: int = 85) -> str:
    """
    Resize image and encode to base64 to reduce size for LLM context.

    Args:
        image_path (Path): Path to the image file.
        max_width (int): Maximum width for resized image.
        max_height (int): Maximum height for resized image.
        quality (int): JPEG compression quality (1-100).

    Returns:
        str: Base64 encoded string of the resized image.
    """
    try:
        # Read the image
        image = cv2.imread(str(image_path.absolute()))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Get original dimensions
        height, width = image.shape[:2]

        # Calculate new dimensions while maintaining aspect ratio
        if width > max_width or height > max_height:
            # Calculate scaling factor
            width_ratio = max_width / width
            height_ratio = max_height / height
            scale_factor = min(width_ratio, height_ratio)

            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Resize the image
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Encode as JPEG with specified quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", image, encode_param)
        if not success:
            raise ValueError(f"Failed to encode image as JPEG: {image_path}")

        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode("utf-8")
        return image_base64

    except FileNotFoundError:
        raise ValueError(f"Image file not found: {image_path}")
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")


def encode_base64_aggressive_compression(
    image_path: Path, max_width: int = 400, max_height: int = 400, quality: int = 50
) -> str:
    """
    Aggressively compress image for minimal base64 size while preserving key features.

    Args:
        image_path (Path): Path to the image file.
        max_width (int): Maximum width for resized image (smaller = less data).
        max_height (int): Maximum height for resized image (smaller = less data).
        quality (int): JPEG compression quality (1-100, lower = smaller file).

    Returns:
        str: Base64 encoded string of the heavily compressed image.
    """
    try:
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Get original dimensions
        height, width = image.shape[:2]

        # Calculate new dimensions with aggressive downsizing
        width_ratio = max_width / width
        height_ratio = max_height / height
        scale_factor = min(width_ratio, height_ratio)

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize with high-quality interpolation for small images
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Apply slight denoising to help compression
        image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

        # Encode as JPEG with aggressive compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, buffer = cv2.imencode(".jpg", image, encode_param)
        if not success:
            raise ValueError(f"Failed to encode image as JPEG: {image_path}")

        # Convert to base64
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        # Log the size for debugging
        size_kb = len(image_base64) * 3 / 4 / 1024  # Approximate KB size
        logger.info(f"Compressed image to {new_width}x{new_height}, base64 size: ~{size_kb:.1f}KB")

        return image_base64

    except FileNotFoundError:
        raise ValueError(f"Image file not found: {image_path}")
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")


def get_image_size_estimate(image_path: Path) -> Dict[str, Union[dict, str]]:
    """
    Get size estimates for different compression levels without actually encoding.

    Args:
        image_path (Path): Path to the image file.

    Returns:
        dict: Size estimates for different compression settings.
    """
    import os

    # Base64 encoding overhead factor: 4/3 (3 bytes become 4 characters)
    BASE64_OVERHEAD_FACTOR = 4 / 3

    try:
        # Get original file size
        original_size_mb = os.path.getsize(image_path) / (1024 * 1024)

        # Read image to get dimensions
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        height, width = image.shape[:2]

        # Estimate base64 sizes for different settings
        estimates = {
            "original": {
                "dimensions": f"{width}x{height}",
                "file_size_mb": original_size_mb,
                "estimated_base64_mb": original_size_mb * BASE64_OVERHEAD_FACTOR,
            },
            "standard_compression": {
                "dimensions": "800x600 (max)",
                "estimated_base64_kb": 150,  # Approximate based on typical JPEG compression
            },
            "aggressive_compression": {
                "dimensions": "400x400 (max)",
                "estimated_base64_kb": 50,  # Approximate based on aggressive JPEG compression
            },
        }

        return estimates

    except Exception as e:
        return {"error": str(e)}
