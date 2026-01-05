import json
import logging
import threading
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.img_utils import blur_background

logger = logging.getLogger(__name__)

# Thread lock for safe initialization of shared class variables
_categories_lock = threading.Lock()

# Use package-relative paths instead of CWD-relative
DATA_PATH_ROOT = Path(__file__).parent.parent / "data"
DATA_PATH_IMAGES = DATA_PATH_ROOT / "images"
DATA_PATH_LABELS = DATA_PATH_ROOT / "labels"


class LabeledImage:
    """
    Represents an image with associated segmentation labels.

    This class manages loading and processing of labeled images in YOLO format,
    including polygon-based segmentation masks and category mapping.

    Class Attributes:
        categories (dict): Shared mapping of category IDs to names (e.g., {0: 'cat', 1: 'dog'})

    Instance Attributes:
        image_path (Path): Path to the image file
        image (np.ndarray): Loaded image as OpenCV BGR array
        labels (list[dict]): List of label dictionaries with keys:
            - 'id' (int): Category ID
            - 'name' (str): Category name (e.g., 'cat', 'dog')
            - 'polygon' (list[str]): Normalized polygon coordinates
        cl (str): Simplified classification label ('cat' or 'dog')

    Example:
        >>> limg = LabeledImage(
        ...     Path('data/images/cat1.jpg'),
        ...     Path('data/labels/cat1.txt'),
        ...     Path('data/notes.json')
        ... )
        >>> print(limg.cl)  # 'cat'
        >>> print(len(limg.labels))  # Number of segmented objects
    """
    categories = {}

    def __init__(self, image_path: Path, label_path: Path, label_db: Path) -> None:
        self.image_path = image_path
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}. File may not exist or be corrupted.")
        self.extract_categories(label_db)
        self.labels = self.extract_labels(label_path)
        self.cl = "cat" if "cat" in [label["name"] for label in self.labels] else "dog"

    def extract_categories(self, label_db: Path) -> None:
        """Load category mappings from JSON database file (thread-safe)."""
        # Create Label<>ID relation db (shared across instances)
        # Use double-checked locking pattern for thread safety
        if not LabeledImage.categories:
            with _categories_lock:
                # Check again inside lock to avoid race condition
                if not LabeledImage.categories:
                    try:
                        with label_db.open("r") as f:
                            data = json.load(f)
                            if "categories" not in data:
                                raise ValueError(f"Missing 'categories' key in {label_db}")
                            for label in data["categories"]:
                                if "id" not in label or "name" not in label:
                                    logger.warning(f"Skipping malformed category entry: {label}")
                                    continue
                                LabeledImage.categories[int(label["id"])] = label["name"]
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in category file {label_db}: {e}") from e
                    except FileNotFoundError:
                        raise FileNotFoundError(f"Category database not found: {label_db}")

    def extract_labels(self, label_path: Path) -> List[dict]:
        """Extract labels from YOLO-format annotation file."""
        labels = []
        try:
            with label_path.open("r") as f:
                for line_num, line in enumerate(f.readlines(), start=1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    try:
                        chunks = line.split()  # Split on any whitespace
                        if len(chunks) < 2:
                            logger.warning(f"Line {line_num} in {label_path}: insufficient data, skipping")
                            continue

                        category_id = int(chunks[0])
                        if category_id not in self.categories:
                            logger.warning(f"Line {line_num} in {label_path}: unknown category ID {category_id}, skipping")
                            continue

                        polygon_coords = chunks[1:]
                        if len(polygon_coords) % 2 != 0:
                            logger.warning(f"Line {line_num} in {label_path}: odd number of polygon coordinates, skipping")
                            continue

                        labels.append({
                            "id": category_id,
                            "name": self.categories[category_id],
                            "polygon": polygon_coords
                        })
                    except ValueError as e:
                        logger.warning(f"Line {line_num} in {label_path}: parse error ({e}), skipping")
                        continue
        except FileNotFoundError:
            raise FileNotFoundError(f"Label file not found: {label_path}")
        return labels

    def create_mask(self, elements: List[dict]) -> np.ndarray:
        """Create a binary mask from polygon elements."""
        # Handle both 3-channel (BGR) and 2-channel (grayscale) images
        if len(self.image.shape) == 3:
            img_h, img_w, _ = self.image.shape
        else:
            img_h, img_w = self.image.shape

        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for e in elements:
            try:
                norm_coords = np.array(e["polygon"], dtype=np.float32)
                if len(norm_coords) < 6:  # Need at least 3 points (6 coords) for polygon
                    logger.warning(f"Polygon has too few coordinates ({len(norm_coords)}), skipping")
                    continue
                if len(norm_coords) % 2 != 0:
                    logger.warning(f"Polygon has odd number of coordinates ({len(norm_coords)}), skipping")
                    continue
                # Validate normalized coordinates are in valid range [0, 1]
                if np.any(norm_coords < 0.0) or np.any(norm_coords > 1.0):
                    logger.warning(
                        f"Polygon has coordinates outside normalized range [0, 1], "
                        f"min={norm_coords.min():.3f}, max={norm_coords.max():.3f}, skipping"
                    )
                    continue
                points = norm_coords.reshape(-1, 2)
                points[:, 0] *= img_w
                points[:, 1] *= img_h
                pixel_points = np.array([points], dtype=np.int32)
                cv2.fillPoly(mask, pixel_points, 255)
            except (ValueError, KeyError) as e_err:
                logger.warning(f"Failed to process polygon: {e_err}")
                continue
        return mask


def load_data() -> List[LabeledImage]:
    """
    Load all labeled images from the data directory.

    Scans the data/images/ directory for JPEG files and loads each with its
    corresponding label file from data/labels/. Creates LabeledImage objects
    for each image-label pair.

    Returns:
        list[LabeledImage]: List of LabeledImage objects, one per image file.

    Expected Directory Structure:
        data/
        ├── images/
        │   ├── img1.jpg
        │   └── img2.jpg
        ├── labels/
        │   ├── img1.txt  (YOLO format)
        │   └── img2.txt
        └── notes.json  (category mappings)

    Example:
        >>> images = load_data()
        >>> print(f"Loaded {len(images)} images")
        >>> print(images[0].cl)  # 'cat' or 'dog'

    Raises:
        FileNotFoundError: If data directories don't exist
        ValueError: If image/label pairs are mismatched
    """
    limgs = []
    # Support case-insensitive image extensions using single directory scan
    # More efficient than multiple glob() calls which each scan the directory
    valid_extensions = {".jpg", ".jpeg", ".png"}
    for img in DATA_PATH_IMAGES.iterdir():
        if not img.is_file():
            continue
        if img.suffix.lower() not in valid_extensions:
            continue
        label_path = DATA_PATH_LABELS / (img.stem + ".txt")
        if not label_path.exists():
            logger.warning(f"No label file found for {img.name}, skipping")
            continue
        try:
            limgs.append(LabeledImage(img, label_path, DATA_PATH_ROOT / "notes.json"))
        except (ValueError, FileNotFoundError) as e:
            logger.warning(f"Failed to load {img.name}: {e}")
            continue
    return limgs


def get_feature_list(labeled_image: LabeledImage) -> List[str]:
    """Get a list of available features in the given image."""
    return [entry["name"] for entry in labeled_image.labels]


def emphasize_feature(labeled_image: LabeledImage, feature: str) -> Optional[np.ndarray]:
    """Get the highlighted section with regard to the `feature` for the given image.
    Used for further inspection of the given feature.

    Args:
        labeled_image: The labeled image to process
        feature: name of the feature to inspect (e.g., 'cat', 'dog')

    Returns:
        Image with blurred background highlighting the feature, or None if feature not found
    """
    category_names = list(labeled_image.categories.values())
    if feature in category_names:
        # Get all labels matching the requested feature name
        matching_labels = [label for label in labeled_image.labels if label["name"] == feature]
        if not matching_labels:
            return None
        mask = labeled_image.create_mask(matching_labels)
        # return emphasized feature in image
        return blur_background(labeled_image.image, mask, blur_intensity=(51, 51))
    else:
        return None
