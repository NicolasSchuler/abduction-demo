import json
from pathlib import Path
from typing import List

import cv2
import numpy as np

from src.img_utils import blur_background

DATA_PATH_ROOT = Path("data").absolute()
DATA_PATH_IMAGES = DATA_PATH_ROOT / "images"
DATA_PATH_LABELS = DATA_PATH_ROOT / "labels"


class LabeledImage:
    categories = {}

    def __init__(self, image_path: Path, label_path: Path, label_db: Path) -> None:
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.extract_categories(label_db)
        self.labels = self.extract_labels(label_path)
        self.cl = "cat" if "cat" in [label["name"] for label in self.labels] else "dog"

    def extract_categories(self, label_db: Path):
        # Create Label<>ID relation db
        if LabeledImage.categories == {}:
            with label_db.open("r") as f:
                labels = json.load(f)
                for label in labels["categories"]:
                    LabeledImage.categories[int(label["id"])] = label["name"]

    def extract_labels(self, label_path: Path):
        labels = []
        with label_path.open("r") as f:
            for line in f.readlines():
                label = {}
                chunks = line.split(" ")
                label["id"] = int(chunks[0])
                label["name"] = self.categories[label["id"]]
                label["polygon"] = chunks[1:]
                labels.append(label)
        return labels

    def create_mask(self, elements: List[dict]):
        img_h, img_w, _ = self.image.shape
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        for e in elements:
            norm_coords = np.array(e["polygon"], dtype=np.float32)
            points = norm_coords.reshape(-1, 2)
            points[:, 0] *= img_w
            points[:, 1] *= img_h
            pixel_points = np.array([points], dtype=np.int32)
            cv2.fillPoly(mask, pixel_points, 255)
        return mask


def load_data():
    limgs = []
    for img in DATA_PATH_IMAGES.glob("*.jpg"):
        limgs.append(LabeledImage(img, DATA_PATH_LABELS / (str(img.stem) + ".txt"), DATA_PATH_ROOT / "notes.json"))
    return limgs


def get_feature_list(labeled_image: LabeledImage):
    """Get a list of available features in the given image."""
    return [entry["name"] for entry in labeled_image.labels]


def emphasize_feature(labeled_image: LabeledImage, feature: str):
    """Get the highlighted section with regard to the `feature` for the given image.
    Used for further inspection of the given feature.

    Args:
        feature: name of the feature to inspect
    """

    categories = labeled_image.categories.values()
    if feature in categories:
        # get ids from feature
        tmp = [(limg["id"] if limg["name"] == feature else None) for limg in labeled_image.labels]
        ids = filter(None, tmp)
        mask = labeled_image.create_mask([labeled_image.labels[x] for x in ids])
        # return emphasized feature in image
        return blur_background(labeled_image.image, mask, blur_intensity=(51, 51))
    else:
        return None
