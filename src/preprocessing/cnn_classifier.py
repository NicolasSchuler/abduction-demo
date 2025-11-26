"""
EfficientNet-B0 CNN Classifier for Cat vs Dog Classification

This module provides a binary (cat vs dog) classifier built on top of a
pre-trained EfficientNet-B0 backbone from torchvision.

Two modes:
1. Fine-tuned mode:
   - If `model_path` points to a checkpoint containing a 2-class head, it loads it.

2. Aggregated initialization mode (no fine-tuned weights):
   - Loads the original 1000-class EfficientNet-B0 weights.
   - Aggregates (averages) the ImageNet classifier weights for cat-related
     synsets and dog-related synsets to synthesize a binary head without
     additional training. This yields deterministic logits based on pretrained
     semantics.

Outputs:
    {
      'predicted_class': 'cat' | 'dog',
      'probabilities': {'cat': p_cat, 'dog': p_dog},
      'confidence': max_prob,
      'xai_target_index': 0 or 1,
      'logits': (optional) tensor,
      'features': (optional) tensor embedding before final FC
    }

Grad-CAM/XAI Compatibility:
- A shim attribute `layer4` is attached (ModuleList with last feature block)
  so existing XAI code expecting `model.layer4[-1]` still works.

Usage:
    from src.preprocessing.cnn_classifier import EfficientNetCatDogClassifier
    clf = EfficientNetCatDogClassifier()
    result = clf.predict("data/images/cat.jpg")
    print(result)

CLI Test:
    python -m src.preprocessing.cnn_classifier data/images/cat.jpg --show-info
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

from src import config

logger = logging.getLogger(__name__)


class EfficientNetCatDogClassifier:
    """
    EfficientNet-B0 based binary (cat vs dog) classifier with optional
    fine-tuned weights or aggregated initialization from ImageNet.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
        pretrained: bool = True,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.device = device or self._get_default_device()
        self.pretrained = pretrained

        self.class_names = ["cat", "dog"]
        self.num_classes = 2

        self.model = self._load_model()
        # Use torchvision weight-defined transforms (224x224 resize, normalization)
        self.transform = models.EfficientNet_B0_Weights.IMAGENET1K_V1.transforms()

        logger.info(
            "Initialized EfficientNet-B0 classifier on device=%s (aggregated_from_imagenet=%s)",
            self.device,
            getattr(self, "aggregated_from_imagenet", False),
        )

    # ------------------------------------------------------------------ #
    # Initialization Helpers
    # ------------------------------------------------------------------ #
    def _get_default_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> nn.Module:
        logger.info("Loading EfficientNet-B0 backbone...")
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if self.pretrained else None
        model = models.efficientnet_b0(weights=weights)

        if not isinstance(model.classifier, nn.Sequential):
            raise RuntimeError("Unexpected EfficientNet classifier structure.")

        last_linear: nn.Linear = model.classifier[-1]
        in_features = last_linear.in_features

        if self.model_path and self.model_path.exists():
            logger.info("Attempting to load fine-tuned binary head weights: %s", self.model_path)
            # Replace head with 2-class head prior to loading
            model.classifier[-1] = nn.Linear(in_features, 2)
            self.aggregated_from_imagenet = False
            try:
                # SECURITY: Use weights_only=True to prevent arbitrary code execution
                # from malicious checkpoint files
                checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
                state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.debug("Missing keys during load: %s", missing)
                if unexpected:
                    logger.debug("Unexpected keys during load: %s", unexpected)
                logger.info("Fine-tuned weights loaded successfully.")
            except Exception as e:
                logger.warning("Failed to load fine-tuned weights (%s). Falling back to aggregation.", e)
                self._apply_aggregated_head(model, last_linear)
        else:
            logger.info("No fine-tuned model provided. Creating aggregated binary head from ImageNet weights.")
            self._apply_aggregated_head(model, last_linear)

        # Grad-CAM compatibility shim
        if not hasattr(model, "layer4"):
            try:
                model.layer4 = nn.ModuleList([model.features[-1]])  # type: ignore[attr-defined]
            except Exception as e:
                logger.debug("Could not attach layer4 shim (non-fatal): %s", e)

        model = model.to(self.device)
        model.eval()
        return model

    def _apply_aggregated_head(self, model: nn.Module, original_linear: nn.Linear):
        """
        Aggregate ImageNet class weights for cats and dogs to synthesize a binary head.

        Cat indices: [281, 282, 283, 284, 285]
        Dog indices: 151..268 (all dog breeds)

        This forms initial weight/bias vectors by averaging each group.
        """
        cat_indices = torch.tensor([281, 282, 283, 284, 285])
        dog_indices = torch.arange(151, 269)

        with torch.no_grad():
            w = original_linear.weight.data
            b = original_linear.bias.data

            cat_weight = w[cat_indices].mean(dim=0)
            cat_bias = b[cat_indices].mean()

            dog_weight = w[dog_indices].mean(dim=0)
            dog_bias = b[dog_indices].mean()

            new_head = nn.Linear(w.shape[1], 2)
            new_head.weight.data[0] = cat_weight
            new_head.bias.data[0] = cat_bias
            new_head.weight.data[1] = dog_weight
            new_head.bias.data[1] = dog_bias

        model.classifier[-1] = new_head
        self.aggregated_from_imagenet = True
        logger.info(
            "Aggregated binary head created (cats=%d indices, dogs=%d indices).",
            cat_indices.numel(),
            dog_indices.numel(),
        )

    # ------------------------------------------------------------------ #
    # Public Methods
    # ------------------------------------------------------------------ #
    def get_model_info(self) -> Dict[str, Union[str, int, bool]]:
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {
            "backbone": "EfficientNet-B0",
            "binary_head": True,
            "aggregated_from_imagenet": getattr(self, "aggregated_from_imagenet", False),
            "device": self.device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "input_size": "(3, 224, 224)",
            "num_classes": self.num_classes,
            "class_names": ", ".join(self.class_names),
            "model_path": str(self.model_path) if self.model_path else "",
            "pretrained_backbone": self.pretrained,
        }

    def _ensure_pil(self, image: Union[Image.Image, str, Path]) -> Image.Image:
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            try:
                return Image.open(image_path).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to open image {image_path}: {e}") from e
        if isinstance(image, Image.Image):
            return image
        raise TypeError(f"Unsupported image type: {type(image)}")

    def preprocess_image(self, image: Union[Image.Image, str, Path]) -> torch.Tensor:
        pil = self._ensure_pil(image)
        tensor = self.transform(pil).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(
        self,
        image: Union[Image.Image, str, Path],
        return_features: bool = False,
        return_raw_logits: bool = False,
    ) -> Dict[str, Union[str, float, Dict[str, float], torch.Tensor]]:
        input_tensor = self.preprocess_image(image)
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

        cat_prob = float(probs[0].item())
        dog_prob = float(probs[1].item())
        predicted_idx = int(torch.argmax(probs).item())
        predicted_class = self.class_names[predicted_idx]
        confidence = max(cat_prob, dog_prob)

        result: Dict[str, Union[str, float, Dict[str, float], torch.Tensor]] = {
            "predicted_class": predicted_class,
            "probabilities": {"cat": cat_prob, "dog": dog_prob},
            "confidence": confidence,
            "xai_target_index": predicted_idx,
        }

        if return_raw_logits:
            result["logits"] = logits.squeeze(0).cpu()

        if return_features:
            result["features"] = self._extract_features(input_tensor)

        return result

    def _extract_features(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the final linear layer.
        EfficientNet path: features -> avgpool -> flatten -> dropout -> linear
        We tap after avgpool flatten, before dropout+linear.
        """
        with torch.no_grad():
            x = self.model.features(input_tensor)
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
        return x.squeeze(0).cpu()

    @torch.no_grad()
    def batch_predict(
        self,
        images: List[Union[Image.Image, str, Path]],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Union[str, float, Dict[str, float]]]]:
        if not images:
            return []
        # Use provided batch_size, or config value, or default fallback of 1
        batch_size = batch_size or getattr(config, "CNN_BATCH_SIZE", 1) or 1
        results: List[Dict[str, Union[str, float, Dict[str, float]]]] = []

        for start in range(0, len(images), batch_size):
            subset = images[start : start + batch_size]
            tensors = [self.preprocess_image(img) for img in subset]
            batch_tensor = torch.cat(tensors, dim=0)
            logits = self.model(batch_tensor)
            probs_batch = torch.softmax(logits, dim=1)

            for probs in probs_batch:
                cat_prob = float(probs[0].item())
                dog_prob = float(probs[1].item())
                predicted_idx = int(torch.argmax(probs).item())
                predicted_class = self.class_names[predicted_idx]
                confidence = max(cat_prob, dog_prob)
                results.append({
                    "predicted_class": predicted_class,
                    "probabilities": {"cat": cat_prob, "dog": dog_prob},
                    "confidence": confidence,
                })

        return results


def classify_image(
    image: Union[Image.Image, str, Path],
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Union[str, float, Dict[str, float], torch.Tensor]]:
    """
    Convenience single-image classification.
    """
    clf = EfficientNetCatDogClassifier(model_path=model_path)
    return clf.predict(image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test EfficientNet-B0 Cat vs Dog Classifier")
    parser.add_argument("image", type=Path, help="Path to test image")
    parser.add_argument("--model", type=Path, help="Path to fine-tuned binary weights")
    parser.add_argument("--device", type=str, help="Device (cpu|cuda|mps)")
    parser.add_argument("--show-info", action="store_true", help="Print model info before inference")
    parser.add_argument("--features", action="store_true", help="Return feature embedding")
    parser.add_argument("--logits", action="store_true", help="Return raw logits")

    args = parser.parse_args()

    if not args.image.exists():
        print(f"Image not found: {args.image}")
        raise SystemExit(1)

    classifier = EfficientNetCatDogClassifier(model_path=args.model, device=args.device)

    if args.show_info:
        info = classifier.get_model_info()
        print("Model Info:")
        for k, v in info.items():
            print(f"  {k}: {v}")
        print()

    result = classifier.predict(
        args.image,
        return_features=args.features,
        return_raw_logits=args.logits,
    )

    print(f"Image: {args.image.name}")
    print(f"Predicted: {result['predicted_class']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Probabilities: {result['probabilities']}")
    if "features" in result:
        print(f"Features shape: {tuple(result['features'].shape)}")
    if "logits" in result:
        print(f"Logits: {result['logits'].tolist()}")

    # Suggested verification command:
    #   python -m src.preprocessing.cnn_classifier data/images/cat.jpg --show-info --features --logits
