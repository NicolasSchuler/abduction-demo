"""
XAI (Explainable AI) Methods for CNN Image Classification

This module implements multiple explainable AI methods to provide visual and numerical
explanations for CNN predictions. It supports Grad-CAM++, Integrated Gradients,
and SHAP to help users understand what features the model uses for classification.

Features:
- Grad-CAM++ for class activation visualization
- Integrated Gradients for pixel-level attribution
- SHAP DeepExplainer for comprehensive feature analysis
- Unified interface for multiple XAI methods
- Compatible with ResNet-50 and other CNN architectures
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# Optional / lazy dependencies
try:
    from captum.attr import IntegratedGradients, Occlusion, Saliency
except ImportError:  # captum missing
    IntegratedGradients = None
    Saliency = None
    Occlusion = None

import os

# Optional OpenCV import for resizing CAM to original image size
try:
    import cv2
except ImportError:
    cv2 = None

try:
    from pytorch_grad_cam import (  # use_cuda omitted later for broader version compatibility
        GradCAM,
        GradCAMPlusPlus,
        GuidedBackpropReLUModel,
        LayerCAM,
        ScoreCAM,
    )
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:  # grad-cam missing
    GradCAM = None
    GradCAMPlusPlus = None
    LayerCAM = None
    ScoreCAM = None
    GuidedBackpropReLUModel = None
    ClassifierOutputTarget = None
    show_cam_on_image = None

try:
    import shap
except ImportError:  # shap missing
    shap = None


from .cnn_classifier import EfficientNetCatDogClassifier

logger = logging.getLogger(__name__)


class XAIExplainer:
    """
    Unified XAI explainer supporting multiple explanation methods.

    This class provides a comprehensive interface for generating explanations
    from CNN models using various XAI techniques.
    """

    def __init__(
        self,
        classifier: EfficientNetCatDogClassifier,
        methods: Optional[List[str]] = None,
        save_explanations: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """
        Initialize the XAI explainer.

        Args:
            classifier: Trained CNN classifier instance
            methods: List of XAI methods to use ['grad_cam', 'integrated_grad', 'shap', 'saliency']
            save_explanations: Whether to save explanation visualizations
            output_dir: Directory to save explanation outputs
        """
        self.classifier = classifier
        self.model = classifier.model
        self.device = classifier.device
        self.class_names = classifier.class_names
        self.save_explanations = save_explanations

        # Set output directory
        if output_dir is None:
            self.output_dir = Path("results/xai_explanations")
        else:
            self.output_dir = Path(output_dir)

        if self.save_explanations:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure XAI methods
        self.methods = methods or ["grad_cam", "integrated_grad", "shap"]
        self._setup_explainers()

        logger.info(f"Initialized XAI explainer with methods: {self.methods}")
        logger.info(f"Output directory: {self.output_dir}")

    def _setup_explainers(self):
        """Setup the specific XAI explainers with Tier 1 Grad-CAM configurability."""
        self.explainers = {}

        # ---- Grad-CAM configuration (Tier 1) ----
        if "grad_cam" in self.methods:
            if (
                (GradCAMPlusPlus is None and GradCAM is None)
                or ClassifierOutputTarget is None
                or show_cam_on_image is None
            ):
                logger.warning("Grad-CAM requested but pytorch-grad-cam not fully installed. Skipping.")
            else:
                try:
                    # Environment-driven configuration
                    cam_method = os.getenv("XAI_GRADCAM_METHOD", "gradcam").lower()  # gradcam | gradcam++ | scorecam
                    layer_index = int(os.getenv("XAI_GRADCAM_LAYER_INDEX", "-1"))  # e.g. -3 for earlier block
                    fusion_spec = os.getenv("XAI_GRADCAM_FUSION", "")  # e.g. "-3,-2,-1"

                    # Resolve candidate feature blocks
                    if hasattr(self.model, "features"):
                        feature_blocks = list(self.model.features)
                    elif hasattr(self.model, "layer4"):
                        feature_blocks = list(self.model.layer4)
                    else:
                        feature_blocks = [list(self.model.children())[-1]]

                    target_layers: List[nn.Module] = []
                    if fusion_spec.strip():
                        for idx_str in fusion_spec.split(","):
                            idx = int(idx_str.strip())
                            if -len(feature_blocks) <= idx < len(feature_blocks):
                                target_layers.append(feature_blocks[idx])
                            else:
                                logger.warning(f"Fusion layer index {idx} out of range; skipping.")
                    else:
                        # Single layer selection
                        if -len(feature_blocks) <= layer_index < len(feature_blocks):
                            target_layers = [feature_blocks[layer_index]]
                        else:
                            logger.warning(f"Layer index {layer_index} out of range; defaulting to last.")
                            target_layers = [feature_blocks[-1]]

                    # Select CAM class
                    if cam_method == "gradcam++" and GradCAMPlusPlus is not None:
                        cam_cls = GradCAMPlusPlus
                    elif cam_method == "scorecam" and ScoreCAM is not None:
                        cam_cls = ScoreCAM
                    else:
                        cam_cls = GradCAM  # Plain Grad-CAM default

                    # Some versions of pytorch-grad-cam don't accept use_cuda; handle gracefully.
                    try:
                        cam_instance = cam_cls(
                            model=self.model,
                            target_layers=target_layers,
                            use_cuda=(self.device == "cuda"),
                        )
                    except TypeError:
                        cam_instance = cam_cls(
                            model=self.model,
                            target_layers=target_layers,
                        )

                    self.explainers["grad_cam"] = {
                        "cam": cam_instance,
                        "method": cam_method,
                        "fusion": len(target_layers) > 1,
                        "layer_indices": [
                            (feature_blocks.index(l) if l in feature_blocks else None) for l in target_layers
                        ],
                    }
                    logger.info(
                        f"Grad-CAM initialized: method={cam_method}; layers={self.explainers['grad_cam']['layer_indices']}; fusion={self.explainers['grad_cam']['fusion']}; device={self.device}; targets={len(target_layers)}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to initialize Grad-CAM: {e}")

        # ---- Integrated Gradients ----
        if "integrated_grad" in self.methods:
            if IntegratedGradients is None:
                logger.warning("Integrated Gradients requested but captum not installed. Skipping.")
            else:
                try:
                    self.explainers["integrated_grad"] = IntegratedGradients(self.model)
                    logger.info("Integrated Gradients explainer initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Integrated Gradients: {e}")

        # ---- Saliency ----
        if "saliency" in self.methods:
            if Saliency is None:
                logger.warning("Saliency requested but captum not installed. Skipping.")
            else:
                try:
                    self.explainers["saliency"] = Saliency(self.model)
                    logger.info("Saliency explainer initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Saliency: {e}")

        # ---- LayerCAM ----
        if "layer_cam" in self.methods:
            if LayerCAM is None or ClassifierOutputTarget is None or show_cam_on_image is None:
                logger.warning("LayerCAM requested but pytorch-grad-cam not fully installed. Skipping.")
            else:
                try:
                    if hasattr(self.model, "features"):
                        lc_target_layers = [self.model.features[-1]]
                    elif hasattr(self.model, "layer4"):
                        lc_target_layers = [self.model.layer4[-1]]
                    else:
                        lc_target_layers = [list(self.model.children())[-1]]
                    self.explainers["layer_cam"] = LayerCAM(model=self.model, target_layers=lc_target_layers)
                    logger.info("LayerCAM explainer initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize LayerCAM: {e}")

        # ---- Occlusion ----
        if "occlusion" in self.methods:
            if Occlusion is None:
                logger.warning("Occlusion requested but captum not installed. Skipping.")
            else:
                try:
                    self.explainers["occlusion"] = Occlusion(self.model)
                    logger.info("Occlusion explainer initialized")
                except Exception as e:
                    logger.warning(f"Failed to initialize Occlusion: {e}")

        # ---- SHAP (lazy) ----
        if "shap" in self.methods:
            if shap is None:
                logger.warning("SHAP requested but shap not installed. Skipping.")
            else:
                if os.getenv("XAI_ENABLE_SHAP", "1") == "0":
                    logger.info("SHAP disabled via XAI_ENABLE_SHAP=0")
                    # Do not register SHAP explainer key to skip generation entirely
                else:
                    # Defer actual explainer creation until first use
                    self.explainers["shap"] = None
                    logger.info("SHAP explainer will be lazily initialized on first use")

    def explain_image(
        self,
        image: Union[Image.Image, str, Path],
        target_class: Optional[str] = None,
        target_index: Optional[int] = None,
        return_raw: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate explanations for a single image.

        Args:
            image: PIL Image, or path to image file
            target_class: Target class for explanations (if None, uses predicted class)
            return_raw: Whether to return raw attribution tensors

        Returns:
            Dictionary containing explanations from each method
        """
        # Load and preprocess image
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            image_name = image_path.stem
            original_image = Image.open(image_path).convert("RGB")
        else:
            image_name = "input_image"
            original_image = image

        # Determine target class/index for explanations
        prediction = None
        if target_class is None and target_index is None:
            prediction = self.classifier.predict(original_image)
            target_class = prediction["predicted_class"]
            logger.info(f"Using predicted class: {target_class} (confidence: {prediction['confidence']:.3f})")

        if target_index is None:
            if prediction is None:
                prediction = self.classifier.predict(original_image)
            target_index = prediction.get("xai_target_index")
            if target_index is None:
                # Fallback to small-head class index (e.g., binary head)
                target_index = self.class_names.index(target_class)

        input_tensor = self.classifier.preprocess_image(original_image)

        explanations = {
            "image_name": image_name,
            "target_class": target_class,
            "target_index": target_index,
            "original_image": original_image,
            "input_tensor": input_tensor,
        }

        # Generate explanations from each method
        for method_name in self.methods:
            if method_name in self.explainers:
                try:
                    explanation = self._generate_explanation(method_name, original_image, input_tensor, target_index)
                    explanations[method_name] = explanation

                    if self.save_explanations:
                        self._save_explanation(method_name, image_name, explanation)

                except Exception as e:
                    logger.error(f"Failed to generate {method_name} explanation: {e}")
                    explanations[method_name] = None

        return explanations

    def _generate_explanation(
        self, method_name: str, original_image: Image.Image, input_tensor: torch.Tensor, target_idx: int
    ) -> Dict[str, Any]:
        """Generate explanation using a specific method."""
        if method_name == "grad_cam":
            return self._grad_cam_explanation(original_image, input_tensor, target_idx)
        elif method_name == "integrated_grad":
            return self._integrated_gradients_explanation(input_tensor, target_idx)
        elif method_name == "saliency":
            return self._saliency_explanation(input_tensor, target_idx)
        elif method_name == "layer_cam":
            return self._layer_cam_explanation(original_image, input_tensor, target_idx)
        elif method_name == "occlusion":
            return self._occlusion_explanation(input_tensor, target_idx)
        elif method_name == "shap":
            return self._shap_explanation(input_tensor, target_idx)
        else:
            raise ValueError(f"Unknown explanation method: {method_name}")

    def _grad_cam_explanation(
        self, original_image: Image.Image, input_tensor: torch.Tensor, target_idx: int
    ) -> Dict[str, Any]:
        """Generate multi-layer fused Grad-CAM with optional guided backprop refinement."""
        expl = self.explainers["grad_cam"]
        cam_obj = expl["cam"]
        cam_method = expl.get("method", "gradcam")

        # Resolve candidate feature blocks (EfficientNet or ResNet fallback)
        if hasattr(self.model, "features"):
            feature_blocks = list(self.model.features)
        elif hasattr(self.model, "layer4"):
            feature_blocks = list(self.model.layer4)
        else:
            feature_blocks = [list(self.model.children())[-1]]

        # Determine layers to use (env fusion spec takes precedence)
        fusion_spec = os.getenv("XAI_GRADCAM_FUSION", "").strip()
        selected_layers: List[nn.Module] = []
        selected_indices: List[Optional[int]] = []

        if fusion_spec:
            for idx_str in fusion_spec.split(","):
                idx_str = idx_str.strip()
                if not idx_str:
                    continue
                try:
                    idx = int(idx_str)
                except ValueError:
                    continue
                if -len(feature_blocks) <= idx < len(feature_blocks):
                    selected_layers.append(feature_blocks[idx])
                    selected_indices.append(idx)

        if not selected_layers:
            # Fallback to configured layers on the CAM instance or last block
            try:
                configured_layers = getattr(cam_obj, "target_layers", None)
                if configured_layers:
                    selected_layers = list(configured_layers)
                    for lyr in selected_layers:
                        try:
                            selected_indices.append(feature_blocks.index(lyr))
                        except Exception:
                            selected_indices.append(None)
                else:
                    selected_layers = [feature_blocks[-1]]
                    selected_indices = [-1]
            except Exception:
                selected_layers = [feature_blocks[-1]]
                selected_indices = [-1]

        # Compute CAM per layer and fuse by mean
        targets = [ClassifierOutputTarget(target_idx)]
        cams: List[np.ndarray] = []
        cam_cls = type(cam_obj)
        for lyr in selected_layers:
            try:
                single_cam = cam_cls(model=self.model, target_layers=[lyr])
                raw = single_cam(input_tensor=input_tensor, targets=targets)[0]
                norm = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
                cams.append(norm)
            except Exception as e:
                logger.warning(f"Grad-CAM per-layer failed on {lyr}: {e}")

        if not cams:
            # Fallback: use the already-configured cam_obj result
            raw_cam_batch = cam_obj(input_tensor=input_tensor, targets=targets)
            fused = raw_cam_batch[0]
            fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
        else:
            stacked = np.stack(cams, axis=0)
            fused = stacked.mean(axis=0)
            fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)

        # Upscale fused CAM to original image size
        orig_w, orig_h = original_image.size
        if cv2 is not None:
            cam_up = cv2.resize(fused, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            cam_up = (
                np.array(Image.fromarray((fused * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR))
                / 255.0
            )

        # Optional Guided Backprop refinement
        guided_enabled = os.getenv("XAI_GRADCAM_GUIDED", "1") == "1"
        if guided_enabled and GuidedBackpropReLUModel is not None:
            try:
                gb_model = GuidedBackpropReLUModel(model=self.model)
                gb = gb_model(input_tensor, target_category=target_idx)[0]  # [C,H,W]
                gb = np.mean(np.abs(gb), axis=0)
                gb = (gb - gb.min()) / (gb.max() - gb.min() + 1e-8)

                # Match sizes if needed
                if gb.shape != cam_up.shape:
                    if cv2 is not None:
                        gb = cv2.resize(gb, (cam_up.shape[1], cam_up.shape[0]), interpolation=cv2.INTER_LINEAR)
                    else:
                        gb = (
                            np.array(
                                Image.fromarray((gb * 255).astype(np.uint8)).resize(
                                    (cam_up.shape[1], cam_up.shape[0]), Image.BILINEAR
                                )
                            )
                            / 255.0
                        )

                cam_up = cam_up * gb
                cam_up = (cam_up - cam_up.min()) / (cam_up.max() - cam_up.min() + 1e-8)
            except Exception as e:
                logger.warning(f"GuidedBackprop refinement failed: {e}")

        # Overlay visualization
        rgb = np.array(original_image.convert("RGB")) / 255.0
        visualization = show_cam_on_image(rgb, cam_up, use_rgb=True)

        return {
            "grayscale_cam": cam_up,
            "visualization": visualization,
            "method": f"Fused Grad-CAM ({cam_method})" + (" + GuidedBackprop" if guided_enabled else ""),
            "description": "Multi-layer fused CAM with optional guided backprop refinement.",
            "layers_used": selected_indices,
            "guided": guided_enabled,
        }

    def _integrated_gradients_explanation(self, input_tensor: torch.Tensor, target_idx: int) -> Dict[str, Any]:
        """Generate Integrated Gradients explanation."""
        # Calculate attributions
        attributions = self.explainers["integrated_grad"].attribute(
            input_tensor, target=target_idx, n_steps=50, internal_batch_size=1
        )

        # Process attributions
        attributions = attributions.squeeze(0).permute(1, 2, 0).cpu().numpy()
        attributions = np.abs(attributions)  # Take absolute values
        attributions = np.mean(attributions, axis=2)  # Average across channels

        # Normalize for visualization
        if attributions.max() > 0:
            attributions = attributions / attributions.max()

        return {
            "attributions": attributions,
            "method": "Integrated Gradients",
            "description": "Pixel-level importance scores showing contribution to classification",
        }

    def _saliency_explanation(self, input_tensor: torch.Tensor, target_idx: int) -> Dict[str, Any]:
        """Generate Saliency explanation."""
        # Calculate saliency
        saliency = self.explainers["saliency"].attribute(input_tensor, target=target_idx)

        # Process saliency
        saliency = saliency.squeeze(0).permute(1, 2, 0).cpu().numpy()
        saliency = np.abs(saliency)
        saliency = np.mean(saliency, axis=2)

        # Normalize
        if saliency.max() > 0:
            saliency = saliency / saliency.max()

        return {
            "attributions": saliency,
            "method": "Saliency",
            "description": "Gradient-based saliency map showing pixel importance",
        }

    def _layer_cam_explanation(
        self, original_image: Image.Image, input_tensor: torch.Tensor, target_idx: int
    ) -> Dict[str, Any]:
        targets = [ClassifierOutputTarget(target_idx)]
        cam = self.explainers["layer_cam"](input_tensor=input_tensor, targets=targets)[0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        orig_w, orig_h = original_image.size
        if cv2 is not None:
            cam_up = cv2.resize(cam, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            cam_up = (
                np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)) / 255.0
            )
        rgb = np.array(original_image.convert("RGB")) / 255.0
        visualization = show_cam_on_image(rgb, cam_up, use_rgb=True)
        return {
            "grayscale_cam": cam_up,
            "visualization": visualization,
            "method": "LayerCAM",
            "description": "Layer-wise CAM producing finer localization",
        }

    def _occlusion_explanation(self, input_tensor: torch.Tensor, target_idx: int) -> Dict[str, Any]:
        patch = int(os.getenv("OCCLUSION_PATCH", "32"))
        stride = int(os.getenv("OCCLUSION_STRIDE", "16"))
        baseline_mode = os.getenv("OCCLUSION_BASELINE", "blur")
        x = input_tensor.clone()
        if baseline_mode == "zero":
            baseline = torch.zeros_like(x)
        elif baseline_mode == "blur":
            baseline = torch.nn.functional.avg_pool2d(x, kernel_size=7, stride=1, padding=3)
        else:
            baseline = torch.zeros_like(x)

        occlusion_attr = self.explainers["occlusion"].attribute(
            x,
            strides=(1, 1, stride, stride),
            target=target_idx,
            sliding_window_shapes=(1, x.size(1), patch, patch),
            baselines=baseline,
        )
        occ = occlusion_attr.squeeze(0).abs().mean(dim=0).cpu().numpy()
        if occ.max() > 0:
            occ = occ / occ.max()

        return {
            "attributions": occ,
            "method": "Occlusion",
            "description": "Probability drop-based occlusion attribution",
        }

    def _shap_explanation(self, input_tensor: torch.Tensor, target_idx: int) -> Dict[str, Any]:
        """Generate SHAP explanation with graceful fallback for inplace-view errors."""
        if "shap" not in self.explainers:
            raise RuntimeError("SHAP not available or disabled")

        input_for_shap = input_tensor.clone().detach().requires_grad_(True)

        # Lazy initialization
        if self.explainers["shap"] is None:
            baseline = torch.zeros_like(input_for_shap).to(self.device)
            try:
                self.explainers["shap"] = shap.DeepExplainer(self.model, baseline)
                shap_method = "DeepExplainer"
            except Exception as e:
                self.explainers["shap"] = shap.GradientExplainer(self.model, baseline)
                shap_method = "GradientExplainer"
                logger.warning(f"Falling back to SHAP GradientExplainer due to init error: {e}")

        # Compute SHAP values with fallback for inplace-view error
        try:
            shap_values = self.explainers["shap"].shap_values(input_for_shap)
            shap_method = getattr(self.explainers["shap"], "__class__", type("X", (), {})).__name__
        except RuntimeError as e:
            msg = str(e).lower()
            if "view" in msg and "inplace" in msg:
                baseline = torch.zeros_like(input_for_shap).to(self.device)
                self.explainers["shap"] = shap.GradientExplainer(self.model, baseline)
                shap_values = self.explainers["shap"].shap_values(input_for_shap)
                shap_method = "GradientExplainer"
                logger.warning("SHAP DeepExplainer raised inplace-view error; retried with GradientExplainer.")
            else:
                raise

        if isinstance(shap_values, list):
            target_shap = shap_values[target_idx]
        else:
            target_shap = shap_values[target_idx : target_idx + 1]

        target_shap = target_shap.squeeze(0).permute(1, 2, 0).cpu().numpy()
        shap_importance = np.abs(target_shap).mean(axis=2)

        if shap_importance.max() > 0:
            shap_importance = shap_importance / shap_importance.max()

        return {
            "shap_values": target_shap,
            "importance_map": shap_importance,
            "method": f"SHAP {shap_method}",
            "description": "SHAP values showing feature contributions to classification (with inplace-view fallback)",
        }

    def _save_explanation(self, method_name: str, image_name: str, explanation: Dict[str, Any]):
        """Save explanation visualization."""
        if not self.save_explanations or explanation is None:
            return

        method_dir = self.output_dir / method_name
        method_dir.mkdir(exist_ok=True)

        if "visualization" in explanation:
            # Save Grad-CAM visualization
            vis_path = method_dir / f"{image_name}_{method_name}.png"
            vis_image = Image.fromarray(explanation["visualization"])
            vis_image.save(vis_path)
            logger.info(f"Saved {method_name} visualization to {vis_path}")

        if "attributions" in explanation or "importance_map" in explanation:
            # Save attribution heatmap
            attr_key = "attributions" if "attributions" in explanation else "importance_map"
            attr_map = explanation[attr_key]

            # Convert to colormap visualization
            import matplotlib.cm as cm

            heatmap = cm.hot(attr_map)[:, :, :3]  # Take RGB channels
            heatmap = (heatmap * 255).astype(np.uint8)

            heatmap_path = method_dir / f"{image_name}_{method_name}_heatmap.png"
            heatmap_image = Image.fromarray(heatmap)
            heatmap_image.save(heatmap_path)
            logger.info(f"Saved {method_name} heatmap to {heatmap_path}")

    def explain_batch(
        self, images: List[Union[Image.Image, str, Path]], target_classes: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate explanations for multiple images.

        Args:
            images: List of PIL Images or image paths
            target_classes: List of target classes (if None, uses predicted classes)

        Returns:
            List of explanation dictionaries
        """
        if target_classes is None:
            target_classes = [None] * len(images)
        elif len(target_classes) != len(images):
            raise ValueError("Number of target classes must match number of images")

        results = []
        for image, target_class in zip(images, target_classes):
            try:
                explanation = self.explain_image(image, target_class)
                results.append(explanation)
            except Exception as e:
                logger.error(f"Failed to explain image {image}: {e}")
                results.append(None)

        return results

    def compare_explanations(
        self, image: Union[Image.Image, str, Path], methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare explanations from different methods for the same image.

        Args:
            image: PIL Image, or path to image file
            methods: Methods to compare (if None, uses all available methods)

        Returns:
            Dictionary comparing explanations from different methods
        """
        if methods is None:
            methods = list(self.explainers.keys())

        # Generate explanations
        explanations = self.explain_image(image)

        # Create comparison
        comparison = {
            "image_name": explanations["image_name"],
            "target_class": explanations["target_class"],
            "methods": {},
        }

        for method in methods:
            if method in explanations and explanations[method] is not None:
                method_explanation = explanations[method]

                # Extract key metrics for comparison
                method_data = {"method": method_explanation["method"], "description": method_explanation["description"]}

                if "attributions" in method_explanation:
                    attr_map = method_explanation["attributions"]
                    method_data["statistics"] = {
                        "mean_importance": float(np.mean(attr_map)),
                        "max_importance": float(np.max(attr_map)),
                        "std_importance": float(np.std(attr_map)),
                        "active_pixels": int(np.sum(attr_map > 0.1)),  # Threshold
                    }

                if "grayscale_cam" in method_explanation:
                    cam_map = method_explanation["grayscale_cam"]
                    method_data["statistics"] = {
                        "mean_activation": float(np.mean(cam_map)),
                        "max_activation": float(np.max(cam_map)),
                        "std_activation": float(np.std(cam_map)),
                        "active_regions": int(np.sum(cam_map > 0.1)),
                    }

                comparison["methods"][method] = method_data

        return comparison

    def get_explainer_info(self) -> Dict[str, Any]:
        """Get information about available explainers."""
        info = {
            "available_methods": list(self.explainers.keys()),
            "configured_methods": self.methods,
            "device": self.device,
            "output_directory": str(self.output_dir),
            "save_explanations": self.save_explanations,
        }

        for method, explainer in self.explainers.items():
            if explainer is not None:
                info[f"{method}_status"] = "initialized"
            else:
                info[f"{method}_status"] = "available (lazy initialization)"

        return info


# Convenience function for quick explanations
def explain_image(
    image: Union[Image.Image, str, Path],
    classifier: Optional[EfficientNetCatDogClassifier] = None,
    methods: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Quick explanation function for single images.

    Args:
        image: PIL Image, or path to image file
        classifier: CNN classifier instance (if None, creates new instance)
        methods: XAI methods to use

    Returns:
        Explanation dictionary
    """
    if classifier is None:
        classifier = EfficientNetCatDogClassifier()

    explainer = XAIExplainer(classifier, methods=methods)
    return explainer.explain_image(image)


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Test XAI Explainer")
    parser.add_argument("image", type=Path, help="Path to test image")
    parser.add_argument("--model", type=Path, help="Path to fine-tuned model weights")
    parser.add_argument("--methods", nargs="+", default=["grad_cam", "integrated_grad"], help="XAI methods to use")
    parser.add_argument("--output-dir", type=Path, help="Output directory for explanations")

    args = parser.parse_args()

    # Initialize classifier and explainer
    classifier = EfficientNetCatDogClassifier(model_path=args.model)
    explainer = XAIExplainer(classifier, methods=args.methods, output_dir=args.output_dir)

    # Print explainer info
    info = explainer.get_explainer_info()
    print("Explainer Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    print()

    # Test explanation generation
    if args.image.exists():
        explanation = explainer.explain_image(args.image)
        print(f"Image: {explanation['image_name']}")
        print(f"Target class: {explanation['target_class']}")
        print(f"Generated explanations for: {list(explanation.keys())}")

        # Compare methods if multiple are available
        if len(args.methods) > 1:
            comparison = explainer.compare_explanations(args.image)
            print("\nMethod Comparison:")
            for method, data in comparison["methods"].items():
                print(f"  {method}: {data['method']}")
                if "statistics" in data:
                    stats = data["statistics"]
                    print(f"    Mean importance: {stats.get('mean_importance', stats.get('mean_activation', 0)):.3f}")
    else:
        print(f"Image not found: {args.image}")
