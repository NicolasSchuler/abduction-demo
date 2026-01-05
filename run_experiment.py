"""
Experiment runner for batch image processing with CNN+XAI and ProbLog inference.

This script processes multiple images using the preprocessing pipeline and
grounding model to generate classification results.
"""
import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

# Set environment variables for better XAI results BEFORE importing config
os.environ["XAI_METHODS"] = "layer_cam"  # Use LayerCAM for fine-grained feature highlighting

# Add project root to sys.path for imports (use script location, not cwd)
_PROJECT_ROOT = Path(__file__).parent.absolute()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
import main_dspy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("experiment")

def parse_default_probabilities(file_path: str | Path) -> Dict[str, Dict[str, float]]:
    """Parses the ProbLog file to extract default probabilities.

    Args:
        file_path: Path to the ProbLog file containing prob_cat/prob_dog facts.

    Returns:
        Dictionary mapping feature names to their cat/dog probabilities.
    """
    probs: Dict[str, Dict[str, float]] = {}
    try:
        content = Path(file_path).read_text()
        # Regex to match prob_cat(feature, prob) and prob_dog(feature, prob)
        # Example: prob_cat(flat_forehead, 0.9).
        cat_matches = re.findall(r'prob_cat\(([\w_]+),\s*([0-9.]+)\)', content)
        dog_matches = re.findall(r'prob_dog\(([\w_]+),\s*([0-9.]+)\)', content)
        
        for feature, prob in cat_matches:
            if feature not in probs: probs[feature] = {}
            probs[feature]['cat'] = float(prob)
            
        for feature, prob in dog_matches:
            if feature not in probs: probs[feature] = {}
            probs[feature]['dog'] = float(prob)
            
        return probs
    except Exception as e:
        logger.error(f"Failed to parse default probabilities: {e}")
        return {}

def main() -> None:
    """Main experiment runner function."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run experiment with configurable grounding model")
    parser.add_argument("--model-grounding", type=str, default=config.MODEL_GROUNDING, help="Model for grounding stage")
    args = parser.parse_args()
    
    # Update config
    config.MODEL_GROUNDING = args.model_grounding
    logger.info(f"Using grounding model: {config.MODEL_GROUNDING}")

    # Configuration
    collage_dir = Path("collage")
    output_dir = Path("results/experiment")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results file
    results_file = output_dir / "experiment_results.jsonl"
    
    # Parse default probabilities
    default_probs = parse_default_probabilities("result-prompts/coding.pl")
    
    # Initialize Preprocessing Pipeline
    # We enable CNN and XAI to get probabilities and Grad-CAM
    # We disable enhancement as per user request "disable-enhancement" equivalent
    pipeline = PreprocessingPipeline(
        enable_cnn=True,
        enable_xai=True,
        enable_enhancement=False,
        save_outputs=False # We will save manually
    )
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = []
    for root, _, files in os.walk(collage_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                images.append(Path(root) / file)
    
    logger.info(f"Found {len(images)} images to process.")
    
    # Load cached data for partial mode
    # We can reuse main_dspy functions but we need to be careful about what they expect
    # run_partial_mode expects config.CACHED_* files to be present.
    # We assume they are.
    
    # We need to ensure config is set up correctly for partial mode
    # main_dspy.run_partial_mode uses config.CACHED_REASONING_FILE etc.
    
    # Keep file handle open for batch writes (more efficient than open/close per image)
    with open(results_file, "a") as results_handle:
        for count, image_path in enumerate(images, start=1):
            logger.info(f"Processing image {count}/{len(images)}: {image_path}")

            try:
                # 1. Run Preprocessing (CNN + XAI)
                # We need this to get CNN probs and Grad-CAM
                proc_result = pipeline.process_image(image_path)

                # Extract CNN probabilities
                cnn_probs = {}
                if proc_result.get("classification"):
                    cnn_probs = proc_result["classification"].get("probabilities", {})

                # Extract and store LayerCAM image (Best for feature localization)
                layer_cam_path = output_dir / "layer_cam" / f"{image_path.stem}_layercam.png"
                layer_cam_path.parent.mkdir(parents=True, exist_ok=True)

                if proc_result.get("explanations") and "layer_cam" in proc_result["explanations"]:
                    layer_cam_data = proc_result["explanations"]["layer_cam"].get("visualization")
                    if layer_cam_data is not None:
                        Image.fromarray(layer_cam_data).save(layer_cam_path)

                # 2. Run Grounding and Inference (Partial Mode)
                # We run grounding on the LayerCAM image

                logger.info(f"Grounding on LayerCAM image: {layer_cam_path}")
                reasoning_desc, problog_prog, feature_list, grounding_res = main_dspy.run_partial_mode(
                    pipeline_image_path=layer_cam_path,
                    highlighted_only=True,
                    preprocessing_result=proc_result,
                    cnn_classification=proc_result.get("classification")
                )
                p_cat, p_dog = main_dspy.execute_logic_program(problog_prog, grounding_res)

                # 4. Store Results
                result_entry = {
                    "image_path": str(image_path),
                    "cnn_probabilities": cnn_probs,
                    "xai_image_path": str(layer_cam_path),
                    "xai_method": "layer_cam",
                    "grounding_results": grounding_res,
                    "grounding_model": config.MODEL_GROUNDING,
                    "inferred_probabilities": {
                        "cat": p_cat,
                        "dog": p_dog
                    },
                    "default_probabilities": default_probs
                }

                results_handle.write(json.dumps(result_entry) + "\n")
                results_handle.flush()  # Ensure data is written even if crash occurs

            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                continue

if __name__ == "__main__":
    main()
