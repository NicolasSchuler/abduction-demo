import os
import sys
import json
import logging
import re
from pathlib import Path
from PIL import Image
import numpy as np
import argparse

# Set environment variables for better XAI results BEFORE importing config
os.environ["XAI_METHODS"] = "layer_cam" # Use LayerCAM for fine-grained feature highlighting

# Add current directory to sys.path to allow imports
sys.path.append(os.getcwd())

from src import config
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline
import main_dspy

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("experiment")

def parse_default_probabilities(file_path):
    """Parses the ProbLog file to extract default probabilities."""
    probs = {}
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

def main():
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
    
    count = 0
    for image_path in images:
        count += 1
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
            
            with open(results_file, "a") as f:
                f.write(json.dumps(result_entry) + "\n")
                
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            continue

if __name__ == "__main__":
    main()
