import os
import logging
from pathlib import Path
import shutil
from sentence_transformers import SentenceTransformer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_models():
    """Setup model files in the container"""
    try:
        # Create directories if they don't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        # Check if models already exist in the container
        if (models_dir / "yahoo_classifier.pth").exists() and (models_dir / "tokenizer.json").exists():
            logger.info("Classifier model files already exist in container")
        else:
            # Check if models exist in the source directory
            source_models_dir = Path("./models")
            if not source_models_dir.exists():
                logger.error("Source models directory not found")
                raise FileNotFoundError("Models directory not found in source")
                
            required_files = ["yahoo_classifier.pth", "tokenizer.json"]
            for file in required_files:
                source_file = source_models_dir / file
                if not source_file.exists():
                    logger.error(f"Required model file {file} not found in source")
                    raise FileNotFoundError(f"Model file {file} not found")
                
                # Copy the file to the container's models directory
                shutil.copy2(source_file, models_dir / file)
                logger.info(f"Copied {file} to container")
        
        # Download and cache sentence-transformer model
        logger.info("Setting up sentence-transformer model...")
        model_name = 'all-MiniLM-L6-v2'
        
        # This will automatically download and cache the model
        SentenceTransformer(model_name)
        logger.info(f"Sentence-transformer model {model_name} setup completed")
        
        logger.info("All model files setup completed successfully")
        
    except Exception as e:
        logger.error(f"Error setting up models: {str(e)}")
        raise

if __name__ == "__main__":
    setup_models() 