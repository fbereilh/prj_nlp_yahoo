import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from pathlib import Path
from typing import Optional
from torch.serialization import add_safe_globals
from fastcore.foundation import L
import logging
import os
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add required safe globals
add_safe_globals([L, np.core.multiarray.scalar])

# Model configuration
MODEL_CONFIG = {
    'embed_dim': 128,
    'num_heads': 2,
    'num_classes': 10,
    'hidden_dim': 256,
    'dropout': 0.1,
    'max_length': 128,
    'categories': [
        'Society & Culture',
        'Science & Mathematics',
        'Health',
        'Education & Reference',
        'Computers & Internet',
        'Sports',
        'Business & Finance',
        'Entertainment & Music',
        'Family & Relationships',
        'Politics & Government'
    ]
}

# Paths
MODEL_PATH = Path("models/yahoo_classifier.pth")
TOKENIZER_PATH = Path("models/tokenizer.json")

# Global variables for model and tokenizer
tokenizer: Optional[PreTrainedTokenizerFast] = None
model: Optional[nn.Module] = None

class YahooTextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_classes, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids):
        x = self.embedding(input_ids)  # (B, T, D)

        attn_output, _ = self.attention(x, x, x)  # No attention mask
        x = self.norm(x + self.dropout(attn_output))

        pooled = x.mean(dim=1)  # Mean over time steps

        return self.classifier(pooled)

def load_tokenizer(tokenizer_file=TOKENIZER_PATH):
    """Load and configure the tokenizer"""
    logger.info(f"Loading tokenizer from {tokenizer_file}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
    tokenizer.pad_token = "[PAD]"
    tokenizer.unk_token = "[UNK]"
    logger.info(f"Tokenizer loaded successfully. Vocab size: {tokenizer.vocab_size}")
    return tokenizer

def create_model(vocab_size):
    """Helper function to create model instance with configured parameters"""
    logger.info(f"Creating model with vocab size: {vocab_size}")
    return YahooTextClassifier(
        vocab_size=vocab_size,
        embed_dim=MODEL_CONFIG['embed_dim'],
        num_heads=MODEL_CONFIG['num_heads'],
        num_classes=MODEL_CONFIG['num_classes'],
        hidden_dim=MODEL_CONFIG['hidden_dim'],
        dropout=MODEL_CONFIG['dropout']
    )

def load_model():
    """Load tokenizer and model"""
    global tokenizer, model
    try:
        logger.info("Starting model loading process...")
        tokenizer = load_tokenizer()
        model = create_model(vocab_size=tokenizer.vocab_size)
        
        # Check environment variable for CUDA usage
        use_cuda = int(os.getenv('USE_CUDA', '0')) and torch.cuda.is_available()
        device = 'cuda' if use_cuda else 'cpu'
        logger.info(f"Using device: {device}")
        
        # Load model with memory efficient settings
        logger.info(f"Loading model weights from {MODEL_PATH}")
        try:
            # First try with weights_only=True
            checkpoint = torch.load(
                str(MODEL_PATH),
                map_location=device,
                weights_only=True
            )
        except Exception as e:
            logger.warning(f"Failed to load with weights_only=True: {str(e)}")
            logger.info("Attempting to load with weights_only=False")
            # If that fails, try with weights_only=False
            checkpoint = torch.load(
                str(MODEL_PATH),
                map_location=device,
                weights_only=False
            )
        
        # Handle checkpoint loading
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            logger.info("Loading from checkpoint dictionary")
            model.load_state_dict(checkpoint['model'])
        else:
            logger.info("Loading direct state dict")
            model.load_state_dict(checkpoint)
        
        model.eval()  # Set to evaluation mode
        if use_cuda:
            model = model.cuda()
            # Enable CUDA optimizations
            torch.backends.cudnn.benchmark = True
        else:
            # Enable CPU optimizations
            torch.set_num_threads(os.cpu_count())
            
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def predict_text(text: str) -> dict:
    """Make prediction for input text"""
    if tokenizer is None or model is None:
        logger.error("Model or tokenizer not loaded")
        return {"error": "Model not loaded"}
    
    try:
        logger.info(f"Tokenizing text: {text[:50]}...")
        tokens = tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MODEL_CONFIG['max_length']
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        input_ids = tokens["input_ids"].to(device)
        
        logger.info("Running prediction...")
        with torch.no_grad():
            logits = model(input_ids)
            probs = torch.nn.functional.softmax(logits, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            
            # Create probabilities dictionary
            probabilities = {
                cat: float(prob)  # Convert to float for JSON serialization
                for cat, prob in zip(MODEL_CONFIG['categories'], probs)
            }
            
            prediction = {
                "prediction": {
                    "category": MODEL_CONFIG['categories'][pred_idx],
                    "index": pred_idx,
                    "confidence": float(probs[pred_idx])
                },
                "probabilities": probabilities,
                "status": "success"
            }
            logger.info(f"Prediction successful: {prediction['prediction']}")
            return prediction
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        return {"error": str(e)} 