from fasthtml.common import *
from fastlite import *
from dataclasses import dataclass
from datetime import datetime
import hashlib

# Initialize database
db = database('data/predictions.db')

@dataclass
class Prediction:
    id: str  # Primary key - will be hash of text
    text: str  # Input text
    category: str  # Predicted category
    confidence: float  # Prediction confidence
    probabilities: str  # JSON string of all probabilities
    created_at: str = None  # Timestamp

def generate_id(text: str) -> str:
    """Generate a hash ID from the text"""
    # Use SHA-256 to generate a hash of the text
    # We'll take first 16 characters of the hex digest as our ID
    return hashlib.sha256(text.encode()).hexdigest()[:16]
    
# Create the predictions table if it doesn't exist
predictions = db.create(
    Prediction,
    pk='id',  # Specify id as primary key
    transform=True  # Allows schema updates if we change the class
)

def add_prediction(text: str, result: dict) -> Prediction:
    """Add a new prediction to the database"""
    import json
    
    # Generate ID from text
    pred_id = generate_id(text)
    
    try:
        # Try to get existing prediction
        return predictions[pred_id]
    except NotFoundError:
        # Create new prediction record if it doesn't exist
        # Use timestamp format that sorts correctly in SQLite
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        pred = predictions.insert(
            id=pred_id,
            text=text,
            category=result['prediction']['category'],
            confidence=result['prediction']['confidence'],
            probabilities=json.dumps(result['probabilities']),
            created_at=timestamp
        )
        return pred

def get_recent_predictions(limit: int = 10) -> list[Prediction]:
    """Get the most recent predictions"""
    return predictions(order_by='-created_at', limit=limit)

def get_prediction(id: str) -> Prediction:
    """Get a specific prediction by ID"""
    return predictions[id]

def delete_prediction(id: str) -> bool:
    """Delete a prediction by ID"""
    try:
        predictions.delete(id)
        return True
    except NotFoundError:
        return False

def get_all_predictions() -> list[Prediction]:
    """Get all predictions from the database"""
    return predictions() 