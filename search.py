from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple
import logging
from database import Prediction, get_all_predictions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticSearch:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.predictions: List[Prediction] = []
        self.embeddings = None
        
    def build_index(self, predictions: List[Prediction]):
        """Build FAISS index from predictions"""
        logger.info("Building FAISS index...")
        self.predictions = predictions
        
        if not predictions:
            logger.info("No predictions to index")
            return
        
        # Get embeddings for all texts
        texts = [p.text for p in predictions]
        embeddings = self.model.encode(texts)
        embeddings = embeddings.reshape(len(texts), -1)  # Ensure 2D array
        
        # Normalize embeddings for cosine similarity
        # faiss.normalize_L2(embeddings)
        # already normalized in this model
        self.embeddings = embeddings
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity for normalized vectors
        
        # Add vectors to the index
        self.index.add(embeddings.astype('float32'))
        logger.info(f"Built index with {len(predictions)} predictions")
        
    def search(self, query: str) -> List[Tuple[Prediction, float]]:
        """Search for similar predictions
        
        Args:
            query: The search query
        Returns:
            List of (prediction, similarity_score) tuples, sorted by similarity
        """
        if not self.index or not self.predictions:
            logger.warning("No predictions indexed yet")
            return []
            
        # Get query embedding and normalize it
        query_embedding = self.model.encode([query])
        query_embedding = query_embedding.reshape(1, -1)  # Ensure 2D array
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index - get all results
        k = len(self.predictions)  # Get all predictions
        similarities, indices = self.index.search(query_embedding.astype('float32'), k)
        
        # Log all results for debugging
        logger.info(f"Search results for query: {query}")
        for idx, sim in zip(indices[0], similarities[0]):
            if idx < len(self.predictions):
                logger.info(f"Text: {self.predictions[idx].text[:100]}...")
                logger.info(f"Similarity: {sim:.3f}")
        
        # Return all results sorted by similarity
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            if idx < len(self.predictions):  # Safety check
                results.append((self.predictions[idx], float(sim)))
                
        # Sort by similarity (most similar first)
        results.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Returning {len(results)} results")
        return results

# Global search instance
semantic_search = SemanticSearch()

def initialize_search():
    """Initialize the search index with all predictions"""
    try:
        predictions = get_all_predictions()
        if not predictions:
            logger.warning("No predictions found in database - search index will be empty")
            return True  # Return success but empty
            
        semantic_search.build_index(predictions)
        logger.info(f"Built search index with {len(predictions)} predictions")
        return True
    except Exception as e:
        logger.error(f"Error initializing search: {str(e)}")
        return False

def reinitialize_search():
    """Reinitialize the search index - call this after adding new predictions"""
    return initialize_search()

def search_predictions(query: str) -> List[Tuple[Prediction, float]]:
    """Search for similar predictions"""
    # Try to reinitialize if index is empty
    if not semantic_search.index or not semantic_search.predictions:
        logger.info("Search index empty, attempting to reinitialize...")
        initialize_search()
    return semantic_search.search(query) 