from model import predict_text, load_model
from database import add_prediction

# Example questions covering different categories
example_texts = [
    # Programming & Technology
    "What's the best way to learn Python programming for beginners? I've heard about different online courses but not sure which one to choose.",
    "How do I set up a local development environment for web development? Need help with Node.js, npm, and VS Code.",
    "Can someone explain the difference between SQL and NoSQL databases? When should I use each one?",
    "What are the pros and cons of using Docker containers in development? Is it worth learning for a junior developer?",
    
    # Science & Mathematics
    "Can someone explain quantum entanglement in simple terms? I'm trying to understand the basics of quantum physics.",
    "How does climate change affect ocean acidification? What are the consequences for marine life?",
    "What's the mathematical principle behind the golden ratio? Where can we find it in nature?",
    "Could you explain how mRNA vaccines work? What makes them different from traditional vaccines?",
    
    # Health & Medicine
    "What are the early symptoms of diabetes? My family has a history of it and I want to be aware of the signs.",
    "How effective is intermittent fasting for weight loss? Are there any health risks I should be aware of?",
    "What's the difference between a cold and seasonal allergies? How can I tell them apart?",
    "How does stress affect sleep quality? What are some evidence-based techniques for better sleep?",
    
    # Business & Finance
    "What's the difference between stocks and bonds? I'm new to investing and trying to understand basic financial concepts.",
    "How do I create a business plan for a small startup? What are the essential components?",
    "Can someone explain cryptocurrency mining? How does it actually work?",
    "What are the key factors to consider when choosing between a traditional IRA and a Roth IRA?",
    
    # Society & Culture
    "How has social media influenced modern political movements? Looking for examples and analysis.",
    "What are the main differences between Eastern and Western philosophy? Particularly interested in concepts of self.",
    "How do different cultures approach mental health? What can we learn from various traditional practices?",
    "What role does art play in social change? Looking for historical examples and modern perspectives."
]

def main():
    print("Loading model...")
    if not load_model():
        print("Failed to load model!")
        return
    
    print("\nAdding example predictions to database...")
    for i, text in enumerate(example_texts, 1):
        print(f"\nProcessing example {i}/{len(example_texts)}:")
        print(f"Text: {text[:100]}...")
        
        result = predict_text(text)
        if "error" in result:
            print(f"Error predicting: {result['error']}")
            continue
            
        pred = add_prediction(text, result)
        print(f"Added prediction: {pred.category} (confidence: {pred.confidence:.1%})")
    
    print(f"\nDone! Added {len(example_texts)} example predictions to the database.")

if __name__ == "__main__":
    main() 