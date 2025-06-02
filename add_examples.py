from model import predict_text, load_model
from database import add_prediction

# Example questions covering different categories
example_texts = [
    # Computers & Internet
    "What's the best way to learn Python programming for beginners? I've heard about different online courses but not sure which one to choose.",
    "How do I protect my computer from viruses and malware? Looking for free and paid options.",
    "What's the difference between 4G and 5G networks? How much faster is 5G really?",
    
    # Science & Mathematics
    "Can someone explain quantum entanglement in simple terms? I'm trying to understand the basics of quantum physics.",
    "How do black holes work and what happens if you fall into one?",
    "What's the mathematical explanation for the Fibonacci sequence in nature?",
    
    # Health
    "What are the early symptoms of diabetes? My family has a history of it and I want to be aware of the signs.",
    "How can I improve my immune system naturally? Looking for diet and lifestyle tips.",
    "What's the difference between good and bad cholesterol? How can I maintain healthy levels?",
    
    # Education & Reference
    "What are some effective study techniques for college exams?",
    "How can I improve my public speaking skills? Need tips for presentations.",
    "What's the best way to learn a new language as an adult?",
    
    # Business & Finance
    "How do I start investing in stocks with little money? Need beginner advice.",
    "What's the difference between a bull and bear market?",
    "How do cryptocurrency exchanges work? Is it safe to use them?",
    
    # Sports
    "What's the offside rule in soccer? Can someone explain it simply?",
    "How do I improve my basketball shooting technique?",
    "What are the basic rules of cricket? It seems complicated.",
    
    # Entertainment & Music
    "Who are considered the greatest guitarists of all time and why?",
    "What's the difference between jazz and blues music?",
    "How do movie special effects work? Especially interested in CGI.",
    
    # Family & Relationships
    "How do I deal with a difficult teenager? Need parenting advice.",
    "What are some tips for maintaining a long-distance relationship?",
    "How to handle conflicts with in-laws diplomatically?",
    
    # Politics & Government
    "How does the electoral college system work in the US?",
    "What's the difference between democracy and republic?",
    "How do international trade agreements affect local economies?",
    
    # Society & Culture
    "How has social media influenced modern society?",
    "What are the main differences between Eastern and Western cultures?",
    "How do different societies approach work-life balance?"
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