#!/bin/bash

# Array of questions from different categories
questions=(
    # Computers & Internet
    "What are the best practices for securing a web application against common cyber attacks?"
    "How do I choose between different cloud providers like AWS, Azure, and Google Cloud?"
    "What's the difference between machine learning and deep learning in AI?"
    
    # Science & Mathematics
    "Can you explain the concept of quantum superposition in simple terms?"
    "How does the theory of relativity affect our understanding of time and space?"
    "What role do dark matter and dark energy play in the universe?"
    
    # Health
    "What are the most effective ways to boost immune system naturally?"
    "How does stress affect mental and physical health long-term?"
    "What's the connection between gut health and overall wellbeing?"
    
    # Education & Reference
    "What are the most effective study techniques based on cognitive science?"
    "How can I improve my critical thinking and analytical skills?"
    "What are the best ways to learn a new language as an adult?"
    
    # Business & Finance
    "How do I create a solid investment portfolio for long-term growth?"
    "What are the key factors to consider when starting a small business?"
    "How do cryptocurrency markets work and what affects their value?"
    
    # Sports
    "What are the most effective training methods for improving athletic performance?"
    "How do professional athletes maintain peak performance throughout a season?"
    "What role does mental preparation play in sports success?"
    
    # Entertainment & Music
    "How has streaming technology changed the music industry?"
    "What makes certain movies become cultural phenomena?"
    "How do film scores enhance storytelling in cinema?"
    
    # Family & Relationships
    "What are effective strategies for maintaining work-life balance?"
    "How can parents help children develop emotional intelligence?"
    "What are the keys to building lasting relationships?"
    
    # Politics & Government
    "How do different electoral systems affect democratic representation?"
    "What impact does lobbying have on policy-making?"
    "How do international trade agreements affect local economies?"
    
    # Society & Culture
    "How has social media transformed human interaction and society?"
    "What role does cultural diversity play in global business?"
    "How do generational differences affect workplace dynamics?"
)

# Loop through each question and send it to the API
for question in "${questions[@]}"; do
    echo "Sending question: $question"
    curl -X POST \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "text=$question" \
        https://yahoo.fbereilh.com/predict
    echo -e "\n---\n"
    # Add a small delay to avoid overwhelming the server
    sleep 2
done 