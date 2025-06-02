# Yahoo! Answers Text Classification Project

## Overview
This project implements a text classification system using the Yahoo! Answers dataset to automatically categorize user-submitted questions into one of ten predefined categories. The project demonstrates both the power of transformer models and the potential for efficient, lightweight custom architectures in practical NLP applications.

## Live Demo
The project is deployed and accessible at: [https://yahoo.fbereilh.com](https://yahoo.fbereilh.com)

## Project Highlights

### Objectives
- Develop and compare different approaches to text classification
- Create an efficient, production-ready model for real-world deployment
- Demonstrate the trade-offs between model complexity and performance

### Key Features
- Implementation of multiple model architectures:
  - Baseline traditional machine learning model
  - Fine-tuned transformer model
  - Custom lightweight PyTorch architecture
- Efficient model design achieving comparable performance with only 10% of parameters
- Production-ready deployment with web interface

### Dataset
The project uses the Yahoo! Answers Topics dataset, which includes:
- Approximately 140,000 samples
- 10 distinct categories
- Long-form text (200-500 words per entry)
- Real-world Q&A forum data

## Key Findings

### Technical Insights
- Transformer models provide excellent out-of-the-box performance for complex NLP tasks
- Custom lightweight architectures can achieve comparable results with significantly reduced computational requirements
- Successful deployment validates the model's robustness in real-world scenarios

### Industry Applications
The developed solution has potential applications in various domains:
- Email and support ticket categorization
- Content moderation systems
- News and document classification
- Semantic tagging for legal, financial, or healthcare documents
- Information retrieval optimization

