# GenAI with Hugging Face

This project contains various Natural Language Processing (NLP) and Generative AI implementations using Hugging Face's transformers and tools.

## Features

- Trip Itinerary Generator UI
- Various NLP implementations:
  - Tokenization
  - Word Embeddings
  - BOW (Bag of Words)
  - TF-IDF
  - N-grams
  - RNN
  - Named Entity Recognition
  - Chatbot implementation

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with your Hugging Face API credentials
4. Run the UI:
```bash
streamlit run prompt_ui.py
```

## Files

- `prompt_ui.py`: Main Streamlit interface for trip itinerary generation
- `Chatbot_HF.py`: Hugging Face chatbot implementation
- `word_embedding.py`: Word embedding implementations
- `RNN.py`: Recurrent Neural Network implementation
- And more NLP-related implementations

## Requirements

See `requirements.txt` for a full list of dependencies.