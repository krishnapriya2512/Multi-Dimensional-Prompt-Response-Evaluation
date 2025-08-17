#!/bin/bash
# Pre-download Hugging Face models so they are cached during deployment

# Sentiment model
python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"

# Summarizer model
python -c "from transformers import pipeline; pipeline('summarization', model='facebook/bart-large-cnn')"
