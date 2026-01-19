# ROUGE-L Score Implementation for LLM Summarization Evaluation

This project implements the **ROUGE-L** (Recall-Oriented Understudy for Gisting Evaluation with Longest Common Subsequence) metric from scratch for evaluating text summarization quality. ROUGE-L measures the quality of machine-generated summaries by computing the longest common subsequence (LCS) between generated and reference summaries.

## Overview

ROUGE-L is a critical metric in text summarization evaluation that differs from traditional n-gram-based metrics by focusing on the longest common subsequence, which better captures word order and fluency. This implementation includes:

- **ROUGE-L**: Basic LCS-based scoring for single-sentence or short summaries
- **ROUGE-LSum**: Extended version that handles multi-sentence summaries by computing LCS scores per sentence
- **Text preprocessing pipeline**: Robust preprocessing for accurate tokenization and normalization
- **LLM integration**: Summary generation using Google's Gemini API
- **Validation**: Comparison with the official `rouge-score` library

## Key Features

### 1. Text Preprocessing Pipeline

A comprehensive preprocessing system that handles:
- **Contraction expansion**: Converts contractions (e.g., "can't" → "cannot")
- **Special character handling**: Removes URLs, emails, and normalizes punctuation
- **Number conversion**: Converts numeric strings to words using `num2words`
- **Tokenization**: Uses NLTK's word tokenizer with fallback mechanisms
- **Normalization**: Case normalization and Porter stemming for word variations
- **Error handling**: Robust fallback mechanisms for edge cases

### 2. ROUGE-L Implementation

**LCS Table Computation**:
- Dynamic programming approach to compute the longest common subsequence
- Efficient O(m×n) time complexity for sequences of length m and n
- Returns a 2D table for traceback and length calculation

**ROUGE-L Score Calculation**:
- **Precision**: LCS length divided by prediction length
- **Recall**: LCS length divided by reference length
- **F1 Score**: Harmonic mean with configurable beta parameter (default: 1.2)

### 3. ROUGE-LSum Implementation

Extended version for multi-sentence summaries:
- **Sentence splitting**: Detects sentence boundaries using punctuation markers
- **Per-sentence LCS**: Computes maximum LCS for each reference sentence against all prediction sentences
- **Aggregated scoring**: Combines sentence-level LCS scores for overall precision, recall, and F1

### 4. LLM Integration

Summary generation using Google's Gemini API:
- Secure API key management via Google Colab userdata
- Rate limiting and retry logic for robust API calls
- Error handling for API failures and rate limit errors
- Configurable model selection (default: `gemini-2.5-flash`)

### 5. Validation and Testing

Comprehensive testing framework:
- Integration with official `rouge-score` library
- Side-by-side comparison of custom vs. official scores
- Analysis of score differences (target: < 5% variance)
- Evaluation on CNN/DailyMail dataset

## Technical Implementation

### Core Algorithms

**Longest Common Subsequence (LCS)**:
```python
def get_lcs_table(ref_tokens, pred_tokens):
    # Dynamic programming table construction
    # Returns LCS length at table[m, n]
```

**ROUGE-L Scoring**:
```python
def compute_rouge_l(reference, prediction, beta=1.2):
    # Returns {'precision': float, 'recall': float, 'f1': float}
```

**ROUGE-LSum Scoring**:
```python
def compute_rouge_lsum(reference, prediction, beta=1.2):
    # Sentence-level LCS aggregation
    # Returns {'precision': float, 'recall': float, 'f1': float}
```

### Dependencies

- `datasets>=3.1.0`: Hugging Face datasets library
- `numpy>=1.17`: Numerical computations
- `nltk>=3.6.3`: Natural language tokenization
- `num2words`: Number-to-word conversion
- `rouge-score`: Official ROUGE library for validation
- `google-genai`: Gemini API client (optional, for summary generation)

## Dataset

Uses the **CNN/DailyMail** dataset:
- Standard benchmark for summarization evaluation
- Contains news articles with human-written summaries
- Version 3.0.0 with train/validation/test splits
- Each example includes article text and reference summary

## Usage

### Basic ROUGE-L Evaluation

```python
from rouge_l_implementation import TextPreprocessor, compute_rouge_l

preprocessor = TextPreprocessor()

# Preprocess reference and prediction
ref_tokens = preprocessor.preprocess(reference_text)
pred_tokens = preprocessor.preprocess(prediction_text)

# Compute ROUGE-L scores
scores = compute_rouge_l(ref_tokens, pred_tokens)
print(f"Precision: {scores['precision']:.4f}")
print(f"Recall: {scores['recall']:.4f}")
print(f"F1: {scores['f1']:.4f}")
```

### ROUGE-LSum Evaluation

```python
from rouge_l_implementation import compute_rouge_lsum

scores = compute_rouge_lsum(ref_tokens, pred_tokens)
```

### Generate Summaries with Gemini API

```python
from rouge_l_implementation import get_summary

summary = get_summary(article_text, model="gemini-2.5-flash")
```

## Results and Validation

The implementation achieves:
- **High accuracy**: Custom scores match official `rouge-score` library within acceptable variance (< 5%)
- **Robust preprocessing**: Handles edge cases including special characters, numbers, and contractions
- **Efficient computation**: LCS table computation optimized for large texts
- **Reliable API integration**: Robust error handling and rate limiting for LLM API calls

### Comparison with Official Library

The custom implementation is validated against the official `rouge-score` library:
- ROUGE-L scores show < 5% difference on average
- ROUGE-LSum scores align closely with official implementation
- Preprocessing differences account for minor variations

## Key Insights

1. **LCS vs. N-grams**: LCS-based metrics better capture word order and fluency compared to n-gram overlap metrics.

2. **Preprocessing Importance**: Careful text normalization (stemming, contraction expansion) significantly impacts score accuracy.

3. **Sentence-level Aggregation**: ROUGE-LSum's per-sentence approach provides more nuanced evaluation for multi-sentence summaries.

4. **Practical Application**: The implementation demonstrates how to evaluate LLM-generated summaries using industry-standard metrics.

## Files

- `rouge_l_implementation.ipynb`: Complete implementation notebook with all code, examples, and validation

## References

- Lin, C. Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries. In Text Summarization Branches Out.
- See, A., Liu, P. J., & Manning, C. D. (2017). Get To The Point: Summarization with Pointer-Generator Networks.
- CNN/DailyMail Dataset: Standard benchmark for abstractive summarization
- Official rouge-score library: https://github.com/google-research/google-research/tree/master/rouge

