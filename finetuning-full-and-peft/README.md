# Fine-Tuning LLMs for Dialogue Summarization

This project demonstrates fine-tuning a Large Language Model (LLM) for enhanced dialogue summarization using two different approaches: full fine-tuning and Parameter Efficient Fine-Tuning (PEFT). The implementation uses the FLAN-T5 model and evaluates performance improvements using ROUGE metrics.

## Overview

Fine-tuning is the process of adapting a pre-trained language model to a specific task by training it on task-specific data. This project explores two fine-tuning strategies:

1. **Full Fine-Tuning**: Updates all model parameters during training
2. **Parameter Efficient Fine-Tuning (PEFT/LoRA)**: Updates only a small subset of parameters using Low-Rank Adaptation

## What Was Achieved

### 1. Full Fine-Tuning Implementation

- **Model**: FLAN-T5-small (77M parameters)
- **Dataset**: DialogSum dataset (10,000+ dialogues with summaries)
- **Approach**: Fine-tuned all 76.9M parameters (100% trainable)
- **Results**: 
  - Achieved significant improvements over zero-shot baseline
  - ROUGE-1: ~40.4% (vs 22.2% baseline)
  - ROUGE-2: ~17.1% (vs 7.1% baseline)
  - ROUGE-L: ~32.7% (vs 19.2% baseline)

### 2. Parameter Efficient Fine-Tuning (PEFT/LoRA)

- **Approach**: LoRA adapter with rank=8, alpha=16
- **Efficiency**: Only 0.45% of parameters trainable (344K out of 77M)
- **Results**:
  - Near-parity performance with full fine-tuning
  - ROUGE-1: ~39.1% (vs 22.2% baseline)
  - ROUGE-2: ~15.5% (vs 7.1% baseline)
  - ROUGE-L: ~31.4% (vs 19.2% baseline)
  - Slightly lower metrics (~1.3-1.6 percentage points) compared to full fine-tuning

### Key Findings

- **Performance**: Both fine-tuning approaches significantly outperform zero-shot inference
- **Efficiency Trade-off**: PEFT achieves ~97% of full fine-tuning performance while training only 0.45% of parameters
- **Storage**: PEFT adapters are much smaller (MBs vs GBs), making them ideal for multi-tenant deployments
- **Qualitative Analysis**: Both models show improved relevance, coherence, and coverage compared to the baseline

## Project Structure

- `finetuning_dialogue_summarization.ipynb`: Main implementation notebook
- `dialogue-summary-training-results.csv`: Evaluation results on test dataset

## Implementation Details

### Dataset Preprocessing
- Converted dialogue-summary pairs into instruction-following format
- Tokenized inputs (max length: 512) and targets (max length: 128)
- Applied dataset subsampling for efficient training

### Training Configuration
- **Full Fine-Tuning**: 10 epochs, learning rate 1e-4, batch size 8
- **PEFT**: 10 epochs, learning rate 1e-3, batch size 16
- Used Seq2SeqTrainer with appropriate data collators
- Training tracked with Weights & Biases (wandb)

### Evaluation
- **Qualitative**: Human evaluation comparing summaries across models
- **Quantitative**: ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum)
- Compared zero-shot baseline, full fine-tuned, and PEFT models

## Technologies Used

- **Transformers**: Hugging Face transformers library
- **PEFT**: Parameter Efficient Fine-Tuning library
- **Datasets**: Hugging Face datasets
- **Evaluation**: ROUGE metric via evaluate library
- **Model**: FLAN-T5-small from Google

## Usage

The notebook `finetuning_dialogue_summarization.ipynb` contains the complete implementation, including:
1. Dataset and model loading
2. Zero-shot baseline evaluation
3. Full fine-tuning pipeline
4. PEFT/LoRA fine-tuning pipeline
5. Comprehensive evaluation and comparison

## Results Summary

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Trainable Params |
|-------|---------|---------|---------|------------------|
| Zero-shot Baseline | 22.2% | 7.1% | 19.2% | 0% |
| Full Fine-tuned | 40.4% | 17.1% | 32.7% | 100% |
| PEFT/LoRA | 39.1% | 15.5% | 31.4% | 0.45% |

The results demonstrate that PEFT provides an excellent balance between performance and efficiency, making it suitable for scenarios where computational resources or storage are constrained.

