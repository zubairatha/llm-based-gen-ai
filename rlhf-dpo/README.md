# Direct Preference Optimization (DPO) for YouTube Title Generation

This project implements **Direct Preference Optimization (DPO)** to fine-tune a large language model for generating engaging YouTube titles. DPO is a modern approach to training language models from preference data without requiring an explicit reward model or reinforcement learning loop.

## Overview

Direct Preference Optimization is a method that directly optimizes a language model's policy using preference pairs (chosen vs. rejected responses). Unlike traditional RLHF (Reinforcement Learning from Human Feedback), DPO eliminates the need for a separate reward model, making it more efficient and easier to implement.

This implementation fine-tunes the **Qwen2.5-14B-Instruct** model using preference data from a YouTube titles dataset, where each example contains:
- A **prompt**: The video idea and instructions for title generation
- A **chosen** title: The preferred/higher-quality title
- A **rejected** title: A less preferred title

## Key Features

- **Efficient Fine-tuning**: Uses LoRA (Low-Rank Adaptation) to train only ~1.57% of model parameters
- **Memory Optimization**: 4-bit quantization with bitsandbytes for reduced VRAM usage
- **Accelerated Training**: Leverages Unsloth for 2x faster training
- **Preference Learning**: Directly optimizes model to prefer chosen titles over rejected ones
- **Comprehensive Evaluation**: Before/after comparison showing DPO's impact on title quality

## Technical Implementation

### Environment Setup

The implementation uses:
- **Unsloth**: Fast fine-tuning library with optimized training loops
- **TRL (Transformers Reinforcement Learning)**: DPO trainer implementation
- **PEFT**: Parameter-efficient fine-tuning with LoRA
- **bitsandbytes**: 4-bit quantization support
- **xformers**: Flash Attention for speed improvements

### Model Architecture

- **Base Model**: Qwen2.5-14B-Instruct (quantized to 4-bit)
- **LoRA Configuration**:
  - Rank (r): 32
  - Alpha: 64
  - Target modules: Attention and MLP projections (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
- **Trainable Parameters**: ~137M out of ~8.7B total (1.57%)

### Dataset

Uses the `"EliasHossain/youtube-titles-dpo"` dataset from Hugging Face:
- **Training split**: 1,026 examples
- **Validation split**: 114 examples
- **Format**: Chat-style messages with prompt, chosen, and rejected fields

### Training Configuration

- **Epochs**: 3
- **Effective batch size**: 16 (per-device: 4, gradient accumulation: 4)
- **Learning rate**: 5e-5 with cosine scheduling
- **DPO beta**: 0.1 (controls preference sharpness)
- **Precision**: bfloat16 (bf16) on A100 GPUs
- **Optimizer**: paged_adamw_8bit (memory-efficient)

## Results

### Training Metrics

After 3 epochs of DPO training:
- **Training loss**: ~0.42
- **Validation loss**: ~0.56
- **Preference accuracy**: ~74% (model correctly prefers chosen over rejected titles)
- **Reward margin**: ~1.85 (average difference between chosen and rejected rewards)

### Model Performance

The DPO fine-tuned model demonstrates:
- **Improved alignment**: Higher probability assigned to preferred titles
- **Better instruction following**: More consistent adherence to "title only" format
- **Enhanced engagement**: Titles show more targeted, explanatory framing aligned with human preferences
- **Maintained quality**: Preserves base model's fluency while improving preference alignment

### Qualitative Observations

- Base model produces fluent but sometimes generic titles
- DPO model shows consistent shift toward FAQ/explanatory framing
- Better alignment with human-preferred title styles
- Improved specificity and engagement-oriented wording

## Files

- `dpo_youtube_titles.ipynb`: Complete implementation notebook with all code, training, and evaluation

## Usage

1. Install dependencies:
   ```bash
   pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
   pip install xformers trl peft accelerate bitsandbytes triton transformers datasets pandas
   ```

2. Load the dataset:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("EliasHossain/youtube-titles-dpo")
   ```

3. Follow the notebook sections:
   - Environment setup and model loading
   - LoRA configuration
   - DPO training
   - Model evaluation and comparison

## Key Insights

1. **DPO Effectiveness**: The method successfully reshapes model preferences, achieving ~74% accuracy in preferring chosen over rejected titles.

2. **Efficiency**: LoRA + 4-bit quantization enables fine-tuning a 14B parameter model on a single A100 GPU with reasonable batch sizes.

3. **Preference Learning**: The model learns subtle stylistic preferences (FAQ framing, explanatory wording) without losing base capabilities.

4. **Practical Application**: DPO provides a lightweight alignment step that improves model outputs for specific use cases like YouTube title generation.

## References

- Direct Preference Optimization (DPO) paper
- Unsloth library for accelerated fine-tuning
- TRL library for DPO implementation
- Qwen2.5 model family

