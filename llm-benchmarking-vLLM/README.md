# LLM Inference Benchmarking with vLLM

This project demonstrates benchmarking Large Language Model (LLM) inference performance using vLLM, a high-performance inference engine. The implementation focuses on understanding how different parameters affect inference throughput and provides comprehensive analysis of performance characteristics.

## Overview

vLLM is a high-performance inference engine designed for efficient LLM serving. This project benchmarks inference performance using the OPT-125M model to understand the impact of various parameters on throughput, including batch size, sequence length, and request concurrency.

## What Was Achieved

### 1. Synthetic Request Generation

- **Implementation**: Created a robust function to generate synthetic requests with exact token lengths
- **Features**:
  - Handles tokenization edge cases across different tokenizers
  - Validates exact input/output token lengths
  - Supports flexible request generation for benchmarking

### 2. Benchmarking Infrastructure

- **Core Function**: Implemented `run_benchmark()` using vLLM's LLM engine
- **Features**:
  - Configurable GPU memory utilization
  - Support for tensor parallelism
  - Warm-up runs to minimize initialization overhead
  - Comprehensive metrics collection (throughput, tokens processed, elapsed time)

### 3. Comprehensive Performance Studies

#### Batch Size Experiments
- **Tested**: Batch sizes from 1 to 32
- **Findings**: 
  - Throughput increased almost linearly from ~12 tok/s (batch=1) to ~595 tok/s (batch=32)
  - Diminishing returns observed at batch sizes 16-32
  - Indicates GPU underutilization at small batches

#### Sequence Length Experiments
- **Tested**: Sequence lengths from 32 to 512 tokens
- **Findings**:
  - Optimal throughput at medium lengths (~128 tokens): ~1413 tok/s
  - Short sequences (32 tokens): ~146 tok/s (overhead dominates)
  - Long sequences (512 tokens): ~895 tok/s (prefill cost increases)
  - Sweet spot balances overhead amortization and KV-cache pressure

#### Request Concurrency Experiments
- **Tested**: 10 to 500 concurrent requests
- **Findings**:
  - Non-monotonic scaling pattern
  - Peak throughput: ~2732 tok/s at 500 requests
  - Performance dip at 200 requests (scheduler batching dynamics)
  - Higher concurrency enables better GPU saturation

### Key Insights

1. **Batch Processing**: Larger batches significantly improve throughput by better utilizing GPU resources, with diminishing returns at higher batch sizes

2. **Sequence Length Trade-offs**: 
   - Short sequences suffer from overhead costs
   - Medium lengths provide optimal balance
   - Very long sequences increase prefill costs and memory pressure

3. **Concurrency Effects**: 
   - More concurrent requests enable better GPU utilization
   - Scheduler dynamics can cause non-monotonic behavior
   - Optimal throughput requires careful tuning of concurrency levels

4. **Practical Recommendations**:
   - **Maximum Throughput**: Use larger batches, moderate sequence lengths, and high concurrency
   - **Lower Latency**: Reduce batch size and sequence lengths, accepting lower throughput
   - **Resource Efficiency**: Monitor GPU memory utilization and adjust `max_num_batched_tokens` accordingly

## Project Structure

- `llm_benchmarking_vllm.ipynb`: Main implementation notebook with all experiments

## Implementation Details

### Environment Setup
- **Hardware**: NVIDIA T4 GPU (tested on Google Colab)
- **Frameworks**: PyTorch, vLLM, Transformers
- **Precision**: FP16 (via `dtype="auto"`)
- **GPU Memory**: 90% utilization target

### Model Configuration
- **Model**: `facebook/opt-125m` (125M parameters)
- **Tokenizer**: Hugging Face AutoTokenizer
- **Sampling**: Greedy decoding (temperature=0.0)

### Benchmarking Methodology
- **Warm-up**: Single untimed request before each measurement
- **Timing**: Generation phase only (excludes model loading)
- **Metrics**: 
  - Primary: Generated tokens per second
  - Secondary: Total (input+output) tokens per second
- **Scheduler Settings**: 
  - Batch size sweep: `max_num_batched_tokens=2048`
  - Length/request sweeps: `max_num_batched_tokens=4096`

### Experimental Configurations

**Batch Size Sweep:**
- Input length: 128 tokens
- Output length: 64 tokens
- Batch sizes: [1, 2, 4, 8, 16, 32]

**Sequence Length Sweep:**
- Number of requests: 64
- Sequence lengths: [32, 64, 128, 256, 512] (input = output)

**Request Count Sweep:**
- Input length: 128 tokens
- Output length: 64 tokens
- Request counts: [10, 50, 100, 200, 500]

## Results Summary

| Experiment Type | Parameter Range | Throughput Range | Optimal Point |
|----------------|-----------------|------------------|---------------|
| Batch Size | 1-32 | 12-595 tok/s | Batch=32 (~595 tok/s) |
| Sequence Length | 32-512 tokens | 146-1413 tok/s | 128 tokens (~1413 tok/s) |
| Request Count | 10-500 | 214-2732 tok/s | 500 requests (~2732 tok/s) |

## Technologies Used

- **vLLM**: High-performance LLM inference engine
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Visualization libraries

## Usage

The notebook `llm_benchmarking_vllm.ipynb` contains:
1. Environment setup and library installation
2. Synthetic request generation implementation
3. Benchmarking function implementation
4. Three comprehensive experimental studies
5. Visualization of results
6. Detailed analysis and insights

## Key Takeaways

- vLLM provides efficient batching and scheduling for LLM inference
- Throughput scales well with batch size up to hardware limits
- Sequence length has a significant impact on performance, with optimal ranges
- Concurrency management is crucial for maximizing GPU utilization
- Performance tuning requires balancing throughput, latency, and resource constraints

This benchmarking framework can be extended to test larger models, different hardware configurations, and various inference strategies.

