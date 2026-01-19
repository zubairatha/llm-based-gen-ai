# LLM-Based Generative AI Projects

This repository contains a collection of projects I've worked on while exploring the field of Large Language Models (LLMs) and generative AI. Each project represents hands-on learning experiences covering various aspects of modern AI systems, from foundational algorithms to advanced techniques and practical applications.

These projects were developed as part of my learning journey in understanding how LLMs work, how to optimize them, and how to build real-world applications around them. I plan to continue adding more projects as I explore new areas and techniques in this rapidly evolving field.

## Projects

### 1. [ANN-HNSW](ann-hnsw/)
Implementation and analysis of Approximate Nearest Neighbor (ANN) search using the Hierarchical Navigable Small World (HNSW) algorithm. Explores efficient similarity search in high-dimensional vector spaces with performance comparisons and real-world evaluations on GloVe embeddings.

### 2. [Semantic Search](semantic-search/)
A comprehensive semantic search engine designed for Retrieval-Augmented Generation (RAG) systems. Implements multiple retrieval strategies including similarity search, Maximum Marginal Relevance (MMR), and hybrid search, with evaluation metrics and an interactive web interface.

### 3. [Post-Training Quantization](post-training-quantization/)
Implementation of post-training quantization techniques to optimize neural network inference by converting floating-point models to fixed-point representations. Analyzes weight, activation, and bias quantization strategies with accuracy impact evaluation on CIFAR-10.

### 4. [Fine-Tuning Full and PEFT](finetuning-full-and-peft/)
Comparison of full fine-tuning versus Parameter Efficient Fine-Tuning (PEFT/LoRA) for dialogue summarization using FLAN-T5. Demonstrates how PEFT achieves near-parity performance while training only 0.45% of parameters, making it ideal for resource-constrained scenarios.

### 5. [LLM Benchmarking with vLLM](llm-benchmarking-vLLM/)
Comprehensive benchmarking of LLM inference performance using vLLM, analyzing the impact of batch size, sequence length, and request concurrency on throughput. Provides insights into optimizing inference for production deployments.

### 6. [ReAct Agent](react-agent/)
Implementation of a ReAct (Reasoning + Acting) agent that combines reasoning and tool use to handle complex queries. Features custom Search, Compare, and Analyze tools orchestrated through LangChain with a Streamlit interface for interactive exploration.

### 7. [TinyTimeMixer](tiny-time-mixer/)
Exploration of TinyTimeMixer for multivariate time series forecasting, comparing zero-shot and few-shot fine-tuning approaches. Evaluates the TTM-1024-96 model on electricity transformer temperature data with various loss functions and channel-specific forecasting.

### 8. [RLHF-DPO](rlhf-dpo/)
Direct Preference Optimization (DPO) implementation for fine-tuning LLMs on preference data without requiring a separate reward model. Fine-tunes Qwen2.5-14B-Instruct for YouTube title generation using LoRA and 4-bit quantization, achieving efficient preference alignment.

### 9. [LLM Summarization Evaluation](llm-summarization-eval/)
From-scratch implementation of ROUGE-L and ROUGE-LSum metrics for evaluating text summarization quality. Includes comprehensive text preprocessing, LCS-based scoring, and integration with LLM-generated summaries using Google's Gemini API.

### 10. [MCP Server](mcp-server/)
A complete Model Context Protocol (MCP) server implementation that enables AI assistants to interact with local data files through natural language queries. Provides tools for analyzing CSV and Parquet files with Claude Desktop integration and programmatic client access.

## Getting Started

Each project folder contains its own README with detailed documentation, requirements, and usage instructions. Navigate to individual project directories to explore specific implementations and learn more about each technique.

## Technologies

These projects utilize a wide range of modern AI and ML technologies including:
- **Frameworks**: PyTorch, Transformers, LangChain, vLLM
- **Models**: FLAN-T5, Qwen2.5, Gemini, TinyTimeMixer
- **Techniques**: LoRA, Quantization, DPO, HNSW, MMR
- **Tools**: Streamlit, ChromaDB, Hugging Face, Unsloth

