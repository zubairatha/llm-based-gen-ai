# Semantic Search for RAG Systems

This project implements and analyzes a **semantic search engine** that serves as the retrieval component for **Retrieval-Augmented Generation (RAG)** systems. The implementation demonstrates multiple search strategies, evaluates their performance, and includes a web interface for interactive exploration.

## Overview

Semantic search enables finding relevant documents based on meaning rather than exact keyword matches. This is crucial for RAG systems, which retrieve contextually relevant information to augment language model responses. This project explores different retrieval strategies and their trade-offs in accuracy, diversity, and efficiency.

## Key Features

### Core Implementation

1. **Document Processing Pipeline**
   - PDF document loading using LangChain's PyPDFLoader
   - Intelligent text chunking with RecursiveCharacterTextSplitter
   - Embedding generation using pre-trained models (sentence-transformers)
   - Vector store creation and persistence with ChromaDB

2. **Multiple Search Methods**
   - **Similarity Search**: Standard vector similarity search using cosine similarity
   - **Maximum Marginal Relevance (MMR)**: Balances relevance with diversity to reduce redundant results
   - **Hybrid Search**: Combines semantic search with keyword-based BM25 scoring
   - **Search with Scores**: Returns similarity scores alongside results

3. **Comprehensive Evaluation**
   - Performance analysis across different chunk sizes (500, 1000, 1500 characters)
   - Latency measurements for various k values (1, 5, 10)
   - Diversity analysis comparing similarity search vs MMR
   - Visualization of embedding spaces using dimensionality reduction (PCA)

4. **Web Interface**
   - Interactive Streamlit application for exploring search results
   - Support for multiple embedding models
   - Real-time comparison of different search strategies

## What is Maximum Marginal Relevance (MMR)?

MMR is a retrieval strategy that balances relevance with diversity. It iteratively selects documents that are both:
- Relevant to the query
- Different from already-selected documents

The MMR formula: `MMR = λ × Sim(doc, query) − (1 − λ) × max(Sim(doc, selected))`

Where λ controls the trade-off between relevance (λ=1) and diversity (λ=0). This helps avoid the "cluster problem" where all retrieved documents cover the same aspect of a query.

## Implementation Details

### Document Processing

- **Chunking Strategy**: RecursiveCharacterTextSplitter with configurable chunk size and overlap
- **Metadata Preservation**: Maintains source information and start indices for traceability
- **Embedding Model**: Uses sentence-transformers models (default: all-MiniLM-L6-v2)

### Search Methods

1. **Similarity Search**: Standard cosine similarity-based retrieval
2. **MMR Search**: Implements diversity-aware retrieval using LangChain's built-in MMR method
3. **Hybrid Search**: Combines semantic embeddings with BM25 keyword matching

### Evaluation Metrics

- **Latency**: Search time for different k values
- **Diversity**: Mean pairwise similarity and unique source coverage
- **Coverage**: Number of unique documents retrieved across queries
- **Accuracy**: Relevance of retrieved results

## Results

The evaluation demonstrates:

- **Chunk Size Impact**: Different chunk sizes affect retrieval quality and latency
- **MMR Benefits**: MMR provides better diversity and coverage compared to standard similarity search
- **Performance Trade-offs**: Larger k values improve coverage but increase latency
- **Embedding Visualization**: PCA reveals clustering patterns in the document space

## Files

- `semantic_search_rag.ipynb`: Main implementation notebook with all experiments and analysis
- `app.py`: Streamlit web interface for similarity and MMR search
- `app_embeddings.py`: Extended web interface with multiple embedding model support
- `data_new_pdfs/`: Collection of diverse PDF documents for testing

## Requirements

- Python 3.x
- numpy, pandas
- sentence-transformers
- scikit-learn
- matplotlib, seaborn
- langchain-community
- langchain-text-splitters
- langchain-chroma
- chromadb
- pypdf
- streamlit (for web interface)
- rank-bm25 (for hybrid search)

## Usage

### Running the Notebook

1. Install required dependencies
2. Place PDF documents in the `data/` directory
3. Run the notebook cells sequentially to:
   - Process documents and create vector stores
   - Perform searches with different methods
   - Evaluate performance metrics
   - Generate visualizations

### Running the Web Interface

```bash
streamlit run app.py
```

Or for the extended version with multiple embedding models:

```bash
streamlit run app_embeddings.py
```

## Key Insights

- **Chunk Size Matters**: Optimal chunk size depends on document characteristics and query types
- **MMR Improves Diversity**: MMR search retrieves more diverse results, covering multiple aspects of queries
- **Hybrid Search Benefits**: Combining semantic and keyword search can improve retrieval quality
- **Latency Considerations**: Larger k values and chunk sizes increase search time but may improve coverage

## Applications

This semantic search implementation is suitable for:

- RAG systems requiring context retrieval
- Document question-answering systems
- Information retrieval applications
- Content recommendation systems
- Knowledge base search interfaces

