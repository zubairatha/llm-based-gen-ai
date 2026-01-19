# Approximate Nearest Neighbor Search with HNSW

This project implements and analyzes an **Approximate Nearest Neighbors (ANN)** solution using the **Hierarchical Navigable Small World (HNSW)** algorithm. HNSW is a state-of-the-art graph-based indexing structure that enables efficient similarity search in high-dimensional vector spaces.

## Overview

Approximate Nearest Neighbor search is a fundamental problem in machine learning and information retrieval, particularly important for applications like:
- Semantic search and retrieval systems
- Recommendation engines
- Image and video similarity search
- Natural language processing embeddings

While exact nearest neighbor search requires examining all vectors (O(n) complexity), ANN algorithms like HNSW provide sub-linear search times with high accuracy, making them practical for large-scale applications.

## What is HNSW?

HNSW (Hierarchical Navigable Small World) builds a multi-layered graph structure where:
- **Bottom layer (Layer 0)**: Contains all data points, densely connected
- **Higher layers**: Contain progressively fewer nodes, forming a "small world" network
- **Search strategy**: Starts at the top layer and greedily navigates downward, refining the search at each level

This hierarchical approach allows the algorithm to quickly narrow down the search space while maintaining high accuracy.

## Implementation

The implementation includes:

### Core Functions

1. **`construct_HNSW(vectors, m_neighbors)`**
   - Builds a hierarchical graph structure from input vectors
   - Each node is assigned a maximum layer level using an exponential distribution
   - Connects each node to its `m_neighbors` nearest neighbors in each layer
   - Returns a list of NetworkX graphs, one per layer

2. **`search_HNSW(graph_layers, query)`**
   - Performs approximate nearest neighbor search using a greedy layer-wise strategy
   - Starts at the entry point in the top layer
   - Greedily moves to closer neighbors within each layer
   - Descends to lower layers when no improvement is found
   - Returns the approximate nearest neighbor index and the search path taken

### Evaluation

The implementation includes comprehensive evaluation:

- **Performance comparison**: Compares ANN search time against brute force search
- **Accuracy metrics**: Measures Accuracy@1 (whether ANN finds the exact nearest neighbor)
- **Visualization**: Plots the search path through the graph layers
- **Parameter analysis**: Tests different values of `m_neighbors` (2, 4, 8) to analyze their impact
- **Real-world testing**: Evaluates on GloVe word embeddings (Wikipedia dataset)

## Results

### Synthetic Dataset (100 vectors, 2D)
- Achieved 100% accuracy (ANN finds the exact nearest neighbor)
- Search times are competitive with brute force for small datasets
- Visualization shows efficient navigation through the graph structure

### Parameter Analysis
Testing with different `m_neighbors` values shows:
- Higher `m` values increase connectivity but may slightly increase search time
- All tested values (2, 4, 8) maintain high accuracy

### Real-World Embeddings
The implementation successfully handles high-dimensional embeddings (100D GloVe vectors) with 20,000 data points, demonstrating scalability to real-world applications.

## Requirements

- Python 3.x
- numpy
- networkx
- matplotlib

## Usage

The main implementation is in `hnsw_implementation.ipynb`. The notebook includes:
- Complete HNSW construction and search implementation
- Evaluation on synthetic 2D data
- Performance analysis with different parameters
- Real-world testing on GloVe embeddings

## Key Features

- **Efficient search**: Sub-linear search complexity for large datasets
- **High accuracy**: Achieves exact nearest neighbor results in many cases
- **Visualization**: Clear visualizations of the graph structure and search paths
- **Scalable**: Works with both low-dimensional synthetic data and high-dimensional real embeddings

