# Post-Training Quantization for Neural Networks

This project implements and analyzes **post-training quantization** techniques to optimize neural network inference by converting floating-point models to fixed-point representations. The implementation focuses on quantizing weights, activations, and biases of a convolutional neural network trained on CIFAR-10.

## Overview

Neural networks are typically trained using 32-bit floating-point arithmetic, which provides high precision but is computationally expensive in hardware. Post-training quantization reduces model size and improves inference speed by converting weights and activations to lower-precision integer representations (e.g., 8-bit integers) without requiring retraining.

This project demonstrates:
- **Weight Quantization**: Converting 32-bit floating-point weights to 8-bit integers
- **Activation Quantization**: Quantizing intermediate layer activations
- **Bias Quantization**: Quantizing biases to 32-bit integers
- **Analysis**: Statistical analysis of weight and activation distributions
- **Evaluation**: Measuring accuracy impact of quantization

## Key Concepts

### Quantization Basics

Quantization maps floating-point values to integers using a scaling factor:

```
quantized_value = round(float_value × scale)
dequantized_value = quantized_value / scale
```

The scale factor determines the precision and range:
- **Symmetric Quantization**: Uses zero-point = 0, suitable for zero-centered distributions
- **Per-Tensor Scaling**: Single scale factor per layer (simpler)
- **Per-Channel Scaling**: Separate scale per output channel (more accurate)

### Scaling Strategies

1. **Max-Abs Scaling**: `scale = 127 / max(|values|)`
   - Uses full range but sensitive to outliers
   
2. **3-Sigma Scaling**: `scale = 127 / (3 × std)`
   - More robust to outliers, uses statistical properties
   
3. **Percentile Scaling**: Uses high percentile (e.g., 99.9th) instead of max
   - Balances robustness and range coverage

## Implementation

### Weight Quantization

- Converts weights from float32 to int8 (-128 to 127)
- Uses symmetric quantization with zero-point = 0
- Implements robust scaling (3-sigma or percentile-based)
- Validates quantized values are integers within int8 range

### Activation Quantization

- Profiles activation distributions across network layers
- Implements per-layer activation quantization
- Accounts for ReLU activations (non-negative outputs)
- Maintains quantization through forward pass

### Bias Quantization

- Quantizes biases to 32-bit integers
- Accounts for weight and activation scaling factors
- Ensures numerical stability in quantized operations

## Results

The implementation demonstrates:

- **Minimal Accuracy Loss**: Weight quantization causes <1% accuracy drop
- **Distribution Analysis**: Different layers show varying quantization sensitivity
- **Robust Scaling**: 3-sigma scaling outperforms max-abs for layers with outliers
- **Layer Sensitivity**: Convolutional layers are more sensitive than fully connected layers

## Key Insights

1. **Weight Distributions**: 
   - Near-zero means support symmetric quantization
   - Variability decreases in middle layers, increases at classifier head
   - Convolutional layers show heavier tails than fully connected layers

2. **Activation Patterns**:
   - ReLU activations are non-negative, requiring asymmetric quantization
   - Activation ranges vary significantly across layers
   - Early layers show wider distributions than later layers

3. **Quantization Impact**:
   - Layers with heavy-tailed distributions benefit from robust scaling
   - Per-tensor scaling is simpler but per-channel can improve accuracy
   - Proper scaling strategy is crucial for maintaining accuracy

## Files

- `quantization_implementation.ipynb`: Complete implementation with analysis and evaluation

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- pandas

## Usage

1. Run the notebook cells sequentially to:
   - Load and train a CNN on CIFAR-10
   - Analyze weight and activation distributions
   - Implement quantization functions
   - Evaluate quantized model accuracy

2. The notebook includes:
   - Statistical analysis of weight/activation distributions
   - Visualization of distributions across layers
   - Quantization implementation with validation
   - Accuracy measurements before and after quantization

## Applications

Post-training quantization is essential for:

- **Edge Devices**: Deploying models on resource-constrained devices
- **Mobile Applications**: Reducing model size and power consumption
- **Hardware Acceleration**: Enabling efficient integer arithmetic operations
- **Production Systems**: Improving inference speed and reducing memory usage

## Trade-offs

- **Accuracy vs. Efficiency**: Lower precision improves speed but may reduce accuracy
- **Simplicity vs. Accuracy**: Per-tensor scaling is simpler but per-channel can be more accurate
- **Robustness vs. Range**: Robust scaling (3-sigma) avoids outliers but may waste range

## References

- Quantization techniques for neural network inference
- Hardware-efficient deep learning
- Model compression and optimization

