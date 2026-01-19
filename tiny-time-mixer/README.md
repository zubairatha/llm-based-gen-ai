# TinyTimeMixer for Time Series Forecasting

This project demonstrates the usage of a pre-trained **TinyTimeMixer (TTM)** model for multivariate time series forecasting tasks. The implementation explores both zero-shot and few-shot fine-tuning approaches using the TTM-1024-96 model architecture.

## About TinyTimeMixer

TinyTimeMixer is a state-of-the-art time series forecasting model that can handle long context sequences and produce accurate multi-step forecasts. The TTM-1024-96 model used in this project:
- Takes an input of **1024 time points** (context length)
- Can forecast up to **96 time points** (forecast length) into the future
- Supports multivariate time series with multiple channels

For details about the model architecture, refer to the [TTM paper](https://arxiv.org/pdf/2401.03955.pdf).

## Features

The implementation covers several key aspects of time series forecasting:

### 1. **Zero-Shot Evaluation**
- Direct evaluation of the pre-trained model on test data without any fine-tuning
- Demonstrates the model's generalization capabilities on unseen datasets
- Includes evaluation with truncated forecast lengths (e.g., 24-step forecasts)

### 2. **Few-Shot Fine-Tuning**
- Quick adaptation using only 5% or 10% of training data
- Freezes the backbone and fine-tunes only the prediction head
- Compares performance with different data fractions

### 3. **Loss Function Experiments**
- Evaluates the impact of different loss functions (default vs. MAE)
- Compares Mean Squared Error (MSE) and Mean Absolute Error (MAE) performance

### 4. **Channel-Specific Forecasting**
- Demonstrates selective channel forecasting
- Shows how to forecast only specific channels (e.g., channels 0 and 2)

## Project Structure

- `tiny_time_mixer.ipynb` - Complete implementation notebook with all experiments

## Dataset

The implementation uses the **ETTh1** (Electricity Transformer Temperature) dataset, which contains:
- Multiple channels: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
- Time series data with date timestamps
- Standard train/validation/test splits

The dataset is available from the [ETDataset repository](https://github.com/zhouhaoyi/ETDataset/tree/main).

## Model Repository

Pre-trained TTM models are fetched from the [Hugging Face TTM Model Repository](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2):

- **TTM-R2 Models**: Used in this implementation
  - For 1024-96 model: `TTM_MODEL_REVISION="1024-96-r2"`
  - Model path: `ibm-granite/granite-timeseries-ttm-r2`

## Installation

The notebook uses the `tsfm_public` library:

```bash
pip install "tsfm_public[notebooks] @ git+https://github.com/ibm-granite/granite-tsfm.git@v0.2.18"
```

Additional dependencies include:
- PyTorch
- Transformers
- Pandas
- Matplotlib
- NumPy

## Usage

### Environment Setup

1. Set the seed for reproducibility
2. Configure data paths and model parameters
3. Set forecasting parameters (context_length, forecast_length)

### Data Processing

The notebook uses `TimeSeriesPreprocessor` (TSP) for:
- Loading and preprocessing time series data
- Creating train/validation/test splits
- Scaling and normalizing the data
- Preparing data loaders for training

### Zero-Shot Evaluation

```python
# Load pre-trained model
zeroshot_model = TinyTimeMixerForPrediction.from_pretrained(
    TTM_MODEL_PATH, 
    revision=TTM_MODEL_REVISION
)

# Evaluate on test dataset
zeroshot_output = zeroshot_trainer.evaluate(test_dataset)
```

### Few-Shot Fine-Tuning

```python
# Create subset of training data (5% or 10%)
train_dataset_subset = Subset(train_dataset, train_indices)

# Load model and freeze backbone
model = TinyTimeMixerForPrediction.from_pretrained(...)
for param in model.backbone.parameters():
    param.requires_grad = False

# Fine-tune for 1 epoch
trainer.train()
```

## Key Findings

The implementation demonstrates several important insights:

1. **Zero-shot performance**: The pre-trained TTM model achieves strong performance (~0.3586 eval loss) without any fine-tuning on the target dataset.

2. **Shorter horizons are easier**: Truncating the forecast length from 96 to 24 steps significantly improves performance (0.307 vs 0.359).

3. **Few-shot fine-tuning**: With limited data (5-10%) and shallow training (1 epoch, frozen backbone), few-shot fine-tuning doesn't always improve over zero-shot performance, suggesting the need for:
   - More training epochs
   - Unfreezing some backbone layers
   - Better hyperparameter tuning

4. **Loss function impact**: Changing from default loss to MAE doesn't provide significant improvements in this shallow fine-tuning setup.

5. **Model flexibility**: The model supports:
   - Adjustable forecast lengths via `prediction_filter_length`
   - Selective channel forecasting via `prediction_channel_indices`

## Results Summary

- **Zero-shot (96-step)**: ~0.3586 eval loss
- **Zero-shot (24-step)**: ~0.3074 eval loss
- **Few-shot 5%**: ~0.3613 eval loss
- **Few-shot 10%**: ~0.3668 eval loss
- **Few-shot 5% (MAE loss)**: ~0.3664 eval loss

## References

- TinyTimeMixer Paper: [arXiv:2401.03955](https://arxiv.org/pdf/2401.03955.pdf)
- TTM Model Repository: [Hugging Face - granite-timeseries-ttm-r2](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)
- TTM Getting Started: [GitHub Notebook](https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/ttm_getting_started.ipynb)
- ETDataset: [GitHub Repository](https://github.com/zhouhaoyi/ETDataset)

