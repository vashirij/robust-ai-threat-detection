# Model Checkpoints

This directory contains saved model checkpoints from training.

## Contents

- `baseline_model.pt` - Best performing baseline model with validation-based training
- Additional model checkpoints will be saved here during training

## Loading Saved Models

To load a saved model:

```python
from cicids_mlp_adv import MLP, ModelCheckpoint
import torch

# Initialize model with same architecture
model = MLP(input_dim)  # Use appropriate input dimension

# Load checkpoint
checkpoint_manager = ModelCheckpoint()
epoch, loss = checkpoint_manager.load('checkpoints/baseline_model.pt', model)

print(f"Loaded model from epoch {epoch} with loss {loss:.4f}")
```

## Model Information

- **Architecture**: Multi-layer perceptron (MLP)
- **Input Features**: 70 (after preprocessing)
- **Output**: Binary classification (benign vs. malicious)
- **Training**: Includes validation-based early stopping and L2 regularization

**Note**: Model checkpoints are excluded from version control due to file size. They will be generated when you run the training code.
