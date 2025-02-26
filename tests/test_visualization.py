import numpy as np
from utils.visualization import plot_training_curves

# Dummy data for testing
train_losses = np.random.rand(10)
val_losses = np.random.rand(10)
train_accs = np.random.rand(10)
val_accs = np.random.rand(10)

# Plot training curves
plot_training_curves(train_losses, val_losses, train_accs, val_accs)
