
import torch
# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 5
NUM_EPOCHS = 45
NUM_WORKERS = 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
TRAIN_IMG_DIR = "../../data/train"
TRAIN_MASK_DIR = "../../data/train_mask"
VAL_IMG_DIR = "../../data/test"
VAL_MASK_DIR = "../../data/test_mask"
THRESHOLD = 0.65 # Threshold chosen via Threshold_optimization.py
