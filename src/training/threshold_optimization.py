#Imports 
import sys
sys.path.append('../../')
import numpy as np
import torch.nn as nn
from src.data.dataset import get_loaders
from src.models.unet import UNET
from src.training.utils import check_accuracy
from src.training.train import train_fn
from torchvision import transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from src.training.hyper_parameters import LEARNING_RATE, DEVICE, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, IMAGE_HEIGHT, IMAGE_WIDTH, PIN_MEMORY, TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, THRESHOLD

# Image transformations as in dataset.py loaded here for indenpendance 
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.RandomRotation(35),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
])

val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
])

# Defined the range of thresholds
thresholds = np.arange(0.0, 1.05, 0.05)
best_threshold = 0.0
best_iou = 0.0

# Loop over thresholds
for threshold in thresholds:
    THRESHOLD = threshold
    print(f"Training with threshold: {threshold}")
    
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, 
        BATCH_SIZE, train_transform, val_transforms, NUM_WORKERS, PIN_MEMORY
    )
    
    # Training and evaluating the model using IoU
    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)
        scheduler.step()
    #    
    accuracy, dice, precision, recall, specificity, iou = check_accuracy(val_loader, model, threshold=THRESHOLD, device=DEVICE)
    print(f"Threshold: {threshold}, IoU: {iou}")
    # Update the best threshold if the current IoU is higher
    if iou > best_iou:
        best_iou = iou
        best_threshold = threshold
print(f"Best threshold: {best_threshold}, Best IoU: {best_iou}")
