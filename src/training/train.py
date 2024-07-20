#IMPORTAI
#added parent directory to sys. path to allow easy imports from other files
import sys
sys.path.append('../../')
#Standard libraries import 
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from torchvision import transforms
# Custom modules and hyper_parameters
from src.data.dataset import get_loaders
from src.models.unet import UNET
from src.training.utils import save_predictions_as_imgs, check_accuracy
from src.training.hyper_parameters import LEARNING_RATE, DEVICE, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, IMAGE_HEIGHT, IMAGE_WIDTH, PIN_MEMORY, TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, THRESHOLD

# Seed - for repeatability !
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# The main training function 
def train_fn(loader, model, optimizer, loss_fn, scaler=None):
    loop = tqdm(loader) # progress bar
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.float().to(DEVICE)

        # Forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())
        #model makes predictions, calculates the error, adjusts the weights to reduce the error, and then updates its parameters
    pass

##T
def main():
    train_transform = transforms.Compose([ #training Image augmentations - resizing,random rotations, horizontal/vertical flips, normalization
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.RandomRotation(35),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ])

    val_transforms = transforms.Compose([#Validation Image augmentations - resizing, normalization
        transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
    ])

    model = UNET(in_channels=3, out_channels=1).to(DEVICE) #Loading model described in unet.py in models
    loss_fn = nn.BCEWithLogitsLoss() # Binary cross entropy loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # ADAM optimizer
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1) # Scheduler while not inherently needed, because ADAM has an auto LR adaptor, was implemented
    
    train_loader, val_loader = get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, train_transform, val_transforms, NUM_WORKERS, PIN_MEMORY) # Calls data loaders as described in dataset.py
    #Validation Metrics
    accuracies = [] # The proportion of correctly predicted pixels out of the total pixels.
    dice_scores = []#A measure of overlap between the predicted and actual masks, with higher values indicating better performance.
    precisions = []#The proportion of true positive predictions out of all positive predictions made by the model.
    recalls = []#The proportion of true positive predictions out of all actual positives in the dataset.
    specificities = []#The proportion of true negative predictions out of all actual negatives in the dataset.
    ious=[]# (Intersection over Union): A metric that calculates the overlap between the predicted and actual masks, divided by the union of both masks, with higher values indicating better performance.
    #Training Metrics
    t_accuracies = []
    t_dice_scores = []
    t_precisions = []
    t_recalls = []
    t_specificities = []
    t_ious=[]

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn)# Calls the training function with saving relevant metrics

        accuracy, dice, precision, recall, specificity,iou = check_accuracy(val_loader, model,threshold=THRESHOLD ,print_metrics=True) # metric saving function is defined in utils.py
        accuracies.append(accuracy)
        dice_scores.append(dice)
        precisions.append(precision)
        recalls.append(recall)
        specificities.append(specificity)
        ious.append(iou)
        #
        t_accuracy, t_dice, t_precision, t_recall, t_specificity,t_iou = check_accuracy(train_loader,model,threshold=THRESHOLD,print_metrics=False)
        t_accuracies.append(t_accuracy)
        t_dice_scores.append(t_dice)
        t_precisions.append(t_precision)
        t_recalls.append(t_recall)
        t_specificities.append(t_specificity)
        t_ious.append(t_iou)
        #
        scheduler.step()
        # Checkpoint  was added but was commented as not essential
        #checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
        #save_checkpoint(checkpoint)
        save_predictions_as_imgs(val_loader, model,threshold=THRESHOLD,  pred_folder="../../reports/predictions/", true_folder="../../reports/true_images/")# Function for saving predicted binary mask and true examples as described in utils.py

    torch.save(model.state_dict(), "Image_segmentation_model.pth")# Saving the model
    print("Model saved as Image_segmentation_model.pth")
    #Code for saving the model in onnx format ( Warning message - should be disregarded)
    dummy_input = torch.randn(1, 3, IMAGE_HEIGHT, IMAGE_WIDTH).to(DEVICE)
    torch.onnx.export(model, dummy_input, "Image_segmentation_model.onnx", 
                      input_names=['input'], output_names=['output'], 
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    print("Image_segmentation_model.onnx")

    #All metrics saved in dataframe
    metrics_df = pd.DataFrame({
        'epoch': list(range(1, NUM_EPOCHS + 1)),
        'accuracy': accuracies,
        'dice': dice_scores,
        'precision': precisions,
        'recall': recalls,
        'specificity': specificities,
        'IoU': ious,
        't_accuracy': t_accuracies,
        't_dice': t_dice_scores,
        't_precision': t_precisions,
        't_recall': t_recalls,
        't_specificity': t_specificities,
        't_IoU': t_ious
    })
# Plotted graphs and saved in reports 
    def plot_metrics(df, metric, folder):
        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df[metric], marker='o', label=f'Validation {metric}')
        plt.plot(df['epoch'], df[f't_{metric}'], marker='x', label=f'Training {metric}')
        plt.title(f'{metric.capitalize()} over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{folder}/{metric}.png")
        plt.close()

    metrics = ['accuracy', 'dice', 'precision', 'recall', 'specificity', 'IoU']
    for metric in metrics:
        plot_metrics(metrics_df, metric, "../../reports")
    
if __name__ == "__main__":
    main()
