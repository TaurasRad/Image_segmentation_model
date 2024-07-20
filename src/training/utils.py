#Standard 
import os
import torch
import torchvision
from src.training.hyper_parameters import LEARNING_RATE, DEVICE, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, IMAGE_HEIGHT, IMAGE_WIDTH, PIN_MEMORY, TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, THRESHOLD

# Function for saving a weights checkpoint during long trainings 
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)
#Loading a checkpoint
def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
# function for saving binary mask predictions as images
def save_predictions_as_imgs(loader, model,threshold, pred_folder="../reports/predictions/", true_folder="../reports/true_images/", ):
    if not os.path.exists(pred_folder):
        os.makedirs(pred_folder)
    if not os.path.exists(true_folder):
        os.makedirs(true_folder)
        
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        with torch.no_grad():
            preds = torch.sigmoid(model(x)) # Last layer is a sigmoid function that discerns 0 - background, 1 - Document
            preds = (preds > threshold).float() # Chosen threshold
        torchvision.utils.save_image(preds, f"{pred_folder}/pred_{idx}.png")
        torchvision.utils.save_image(y, f"{true_folder}/true_{idx}.png")
    model.train()

def check_accuracy(loader, model, threshold, print_metrics=True):# Function for tracking and saving model performance metrics
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    iou_score = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            preds = torch.sigmoid(model(x))
            preds = (preds > threshold).float()
            # Formulas for metrics
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)
            intersection = (preds * y).sum()
            union = preds.sum() + y.sum() - intersection
            iou_score += (intersection / (union + 1e-8))
            # Definition of TP,FP,FN,TN
            true_positives += ((preds == 1) & (y == 1)).sum()
            false_positives += ((preds == 1) & (y == 0)).sum()
            false_negatives += ((preds == 0) & (y == 1)).sum()
            true_negatives += ((preds == 0) & (y == 0)).sum()
    # Formulas for metrics
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    specificity = true_negatives / (true_negatives + false_positives + 1e-8)
    accuracy = num_correct / num_pixels * 100
    dice = dice_score / len(loader)
    iou = iou_score / len(loader)

    if print_metrics:
        print(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}") # Pixel accuracy
        print(f"Dice score: {dice}") # Dice coefficient
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"IoU: {iou:.4f}")

    model.train()
    return accuracy.item(), dice.item(), precision.item(), recall.item(), specificity.item(), iou.item()
