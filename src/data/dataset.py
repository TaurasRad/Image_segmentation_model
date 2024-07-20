import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.training.hyper_parameters import LEARNING_RATE, DEVICE, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, IMAGE_HEIGHT, IMAGE_WIDTH, PIN_MEMORY, TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, THRESHOLD

class DATASET(Dataset): # Creation of Dataset class - img_dir: training image directory , mask_dir: training mask directory
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(img_dir)

    def __len__(self): # Length of the dataset
        return len(self.images)

    def __getitem__(self, index):# defines a method for item (image) extraction
        img_path = os.path.join(self.img_dir, self.images[index]) # full path each image
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpeg", ".png"))#full path to each mask

        image = Image.open(img_path).convert("RGB").resize((256, 256), Image.BILINEAR) # RGB images
        mask = Image.open(mask_path).convert("L").resize((256, 256), Image.NEAREST) # mask images conversion to grayscale

        if self.transform is not None: # Checks if transformations are made
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)
        else:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)

        mask = mask.float()
        mask[mask == 255.0] = 1.0  # Preprocess the mask
        return image, mask

def get_loaders(TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR, BATCH_SIZE, train_transform, val_transform, NUM_WORKERS, PIN_MEMORY): # function for dataset and dataloader creation
    train_dataset = DATASET(img_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)
    val_dataset = DATASET(img_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)
    return train_loader, val_loader
