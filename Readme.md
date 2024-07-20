# Segmentation Model

## Project Structure

<pre>
Segmentation_Model
│
├── data               
│   ├── train           <- Folder containing training images.
│   ├── train_mask      <- Folder containing training masks.
│   ├── test            <- Folder containing testing images.
│   ├── test_mask       <- Folder containing testing masks.
│
├── notebooks           <- Folder containing Jupyter notebooks for easy access.
│   ├── EXP_gpu.ipynb   <- Notebook for modular access to the code
|
├── reports             <- Folder for storing plots of model performance metrics. 
│   ├── predictions     <- Folder for storing model predicted binary masks.
│   ├── true_images     <- Folder for storing true test masks.
│   
│
├── src                
│   ├── data           <- Code for downloading, preprocessing, and loading the datasets.
│   │   ├── dataset.py <- Script for dataset handling.
│   │   
│   ├── models         <- Code for defining the models.
│   │   ├── unet.py    <- UNet model definition.
│   │   
│   ├── training       <- Training and evaluation code.
│       ├── train.py       <- Training script.
│       ├── utils.py       <- Utility functions.
│       ├── threshold_optimization.py <- Script for threshold optimization.
│       ├── hyper_parameters.py <- Script for defining hyper parameters
│      
│
├── requirements.txt   <- File for installing dependencies.
└── README.md          <- Project documentation.
</pre>

## Usage

1. **Creating a new environment for the project:**
    ```sh
    conda create --name segmentation_model python=3.9
    ```

2. **Activating the environment:**
    ```sh
    conda activate segmentation_model
    ```

3. **Setting up the environment:**
    Navigate to the folder containing `requirements.txt` and run this code to establish the environment:
    ```sh
    pip install -r requirements.txt
    ```

4. **Defining the hyperparameters:**
    Modify the `hyper_parameters.py` script as needed.

5. **Training the model:**
    Navigate to the folder containing the `train.py` script and run:
    ```sh
    python train.py
    ```

6. **Optimizing the threshold:**
    Navigate to the folder containing the `threshold_optimization.py` script and run:
    ```sh
    python threshold_optimization.py
    ```

7. **Evaluating the models:**
    Review the evaluation results in the `reports` folder.

## Data

1. **Example training and test data:**
   Upload train and test datasets - resolution can be any, true images must be jpeg and binary masks png

2. **Loading the datasets:**
   Place the datasets in the `data` folder and split into `train`, `train_mask`, `test`, `test_mask` subfolders.

3. **Data split:**
   - Training subset: ~75% (519 images)
   - Testing subset: ~25% (158 images)
   - `train` and `test` subfolders contain unaugmented images of documents with backgrounds.
   - `train_mask` and `test_mask` subfolders contain binary image masks of those documents.

4. **Preparing data for training:**
   Data is loaded and prepared using the `dataset.py` script.

5. **Data augmentations:**
   Defined and executed in the `train.py` main training function.
   - Training dataset: resized to 256x256, randomly rotated, horizontally and vertically flipped, converted to tensors, and normalized.
   - Test dataset: resized to 256x256, converted to tensors, and normalized.
   - Note: The model performs better on test data than on training data, even at early epochs, because the test data is less augmented and thus simpler to predict.

## Training

1. **Hyperparameters:**
   Defined in `hyper_parameters.py`. Defaults:
   ```python
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
   THRESHOLD = 0.65  # Threshold chosen via threshold_optimization.py

2. **Model Details:**
   
   The model used has a UNET network structure and employs the binary cross-entropy loss function. The final layer uses a sigmoid function for binary classification, with 0 representing the background and 1 representing the document. Network layers are given below:

   | Layer Type         | Input Channels | Output Channels | Kernel Size | Stride | Padding | Additional Information                   |
   |--------------------|----------------|-----------------|-------------|--------|---------|------------------------------------------|
   | DoubleConv Block 1 | 3              | 64              | (3, 3)      | 1      | 1       | Two Conv layers, BatchNorm, ReLU         |
   | MaxPool2d          |                |                 | (2, 2)      | 2      | 0       |                                          |
   | DoubleConv Block 2 | 64             | 128             | (3, 3)      | 1      | 1       | Two Conv layers, BatchNorm, ReLU         |
   | MaxPool2d          |                |                 | (2, 2)      | 2      | 0       |                                          |
   | DoubleConv Block 3 | 128            | 256             | (3, 3)      | 1      | 1       | Two Conv layers, BatchNorm, ReLU         |
   | MaxPool2d          |                |                 | (2, 2)      | 2      | 0       |                                          |
   | DoubleConv Block 4 | 256            | 512             | (3, 3)      | 1      | 1       | Two Conv layers, BatchNorm, ReLU         |
   | MaxPool2d          |                |                 | (2, 2)      | 2      | 0       |                                          |
   | Bottleneck         | 512            | 1024            | (3, 3)      | 1      | 1       | Two Conv layers, BatchNorm, ReLU         |
   | ConvTranspose2d    | 1024           | 512             | (2, 2)      | 2      | 0       |                                          |
   | DoubleConv Block 1 | 1024           | 512             | (3, 3)      | 1      | 1       | Two Conv layers, BatchNorm, ReLU         |
   | ConvTranspose2d    | 512            | 256             | (2, 2)      | 2      | 0       |                                          |
   | DoubleConv Block 2 | 512            | 256             | (3, 3)      | 1      | 1       | Two Conv layers, BatchNorm, ReLU         |
   | ConvTranspose2d    | 256            | 128             | (2, 2)      | 2      | 0       |                                          |
   | DoubleConv Block 3 | 256            | 128             | (3, 3)      | 1      | 1       | Two Conv layers, BatchNorm, ReLU         |
   | ConvTranspose2d    | 128            | 64              | (2, 2)      | 2      | 0       |                                          |
   | DoubleConv Block 4 | 128            | 64              | (3, 3)      | 1      | 1       | Two Conv layers, BatchNorm, ReLU         |
   | Final Conv2d       | 64             | 1               | (1, 1)      | 1      | 0       |                                          |

3. **Evaluation Metrics:**

   The metrics chosen for the evaluation of the model were pixel accuracy, dice score, precision, recall, specificity, and Intersection over Union (IoU). IoU was chosen for threshold optimization over Dice because it is less sensitive to small objects.

4. **Model Performance:**

   After 45 epochs of training, the saved model's performance metrics on the given test dataset with default hyperparameters are:
   
   - Pixel accuracy: 96.14% (9,954,582 correct pixels out of 10,354,688 total)
   - Dice score: 0.93
   - Precision: 0.97
   - Recall: 0.90
   - Specificity: 0.99
   - IoU: 0.87

   Full training and testing metric data are stored as plots in the `reports` subfolder after training.
