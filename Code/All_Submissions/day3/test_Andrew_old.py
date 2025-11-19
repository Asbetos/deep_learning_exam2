import argparse
import random
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils import data
from tqdm import tqdm

import os
import sys
import subprocess

def install_dependencies():
    """
    Try multiple installation methods with fallback
    """
    packages = ['albumentations', 'opencv-python-headless', 'timm']

    # Method 1:
    try:
        print("Attempting Method 1: sudo pip install (system-wide)...")
        result = os.system("sudo pip install albumentations opencv-python-headless timm")

        if result == 0:  # Success
            print("Method 1 successful: Packages installed system-wide")
            return True
        else:
            print("Method 1 failed, trying fallback...")
            raise Exception("sudo pip install failed")

    except Exception as e:
        print(f"Method 1 error: {e}")
        print("Attempting Method 2: User-space installation (no sudo)...")

        # Method 2: Fallback to user-space installation
        try:
            for package in packages:
                print(f"  Installing {package}...")
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install',
                    '--user', package, '--quiet'
                ])
            print("Method 2 successful: Packages installed in user space")
            return True

        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            print("ERROR: Could not install required packages")
            return False


# Install packages with fallback
install_dependencies()

# Now import - try importing and provide helpful error if still fails
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import timm
    import cv2

    print("All packages imported successfully\n")
except ImportError as e:
    print(f"\nIMPORT ERROR: {e}")
    print("Please install packages manually:")
    print("sudo pip install albumentations opencv-python-headless timm")
    sys.exit(1)

'''
IMPROVED TEST SCRIPT - Matches enhanced training
'''

parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)
parser.add_argument("--split", default=False, type=str, required=True)

args = parser.parse_args()

PATH = args.path
DATA_DIR = args.path + os.path.sep + 'Data' + os.path.sep
SPLIT = args.split

SEED = 67128
BATCH_SIZE = 64
USE_PRETRAINED = True
CHANNELS = 3
IMAGE_SIZE = 224
NICKNAME = "Andrew"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# THRESHOLD = 0.5
OPTIMAL_THRESHOLDS = [0.5, 0.65, 0.6, 0.65, 0.65, 0.65, 0.65, 0.75, 0.65, 0.65]

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ImprovedCNN(nn.Module):
    """Enhanced CNN matching training script"""

    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, (3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, (3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, (3, 3), padding=1)
        self.bn4 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(2, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Dataset(data.Dataset):
    def __init__(self, list_IDs, type_data, target_type, transform=None):
        self.type_data = type_data
        self.list_IDs = list_IDs
        self.target_type = target_type
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]

        if self.type_data == 'train':
            y = xdf_dset.target_class.get(ID)
            if self.target_type == 2:
                y = y.split(",")
        else:
            y = xdf_dset_test.target_class.get(ID)
            if self.target_type == 2:
                y = y.split(",")

        if self.target_type == 2:
            labels_ohe = [int(e) for e in y]
        else:
            labels_ohe = np.zeros(OUTPUTS_a)
            for idx, label in enumerate(range(OUTPUTS_a)):
                if label == y:
                    labels_ohe[idx] = 1

        y = torch.FloatTensor(labels_ohe)

        if self.type_data == 'train':
            file = DATA_DIR + xdf_dset.id.get(ID)
        else:
            file = DATA_DIR + xdf_dset_test.id.get(ID)

        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        else:
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img.astype(np.float32) / 255.0
            img = (img - MEAN) / STD
            img = torch.FloatTensor(img).permute(2, 0, 1)

        return img, y


def get_test_transforms():
    """Test time augmentation transforms"""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def read_data(target_type):
    list_of_ids_test = list(xdf_dset_test.index)

    partition = {
        'test': list_of_ids_test
    }

    params = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 4, 'pin_memory': True}

    test_set = Dataset(partition['test'], 'test', target_type, transform=get_test_transforms())
    test_generator = data.DataLoader(test_set, **params)

    return test_generator


def apply_optimal_thresholds(probabilities, thresholds):
    """
    Apply per-class optimal thresholds to probability predictions

    Args:
        probabilities: numpy array of shape (batch_size, num_classes)
        thresholds: list of thresholds for each class

    Returns:
        predictions: binary predictions array
    """
    predictions = np.zeros_like(probabilities, dtype=int)
    for i, threshold in enumerate(thresholds):
        predictions[:, i] = (probabilities[:, i] >= threshold).astype(int)
    return predictions

def model_definition(pretrained=False):
    """
    Load model architecture and weights with optimal thresholds
    Matches training_script.py model definition
    """

    # Initialize model architecture (must match training_script.py)
    if pretrained:
        print("Using pretrained model architecture...")

        # Try Vision Transformer (as per training_script.py line with vit_base_patch16_224)
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=OUTPUTS_a)
        print("Model: Vision Transformer (vit_base_patch16_224)")
    else:
        print("Using ImprovedCNN architecture...")
        model = ImprovedCNN(OUTPUTS_a)

    # Load from simple model file
    model_path = f'model_{NICKNAME}.pt'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model weights loaded from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = model.to(device)
    model.eval()

    # Save model summary
    print(model, file=open(f'summary_{NICKNAME}.txt', "w"))

    criterion = nn.BCEWithLogitsLoss()

    return model, criterion


def test_model(test_ds, list_of_metrics, list_of_agg, pretrained=False):
    model, criterion = model_definition(pretrained)

    pred_logits, real_labels = [], []

    model.eval()

    test_loss, steps_test = 0, 0

    with torch.no_grad():
        with tqdm(total=len(test_ds), desc="Testing") as pbar:
            for xdata, xtarget in test_ds:
                xdata, xtarget = xdata.to(device), xtarget.to(device)

                output = model(xdata)

                loss = criterion(output, xtarget)
                test_loss += loss.item()
                steps_test += 1

                pbar.update(1)
                pbar.set_postfix_str("Test Loss: {:.5f}".format(test_loss / steps_test))

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(output)
                pred_logits.append(probs.cpu().numpy())
                real_labels.append(xtarget.cpu().numpy())

    pred_probs = np.vstack(pred_logits)
    real_labels = np.vstack(real_labels)

    # Apply threshold
    # pred_labels = (pred_probs >= THRESHOLD).astype(int)
    pred_labels = apply_optimal_thresholds(pred_probs, OPTIMAL_THRESHOLDS)

    # Run the statistics
    test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels, pred_labels)

    avg_test_loss = test_loss / steps_test

    # Print metrics
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Average Test Loss: {avg_test_loss:.5f}")
    for met, dat in test_metrics.items():
        print(f"Test {met}: {dat:.5f}")
    print("=" * 60)

    # Convert predictions to string format
    xfinal_pred_labels = []
    for i in range(len(pred_labels)):
        joined_string = ",".join(str(int(e)) for e in pred_labels[i])
        xfinal_pred_labels.append(joined_string)

    # Save the results
    xdf_dset_test['results'] = xfinal_pred_labels
    xdf_dset_test.to_excel('results_{}.xlsx'.format(NICKNAME), index=False)

    print(f"\nResults saved to: results_{NICKNAME}.xlsx")

    return test_metrics, avg_test_loss


def metrics_func(metrics, aggregates, y_true, y_pred):
    def f1_score_metric(y_true, y_pred, type):
        return f1_score(y_true, y_pred, average=type, zero_division=0)

    def cohen_kappa_metric(y_true, y_pred):
        return cohen_kappa_score(y_true.flatten(), y_pred.flatten())

    def accuracy_metric(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    def matthews_metric(y_true, y_pred):
        return matthews_corrcoef(y_true.flatten(), y_pred.flatten())

    def hamming_metric(y_true, y_pred):
        return hamming_loss(y_true, y_pred)

    xcont = 0
    xsum = 0
    res_dict = {}

    for xm in metrics:
        if xm == 'f1_micro':
            xmet = f1_score_metric(y_true, y_pred, 'micro')
        elif xm == 'f1_macro':
            xmet = f1_score_metric(y_true, y_pred, 'macro')
        elif xm == 'f1_weighted':
            xmet = f1_score_metric(y_true, y_pred, 'weighted')
        elif xm == 'coh':
            xmet = cohen_kappa_metric(y_true, y_pred)
        elif xm == 'acc':
            xmet = accuracy_metric(y_true, y_pred)
        elif xm == 'mat':
            xmet = matthews_metric(y_true, y_pred)
        elif xm == 'hlm':
            xmet = hamming_metric(y_true, y_pred)
        else:
            xmet = 0

        res_dict[xm] = xmet
        xsum += xmet
        xcont += 1

    if 'sum' in aggregates:
        res_dict['sum'] = xsum
    if 'avg' in aggregates and xcont > 0:
        res_dict['avg'] = xsum / xcont

    return res_dict


def process_target(target_type):
    if target_type == 2:
        target = np.array(xdf_data['target'].apply(lambda x: x.split(",")))
        final_target = mlb.fit_transform(target)
        xfinal = []
        if len(final_target) == 0:
            xerror = 'Could not process Multilabel'
        else:
            class_names = mlb.classes_
            for i in range(len(final_target)):
                joined_string = ",".join(str(e) for e in final_target[i])
                xfinal.append(joined_string)
            xdf_data['target_class'] = xfinal

    if target_type == 1:
        xtarget = list(np.array(xdf_data['target'].unique()))
        le = LabelEncoder()
        le.fit(xtarget)
        final_target = le.transform(np.array(xdf_data['target']))
        class_names = (xtarget)
        xdf_data['target_class'] = final_target

    return class_names


if __name__ == '__main__':
    # Set random seeds for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    for file in os.listdir(PATH + os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    xdf_data = pd.read_excel(FILE_NAME)

    class_names = process_target(target_type=2)

    xdf_dset_test = xdf_data[xdf_data["split"] == SPLIT].copy()

    test_ds = read_data(target_type=2)

    OUTPUTS_a = len(class_names)

    list_of_metrics = ['f1_micro', 'f1_macro', 'f1_weighted']
    list_of_agg = ['avg']

    test_model(test_ds, list_of_metrics, list_of_agg, pretrained=USE_PRETRAINED)