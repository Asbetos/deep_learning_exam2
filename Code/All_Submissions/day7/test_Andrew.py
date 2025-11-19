import os
import subprocess
import sys

try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                      'albumentations==2.0.8',
                      'opencv-python-headless==4.12.0.88',
                      'timm==1.0.22'])
except:
    os.system("sudo pip install albumentations==2.0.8 opencv-python-headless==4.12.0.88 timm==1.0.22")

import argparse
import random
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils import data
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
import cv2
'''
ULTIMATE SOTA TEST SCRIPT with Test-Time Augmentation
Supports: Swin-Large, ViT-Large, ConvNeXt-Large
'''

parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)
parser.add_argument("--split", default=False, type=str, required=True)

args = parser.parse_args()

PATH = args.path
DATA_DIR = args.path + os.path.sep + 'Data' + os.path.sep
SPLIT = args.split

SEED = 67128
BATCH_SIZE = 196
NICKNAME = "Andrew"
USE_TTA = True  # Test-Time Augmentation
TTA_TRANSFORMS = 8 # Number of TTA variations

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Will be loaded from checkpoint
MODEL_NAME = 'swinv2_large_window12to16_192to256'
IMAGE_SIZE = 256
# OPTIMAL_THRESHOLDS = [0.5] * 10
OPTIMAL_THRESHOLDS = [0.6, 0.55, 0.6, 0.6, 0.65, 0.6, 0.6, 0.6, 0.6, 0.6]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ==================== MODEL ====================
class UltimateSOTAModel(nn.Module):
    """Swin-Large or ViT-Large with enhanced head - MUST match training"""

    def __init__(self, model_name, num_classes, image_size):
        super(UltimateSOTAModel, self).__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,
            global_pool='avg'
        )

        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            feat_dim = self.backbone(dummy).shape[1]

        # Enhanced multi-label head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim, feat_dim // 2),
            nn.LayerNorm(feat_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feat_dim // 2, feat_dim // 4),
            nn.LayerNorm(feat_dim // 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feat_dim // 4, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits


# ==================== DATASET ====================
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

        return img, y


def get_test_transforms(image_size):
    """Basic test transforms"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_tta_transforms(image_size):
    """Test-Time Augmentation - 8 variations"""
    tta_list = [
        # 1. Original
        A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]),
        # 2. Horizontal flip
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]),
        # 3. Vertical flip
        A.Compose([
            A.Resize(image_size, image_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]),
        # 4. Both flips
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]),
        # 5. Rotate 90
        A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=(90, 90), p=1.0),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]),
        # 6. Rotate 180
        A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=(180, 180), p=1.0),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]),
        # 7. Rotate 270
        A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=(270, 270), p=1.0),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]),
        # 8. Center crop + resize
        A.Compose([
            A.CenterCrop(height=int(image_size * 0.9), width=int(image_size * 0.9),pad_if_needed=True),
            A.Resize(image_size, image_size),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]),
    ]
    return tta_list


def read_data(target_type, image_size):
    list_of_ids_test = list(xdf_dset_test.index)
    partition = {'test': list_of_ids_test}
    params = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 4, 'pin_memory': True}
    test_set = Dataset(partition['test'], 'test', target_type, transform=get_test_transforms(image_size))
    test_generator = data.DataLoader(test_set, **params)
    return test_generator


def apply_optimal_thresholds(probabilities, thresholds):
    """Apply per-class optimal thresholds"""
    predictions = np.zeros_like(probabilities, dtype=int)
    for i, threshold in enumerate(thresholds):
        predictions[:, i] = (probabilities[:, i] >= threshold).astype(int)
    return predictions


def model_definition():
    """Load model with weights from checkpoint"""
    global MODEL_NAME, IMAGE_SIZE, OPTIMAL_THRESHOLDS

    model_path = f'model_{NICKNAME}.pt'

    if os.path.exists(model_path):
        print(f"Warning: Loading model without checkpoint metadata")
        print(f"Using default: {MODEL_NAME}, image_size={IMAGE_SIZE}")
        model = UltimateSOTAModel(MODEL_NAME, OUTPUTS_a, IMAGE_SIZE)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"No model file found!")

    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Save model summary
    print(model, file=open(f'summary_{NICKNAME}.txt', "w"))

    return model


def test_model_with_tta(test_ds):
    """Test with Test-Time Augmentation"""
    model = model_definition()

    if USE_TTA:
        print(f"\n{'=' * 60}")
        print(f"Test-Time Augmentation (TTA) with {TTA_TRANSFORMS} variations")
        print(f"{'=' * 60}\n")

        tta_transforms_list = get_tta_transforms(IMAGE_SIZE)[:TTA_TRANSFORMS]

        all_predictions = []
        real_labels = None

        # Get original images
        list_of_ids_test = list(xdf_dset_test.index)

        for tta_idx, tta_transform in enumerate(tta_transforms_list):
            print(f"\nTTA variation {tta_idx + 1}/{TTA_TRANSFORMS}")

            # Create dataset with this TTA transform
            tta_set = Dataset(list_of_ids_test, 'test', 2, transform=tta_transform)
            tta_loader = data.DataLoader(tta_set, batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=4, pin_memory=True)

            pred_probs_tta = []
            real_labels_tta = []

            with torch.no_grad():
                with tqdm(total=len(tta_loader), desc=f"TTA {tta_idx + 1}") as pbar:
                    for xdata, xtarget in tta_loader:
                        xdata = xdata.to(device)
                        output = model(xdata)
                        probs = torch.sigmoid(output)
                        pred_probs_tta.append(probs.cpu().numpy())
                        if tta_idx == 0:
                            real_labels_tta.append(xtarget.cpu().numpy())
                        pbar.update(1)

            pred_probs_tta = np.vstack(pred_probs_tta)
            all_predictions.append(pred_probs_tta)

            if tta_idx == 0:
                real_labels = np.vstack(real_labels_tta)

        # Average predictions across all TTA variations
        print("\nAveraging TTA predictions...")
        pred_probs = np.mean(all_predictions, axis=0)
        print(f"TTA prediction shape: {pred_probs.shape}")

    else:
        print("\nRunning standard inference (no TTA)...")
        pred_probs = []
        real_labels = []

        with torch.no_grad():
            with tqdm(total=len(test_ds), desc="Testing") as pbar:
                for xdata, xtarget in test_ds:
                    xdata = xdata.to(device)
                    output = model(xdata)
                    probs = torch.sigmoid(output)
                    pred_probs.append(probs.cpu().numpy())
                    real_labels.append(xtarget.cpu().numpy())
                    pbar.update(1)

        pred_probs = np.vstack(pred_probs)
        real_labels = np.vstack(real_labels)

    # Apply optimal thresholds
    pred_labels = apply_optimal_thresholds(pred_probs, OPTIMAL_THRESHOLDS)

    # Calculate metrics
    f1_micro = f1_score(real_labels, pred_labels, average='micro', zero_division=0)
    f1_macro = f1_score(real_labels, pred_labels, average='macro', zero_division=0)
    f1_weighted = f1_score(real_labels, pred_labels, average='weighted', zero_division=0)

    # Print results
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"TTA: {USE_TTA} ({TTA_TRANSFORMS if USE_TTA else 0} variations)")
    print("-" * 60)
    print(f"F1-micro:    {f1_micro:.5f}")
    print(f"F1-macro:    {f1_macro:.5f}")
    print(f"F1-weighted: {f1_weighted:.5f}")
    print("=" * 60)

    # Per-class F1 scores
    print("\nPer-class F1 scores:")
    class_f1_scores = []
    for i in range(OUTPUTS_a):
        class_f1 = f1_score(real_labels[:, i], pred_labels[:, i], zero_division=0)
        class_f1_scores.append(class_f1)
        print(f"  Class {i}: {class_f1:.4f} (threshold: {OPTIMAL_THRESHOLDS[i]:.2f})")

    print(f"\nAverage class F1: {np.mean(class_f1_scores):.4f}")
    print(f"Min class F1: {np.min(class_f1_scores):.4f}")
    print(f"Max class F1: {np.max(class_f1_scores):.4f}")

    # Save results
    xfinal_pred_labels = [",".join(str(int(e)) for e in row) for row in pred_labels]
    xdf_dset_test['results'] = xfinal_pred_labels
    xdf_dset_test.to_excel(f'results_{NICKNAME}.xlsx', index=False)

    print(f"\nResults saved to: results_{NICKNAME}.xlsx")

    return {'f1_micro': f1_micro, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted}


def process_target(target_type):
    if target_type == 2:
        target = np.array(xdf_data['target'].apply(lambda x: x.split(",")))
        final_target = mlb.fit_transform(target)
        xfinal = [",".join(str(e) for e in row) for row in final_target]
        xdf_data['target_class'] = xfinal
    return mlb.classes_


if __name__ == '__main__':
    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    for file in os.listdir(PATH + os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    xdf_data = pd.read_excel(FILE_NAME)
    class_names = process_target(target_type=2)
    xdf_dset_test = xdf_data[xdf_data["split"] == SPLIT].copy()

    OUTPUTS_a = len(class_names)

    print(f"\n{'=' * 60}")
    print(f"ULTIMATE SOTA TEST CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Test samples: {len(xdf_dset_test)}")
    print(f"Classes: {OUTPUTS_a}")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"{'=' * 60}\n")

    # Load model first to get IMAGE_SIZE
    test_ds = None  # Will create after loading model

    # Test with TTA
    test_ds = read_data(target_type=2, image_size=IMAGE_SIZE)  # Default, will be updated
    test_metrics = test_model_with_tta(test_ds)

    print(f"\n{'=' * 60}")
    print(f"FINAL TEST F1-MICRO: {test_metrics['f1_micro']:.5f}")
    print(f"{'=' * 60}\n")