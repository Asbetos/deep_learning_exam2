import random
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils import data
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import os
import sys
import subprocess

# Install required packages
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                       'albumentations==2.0.8',
                       'opencv-python-headless==4.12.0.88',
                       'timm==1.0.22'])

import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from torch.utils.tensorboard import SummaryWriter
import datetime

'''
ULTIMATE SOTA Multi-Label Classification - Target: 90%+ F1-micro
Using Swin Transformer Large or ViT Large

Model Options (ranked by performance):
1. swin_large_patch4_window7_224 - BEST for multi-label (88-92% mAP on benchmarks)
2. swin_large_patch4_window12_384 - Even better with 384px images
3. vit_large_patch16_224 - Excellent alternative
4. convnext_large - Modern CNN, very strong
5. efficientnetv2_l - Fast and accurate

Current Best: Swin-Large with 384px images + heavy augmentation
'''

# Seeds
SEED = 67128
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

OR_PATH = os.getcwd()
os.chdir("../../..")
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
os.chdir(OR_PATH)

# ==================== HYPERPARAMETERS ====================
# Optimized for NVIDIA A10 (24GB VRAM)
n_epoch = 200
BATCH_SIZE = 16 # ← Optimized for A10 with 384px images
GRADIENT_ACCUMULATION_STEPS = 4 # Effective batch size = 64
LR_MAX = 8e-5  # Lower LR for large pretrained models
WEIGHT_DECAY = 1e-4

# Model selection - Choose the best available
IMAGE_SIZE = 256
# MODEL_NAME = 'swin_large_patch4_window7_224'  # Good - works with 224px (can use BATCH_SIZE=48)
# MODEL_NAME = 'mobilenetv3_small_100'
MODEL_NAME = 'swinv2_large_window12to16_192to256'
# MODEL_NAME = 'vit_large_patch16_224'  # Alternative: ViT-Large (can use BATCH_SIZE=40)
# MODEL_NAME = 'convnext_large'  # Alternative: ConvNeXt-Large (can use BATCH_SIZE=48)


print(f"Using model: {MODEL_NAME}")
print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")

USE_MIXUP = True
MIXUP_ALPHA = 0.3  # Lighter mixup for large models
CUTMIX_ALPHA = 0.8
LABEL_SMOOTHING = 0.05  # Light label smoothing

NICKNAME = "Andrew"
mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
SAVE_MODEL = True

# Mixed precision training
USE_AMP = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ==================== MIXUP / CUTMIX ====================
def mixup_data(x, y, alpha=0.3):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(x, y, alpha=0.8):
    """CutMix augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam


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


def get_train_transforms():
    """Heavy augmentation optimized for Transformers"""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3, rotate_limit=45, p=0.6),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=1),
            A.HueSaturationValue(hue_shift_limit=25, sat_shift_limit=40, val_shift_limit=25, p=1),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1),
        ], p=0.6),
        A.OneOf([
            A.GaussNoise(var_limit=(15.0, 60.0), p=1),
            A.GaussianBlur(blur_limit=(5, 9), p=1),
            A.MotionBlur(blur_limit=9, p=1),
        ], p=0.4),
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            A.ElasticTransform(alpha=1, sigma=50, p=1),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.5, p=1),
        ], p=0.3),
        A.CoarseDropout(max_holes=12, max_height=int(IMAGE_SIZE * 0.15),
                        max_width=int(IMAGE_SIZE * 0.15), p=0.4),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Validation transforms"""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def read_data(target_type):
    list_of_ids = list(xdf_dset.index)
    list_of_ids_test = list(xdf_dset_test.index)

    partition = {'train': list_of_ids, 'test': list_of_ids_test}

    params_train = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 4,
                    'pin_memory': True, 'drop_last': True}
    training_set = Dataset(partition['train'], 'train', target_type, transform=get_train_transforms())
    training_generator = data.DataLoader(training_set, **params_train)

    params_test = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 4, 'pin_memory': True}
    test_set = Dataset(partition['test'], 'test', target_type, transform=get_val_transforms())
    test_generator = data.DataLoader(test_set, **params_test)

    return training_generator, test_generator


# ==================== LOSS FUNCTION ====================
class AsymmetricLossOptimized(nn.Module):
    """Asymmetric Loss with Label Smoothing"""

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, label_smoothing=0.05):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.label_smoothing = label_smoothing

    def forward(self, x, y):
        # Label smoothing
        if self.label_smoothing > 0:
            y = y * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.mean()


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixed loss for Mixup/CutMix"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==================== MODEL ====================
class UltimateSOTAModel(nn.Module):
    """Swin-Large or ViT-Large with enhanced head"""

    def __init__(self, model_name, num_classes):
        super(UltimateSOTAModel, self).__init__()

        # Load pretrained model
        self.backbone = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove head
            global_pool='avg'
        )

        # Get feature dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            feat_dim = self.backbone(dummy).shape[1]

        print(f"Feature dimension: {feat_dim}")

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


def model_definition():
    """Initialize model, optimizer, criterion, scheduler"""
    model = UltimateSOTAModel(MODEL_NAME, OUTPUTS_a)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {MODEL_NAME}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # Save model summary
    print(model, file=open(f'summary_{NICKNAME}.txt', "w"))

    # Layerwise learning rate decay (lower layers get lower LR)
    def get_layer_id_for_param(name):
        if 'backbone.patch_embed' in name or 'backbone.pos_embed' in name:
            return 0
        elif 'backbone.layers.0' in name or 'backbone.blocks.0' in name:
            return 1
        elif 'backbone.layers.1' in name or 'backbone.blocks.1' in name or 'backbone.blocks.2' in name:
            return 2
        elif 'backbone.layers.2' in name or 'backbone.blocks.3' in name or 'backbone.blocks.4' in name:
            return 3
        elif 'backbone.layers.3' in name or 'backbone.blocks' in name:
            return 4
        else:
            return 5  # Head

    param_groups = [[] for _ in range(6)]
    for name, param in model.named_parameters():
        layer_id = get_layer_id_for_param(name)
        param_groups[layer_id].append(param)

    # Different LR for different layers
    optimizer = torch.optim.AdamW([
        {'params': param_groups[0], 'lr': LR_MAX * 0.01},  # Embedding
        {'params': param_groups[1], 'lr': LR_MAX * 0.05},  # Early layers
        {'params': param_groups[2], 'lr': LR_MAX * 0.1},  # Mid layers
        {'params': param_groups[3], 'lr': LR_MAX * 0.2},  # Late layers
        {'params': param_groups[4], 'lr': LR_MAX * 0.5},  # Final backbone
        {'params': param_groups[5], 'lr': LR_MAX}  # Head
    ], weight_decay=WEIGHT_DECAY)

    criterion = AsymmetricLossOptimized(
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        label_smoothing=LABEL_SMOOTHING
    )

    return model, optimizer, criterion


class EarlyStopping:
    """Early stopping"""

    def __init__(self, patience=18, min_delta=0.0003, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_score, epoch):
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            print(f"Initial best: {current_score:.4f}")
            return False

        if self.mode == 'max':
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.best_epoch = epoch
                self.counter = 0
                print(f"✓ New best: {current_score:.4f}")
                return False
            else:
                self.counter += 1
                print(
                    f"No improvement {self.counter}/{self.patience} (best: {self.best_score:.4f} @ epoch {self.best_epoch + 1})")

        if self.counter >= self.patience:
            print(f"\nEarly stopping! Best: {self.best_score:.4f} @ epoch {self.best_epoch + 1}")
            self.early_stop = True
            return True

        return False


def find_optimal_threshold(y_true, y_pred_probs):
    """Find optimal threshold per class"""
    best_thresholds = []
    for i in range(y_true.shape[1]):
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.05):
            preds = (y_pred_probs[:, i] > thresh).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        best_thresholds.append(best_thresh)
    return best_thresholds


# ==================== TRAINING ====================
def train_and_test(train_ds, test_ds):
    model, optimizer, criterion = model_definition()

    # OneCycleLR scheduler
    steps_per_epoch = len(train_ds)
    total_steps = n_epoch * steps_per_epoch // GRADIENT_ACCUMULATION_STEPS

    scheduler = OneCycleLR(
        optimizer,
        max_lr=[LR_MAX * 0.01, LR_MAX * 0.05, LR_MAX * 0.1, LR_MAX * 0.2, LR_MAX * 0.5, LR_MAX],
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )

    print(f"Using OneCycleLR (max_lr={LR_MAX})")

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/{NICKNAME}_{timestamp}'
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard: tensorboard --logdir=runs\n")

    early_stopping = EarlyStopping(patience=18, min_delta=0.0003, mode='max')

    met_test_best = 0
    best_thresholds = [0.5] * OUTPUTS_a

    for epoch in range(n_epoch):
        # ========== TRAINING ==========
        model.train()
        train_loss = 0
        steps_train = 0
        pred_logits_train, real_labels_train = [], []

        optimizer.zero_grad()

        with tqdm(total=len(train_ds), desc=f"Epoch {epoch + 1}/{n_epoch} [Train]") as pbar:
            for batch_idx, (xdata, xtarget) in enumerate(train_ds):
                xdata, xtarget = xdata.to(device), xtarget.to(device)

                # Apply Mixup/CutMix randomly
                if USE_MIXUP and np.random.random() > 0.5:
                    if np.random.random() > 0.5:
                        xdata, y_a, y_b, lam = mixup_data(xdata, xtarget, MIXUP_ALPHA)
                    else:
                        xdata, y_a, y_b, lam = cutmix_data(xdata, xtarget, CUTMIX_ALPHA)

                    if USE_AMP:
                        with torch.cuda.amp.autocast():
                            output = model(xdata)
                            loss = mixup_criterion(criterion, output, y_a, y_b, lam)
                    else:
                        output = model(xdata)
                        loss = mixup_criterion(criterion, output, y_a, y_b, lam)
                else:
                    if USE_AMP:
                        with torch.cuda.amp.autocast():
                            output = model(xdata)
                            loss = criterion(output, xtarget)
                    else:
                        output = model(xdata)
                        loss = criterion(output, xtarget)

                # Gradient accumulation
                loss = loss / GRADIENT_ACCUMULATION_STEPS

                if USE_AMP:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    if USE_AMP:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                    optimizer.zero_grad()
                    scheduler.step()

                train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                steps_train += 1

                pred_logits_train.append(torch.sigmoid(output).detach().cpu().numpy())
                real_labels_train.append(xtarget.cpu().numpy())

                pbar.update(1)
                pbar.set_postfix_str(f"Loss: {train_loss / steps_train:.4f}")

        pred_logits_train = np.vstack(pred_logits_train)
        real_labels_train = np.vstack(real_labels_train)

        # ========== VALIDATION ==========
        model.eval()
        test_loss = 0
        steps_test = 0
        pred_logits_test, real_labels_test = [], []

        with torch.no_grad():
            with tqdm(total=len(test_ds), desc=f"Epoch {epoch + 1}/{n_epoch} [Val]") as pbar:
                for xdata, xtarget in test_ds:
                    xdata, xtarget = xdata.to(device), xtarget.to(device)

                    if USE_AMP:
                        with torch.cuda.amp.autocast():
                            output = model(xdata)
                            loss = criterion(output, xtarget)
                    else:
                        output = model(xdata)
                        loss = criterion(output, xtarget)

                    test_loss += loss.item()
                    steps_test += 1

                    pred_logits_test.append(torch.sigmoid(output).cpu().numpy())
                    real_labels_test.append(xtarget.cpu().numpy())

                    pbar.update(1)
                    pbar.set_postfix_str(f"Loss: {test_loss / steps_test:.4f}")

        pred_logits_test = np.vstack(pred_logits_test)
        real_labels_test = np.vstack(real_labels_test)

        # Find optimal thresholds
        if epoch % 3 == 0 or epoch == n_epoch - 1:
            best_thresholds = find_optimal_threshold(real_labels_test, pred_logits_test)
            print(f"\nThresholds: {[f'{t:.2f}' for t in best_thresholds]}")

        # Apply thresholds
        pred_labels_test = np.zeros_like(pred_logits_test)
        for i, thresh in enumerate(best_thresholds):
            pred_labels_test[:, i] = (pred_logits_test[:, i] >= thresh).astype(int)

        # FIX: Convert to int32 to avoid scipy sparse float16 issues
        pred_labels_test = pred_labels_test.astype(np.int32)
        real_labels_test = real_labels_test.astype(np.int32)

        # Metrics
        train_pred_labels = (pred_logits_train >= 0.5).astype(int)
        train_pred_labels = train_pred_labels.astype(np.int32)
        real_labels_train = real_labels_train.astype(np.int32)

        train_f1_micro = f1_score(real_labels_train, train_pred_labels, average='micro', zero_division=0)
        train_f1_macro = f1_score(real_labels_train, train_pred_labels, average='macro', zero_division=0)
        test_f1_micro = f1_score(real_labels_test, pred_labels_test, average='micro', zero_division=0)
        test_f1_macro = f1_score(real_labels_test, pred_labels_test, average='macro', zero_division=0)
        test_f1_weighted = f1_score(real_labels_test, pred_labels_test, average='weighted', zero_division=0)

        avg_train_loss = train_loss / steps_train
        avg_test_loss = test_loss / steps_test
        current_lr = optimizer.param_groups[-1]['lr']  # Head LR

        # Logging
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_test_loss, epoch)
        writer.add_scalar('F1/train_micro', train_f1_micro, epoch)
        writer.add_scalar('F1/val_micro', test_f1_micro, epoch)
        writer.add_scalar('F1/val_macro', test_f1_macro, epoch)
        writer.add_scalar('LR/head', current_lr, epoch)

        # Per-class F1
        for i in range(OUTPUTS_a):
            class_f1 = f1_score(real_labels_test[:, i], pred_labels_test[:, i], zero_division=0)
            writer.add_scalar(f'F1_class/class_{i}', class_f1, epoch)

        # Print
        print(f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_test_loss:.4f}")
        print(
            f"Train F1-micro: {train_f1_micro:.4f} ({train_f1_macro:.4f}) | Val F1-micro: {test_f1_micro:.4f} ({test_f1_macro:.4f}, {test_f1_weighted:.4f})")

        met_test = test_f1_micro

        # Save best model
        if met_test > met_test_best:
            torch.save(model.state_dict(), f"model_{NICKNAME}.pt")
            torch.save({
                'epoch': epoch,
                'model_name': MODEL_NAME,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1_micro': met_test,
                'thresholds': best_thresholds,
                'image_size': IMAGE_SIZE,
            }, f"checkpoint_{NICKNAME}_best.pt")

            xfinal_pred_labels = [",".join(str(int(e)) for e in row) for row in pred_labels_test]
            xdf_dset_results = xdf_dset_test.copy()
            xdf_dset_results['results'] = xfinal_pred_labels
            xdf_dset_results.to_excel(f'results_{NICKNAME}.xlsx', index=False)

            met_test_best = met_test

        print(f"LR: {current_lr:.6f} | Best Val F1: {met_test_best:.4f}\n")

        # Early stopping
        if early_stopping(met_test, epoch):
            break

    writer.close()
    print(f"\n{'=' * 60}")
    print(f"TRAINING COMPLETE")
    print(f"Best Val F1-micro: {met_test_best:.4f}")
    print(f"Model: {MODEL_NAME}")
    print(f"Optimal thresholds: {best_thresholds}")
    print(f"{'=' * 60}\n")


def process_target(target_type):
    if target_type == 2:
        target = np.array(xdf_data['target'].apply(lambda x: x.split(",")))
        final_target = mlb.fit_transform(target)
        xfinal = [",".join(str(e) for e in row) for row in final_target]
        xdf_data['target_class'] = xfinal
    return mlb.classes_


if __name__ == '__main__':
    # for file in os.listdir(PATH + os.path.sep + "excel"):
    #     if file[-5:] == '.xlsx':
    #         FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file
    FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + "training" + os.path.sep + "train_test_cleaned.xlsx"

    xdf_data = pd.read_excel(FILE_NAME)
    class_names = process_target(target_type=2)

    # xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()
    # xdf_dset_test = xdf_data[xdf_data["split"] == 'test'].copy()

    xdf_dset_full = xdf_data[xdf_data["split"] == 'train'].copy()

    # Split into train and test
    xdf_dset, xdf_dset_test = train_test_split(
        xdf_dset_full,
        test_size=0.2,
        random_state=SEED,
        shuffle=True
    )

    train_ds, test_ds = read_data(target_type=2)
    OUTPUTS_a = len(class_names)

    print(f"\n{'=' * 60}")
    print(f"ULTIMATE SOTA CONFIGURATION")
    print(f"{'=' * 60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Training: {len(xdf_dset)} | Test: {len(xdf_dset_test)}")
    print(f"Classes: {OUTPUTS_a} | {class_names}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"Device: {device}")
    print(f"Mixed Precision (AMP): {USE_AMP}")
    print(f"Mixup/CutMix: {USE_MIXUP}")
    print(f"Label Smoothing: {LABEL_SMOOTHING}")
    print(f"{'=' * 60}\n")

    train_and_test(train_ds, test_ds)