import random
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils import data
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchvision import transforms
from tqdm import tqdm
import os
os.system("sudo pip install albumentations opencv-python-headless timm")

# Add these imports at the top
from torchvision import models
import timm 

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
import datetime

'''
IMPROVED VERSION - Enhanced for better performance
Target: F1-micro score 0.75-0.80
'''

# Set random seeds for reproducibility
SEED = 67128
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

OR_PATH = os.getcwd()
os.chdir("../../..")
PATH = os.getcwd()
DATA_DIR = os.getcwd() + os.path.sep + 'Data' + os.path.sep
sep = os.path.sep

os.chdir(OR_PATH)

# Hyperparameters
n_epoch = 150
BATCH_SIZE = 64  # Reduced for better generalization
LR = 0.0001
USE_PRETRAINED = True
WEIGHT_DECAY = 1e-4

# Scheduler type: 'cosine', 'cosine_restarts', or 'reduce_on_plateau'
SCHEDULER_TYPE = 'cosine_restarts'  # Best for your case
T_0 = 10  # For cosine restarts: epochs before first restart
T_MULT = 2  # For cosine restarts: factor to increase T_0 after each restart

# Image processing - INCREASE SIZE
CHANNELS = 3
IMAGE_SIZE = 224  # Increase from 224 to 384 (or 512 if GPU allows)

NICKNAME = "Andrew"

mlb = MultiLabelBinarizer()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
THRESHOLD = 0.5
SAVE_MODEL = True

# ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ImprovedCNN(nn.Module):
    """Enhanced CNN with dropout and better architecture"""

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

        # Load label
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

        # Load image
        if self.type_data == 'train':
            file = DATA_DIR + xdf_dset.id.get(ID)
        else:
            file = DATA_DIR + xdf_dset_test.id.get(ID)

        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Apply augmentation/transformation
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        else:
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            img = img.astype(np.float32) / 255.0
            img = (img - MEAN) / STD
            img = torch.FloatTensor(img).permute(2, 0, 1)

        return img, y


def get_train_transforms():
    """Strong augmentation for training"""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1),
        ], p=0.5),
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1),
            A.GaussianBlur(blur_limit=(3, 7), p=1),
            A.MotionBlur(blur_limit=7, p=1),
        ], p=0.3),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Minimal transformation for validation"""
    return A.Compose([
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])


def read_data(target_type):
    list_of_ids = list(xdf_dset.index)
    list_of_ids_test = list(xdf_dset_test.index)

    partition = {
        'train': list_of_ids,
        'test': list_of_ids_test
    }

    # Training set with augmentation
    params_train = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 4, 'pin_memory': True}
    training_set = Dataset(partition['train'], 'train', target_type, transform=get_train_transforms())
    training_generator = data.DataLoader(training_set, **params_train)

    # Test set without augmentation
    params_test = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 4, 'pin_memory': True}
    test_set = Dataset(partition['test'], 'test', target_type, transform=get_val_transforms())
    test_generator = data.DataLoader(test_set, **params_test)

    return training_generator, test_generator


def save_model(model):
    print(model, file=open('summary_{}.txt'.format(NICKNAME), "w"))


def get_scheduler(optimizer, scheduler_type='cosine_restarts'):
    """
    Get learning rate scheduler based on type

    Comparison:
    - cosine: Smooth decay from max to min, good for stable training
    - cosine_restarts: Periodic restarts, helps escape local minima (RECOMMENDED)
    - reduce_on_plateau: Reactive, reduces LR when metric plateaus
    """
    if scheduler_type == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=n_epoch,  # Full cycle length
            eta_min=1e-6  # Minimum learning rate
        )
        print(f"Using CosineAnnealingLR scheduler")

    elif scheduler_type == 'cosine_restarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,  # Epochs before first restart
            T_mult=T_MULT,  # Multiply T_0 after each restart
            eta_min=1e-6  # Minimum learning rate
        )
        print(f"Using CosineAnnealingWarmRestarts scheduler (T_0={T_0}, T_mult={T_MULT})")

    elif scheduler_type == 'reduce_on_plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',  # Maximize F1 score
            factor=0.5,  # Reduce LR by half
            patience=5,  # Wait 5 epochs
            verbose=True,
            min_lr=1e-6
        )
        print(f"Using ReduceLROnPlateau scheduler")

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


def model_definition(pretrained=False):
    if pretrained:
        # OPTION 1: EfficientNet (Best balance of speed/accuracy)
        # model = timm.create_model('tf_efficientnet_b4_ns', pretrained=True, num_classes=OUTPUTS_a)
        
        # OPTION 2: Vision Transformer (Highest accuracy, slower)
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=OUTPUTS_a)
        
        # OPTION 3: ConvNeXt (Modern CNN, very strong)
        # model = timm.create_model('convnext_base', pretrained=True, num_classes=OUTPUTS_a)
        
        # OPTION 4: Swin Transformer (Excellent for multi-label)
        # model = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=OUTPUTS_a)
        
        # Fine-tune only last layers initially
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'fc' not in name and 'head' not in name:
                param.requires_grad = False
                
    else:
        model = ImprovedCNN(OUTPUTS_a)

    model = model.to(device)

    # Use AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # Asymmetric Loss - BEST for multi-label with imbalance
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)

    # Get scheduler based on type
    scheduler = get_scheduler(optimizer, scheduler_type=SCHEDULER_TYPE)

    save_model(model)

    return model, optimizer, criterion, scheduler


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification
    Paper: https://arxiv.org/abs/2009.14119
    Best for multi-label with class imbalance
    """
    def __init__(self, gamma_neg=4, gamma_pos=0, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        """
        x: logits (raw outputs)
        y: targets (multi-hot encoded)
        """
        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            loss *= one_sided_w

        return -loss.sum()


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving"""

    def __init__(self, patience=7, min_delta=0.001, mode='max', verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as an improvement
            mode (str): 'max' for metrics to maximize (f1, accuracy), 'min' for metrics to minimize (loss)
            verbose (bool): Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, current_score, epoch):
        if self.best_score is None:
            self.best_score = current_score
            self.best_epoch = epoch
            if self.verbose:
                print(f"Initial best score: {current_score:.4f}")
            return False

        if self.mode == 'max':
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.best_epoch = epoch
                self.counter = 0
                if self.verbose:
                    print(f"✓ Improvement! New best score: {current_score:.4f}")
                return False
            else:
                self.counter += 1
                if self.verbose:
                    print(
                        f"No improvement for {self.counter}/{self.patience} epochs (best: {self.best_score:.4f} at epoch {self.best_epoch})")
        else:  # mode == 'min'
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.best_epoch = epoch
                self.counter = 0
                if self.verbose:
                    print(f"✓ Improvement! New best score: {current_score:.4f}")
                return False
            else:
                self.counter += 1
                if self.verbose:
                    print(
                        f"No improvement for {self.counter}/{self.patience} epochs (best: {self.best_score:.4f} at epoch {self.best_epoch})")

        if self.counter >= self.patience:
            if self.verbose:
                print(f"\n{'=' * 60}")
                print(f"Early stopping triggered!")
                print(f"Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                print(f"{'=' * 60}\n")
            self.early_stop = True
            return True

        return False


def find_optimal_threshold(y_true, y_pred_probs):
    """Find optimal threshold for each class"""
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


def train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on, pretrained=False):
    model, optimizer, criterion, scheduler = model_definition(pretrained)

    # Initialize TensorBoard writer
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/{NICKNAME}_{timestamp}'
    writer = SummaryWriter(log_dir)
    print(f"\n{'=' * 60}")
    print(f"TensorBoard logging to: {log_dir}")
    print(f"Run: tensorboard --logdir=runs")
    print(f"{'=' * 60}\n")

    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=10,  # Wait 10 epochs without improvement
        min_delta=0.001,  # Minimum improvement threshold
        mode='max',  # Maximize F1 score
        verbose=True
    )

    train_loss_item = []
    test_loss_item = []
    met_test_best = 0
    best_thresholds = [THRESHOLD] * OUTPUTS_a

    # Track history for plotting/analysis
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_f1': [],
        'val_f1': [],
        'learning_rate': []
    }

    for epoch in range(n_epoch):
        # Training phase
        model.train()
        train_loss, steps_train = 0, 0
        pred_logits_train, real_labels_train = [], []

        with tqdm(total=len(train_ds), desc=f"Epoch {epoch + 1}/{n_epoch} [Train]") as pbar:
            for batch_idx, (xdata, xtarget) in enumerate(train_ds):
                xdata, xtarget = xdata.to(device), xtarget.to(device)

                optimizer.zero_grad()
                output = model(xdata)
                loss = criterion(output, xtarget)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()
                steps_train += 1
                train_loss_item.append([epoch, loss.item()])

                pred_logits_train.append(torch.sigmoid(output).detach().cpu().numpy())
                real_labels_train.append(xtarget.cpu().numpy())

                # Log batch-level metrics to TensorBoard
                global_step = epoch * len(train_ds) + batch_idx
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)

                pbar.update(1)
                pbar.set_postfix_str(f"Loss: {train_loss / steps_train:.4f}")

        # Convert to numpy arrays
        pred_logits_train = np.vstack(pred_logits_train)
        real_labels_train = np.vstack(real_labels_train)

        # Validation phase
        model.eval()
        test_loss, steps_test = 0, 0
        pred_logits_test, real_labels_test = [], []

        with torch.no_grad():
            with tqdm(total=len(test_ds), desc=f"Epoch {epoch + 1}/{n_epoch} [Val]") as pbar:
                for xdata, xtarget in test_ds:
                    xdata, xtarget = xdata.to(device), xtarget.to(device)
                    output = model(xdata)
                    loss = criterion(output, xtarget)

                    test_loss += loss.item()
                    steps_test += 1
                    test_loss_item.append([epoch, loss.item()])

                    pred_logits_test.append(torch.sigmoid(output).cpu().numpy())
                    real_labels_test.append(xtarget.cpu().numpy())

                    pbar.update(1)
                    pbar.set_postfix_str(f"Loss: {test_loss / steps_test:.4f}")

        # Convert to numpy arrays
        pred_logits_test = np.vstack(pred_logits_test)
        real_labels_test = np.vstack(real_labels_test)

        # Find optimal thresholds every 5 epochs
        if epoch % 5 == 0 or epoch == n_epoch - 1:
            best_thresholds = find_optimal_threshold(real_labels_test, pred_logits_test)
            print(f"\nOptimal thresholds: {[f'{t:.2f}' for t in best_thresholds]}")

            # Log thresholds to TensorBoard
            for i, thresh in enumerate(best_thresholds):
                writer.add_scalar(f'Thresholds/class_{i}', thresh, epoch)

        # Apply thresholds
        pred_labels_test = np.zeros_like(pred_logits_test)
        for i, thresh in enumerate(best_thresholds):
            pred_labels_test[:, i] = (pred_logits_test[:, i] >= thresh).astype(int)

        # Calculate metrics
        train_pred_labels = (pred_logits_train >= THRESHOLD).astype(int)
        train_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels_train, train_pred_labels)
        test_metrics = metrics_func(list_of_metrics, list_of_agg, real_labels_test, pred_labels_test)

        avg_train_loss = train_loss / steps_train
        avg_test_loss = test_loss / steps_test
        current_lr = optimizer.param_groups[0]['lr']

        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_test_loss)
        history['train_f1'].append(train_metrics.get('f1_micro', 0))
        history['val_f1'].append(test_metrics.get('f1_micro', 0))
        history['learning_rate'].append(current_lr)

        # Log epoch-level metrics to TensorBoard
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', avg_test_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        # Log all metrics
        for met, dat in train_metrics.items():
            writer.add_scalar(f'Metrics/train_{met}', dat, epoch)
        for met, dat in test_metrics.items():
            writer.add_scalar(f'Metrics/val_{met}', dat, epoch)

        # Create comparison scalars
        writer.add_scalars('Loss/train_vs_val', {
            'train': avg_train_loss,
            'val': avg_test_loss
        }, epoch)

        writer.add_scalars('F1/train_vs_val', {
            'train': train_metrics.get('f1_micro', 0),
            'val': test_metrics.get('f1_micro', 0)
        }, epoch)

        # Log per-class F1 scores
        for i in range(OUTPUTS_a):
            class_f1_train = f1_score(real_labels_train[:, i], train_pred_labels[:, i], zero_division=0)
            class_f1_val = f1_score(real_labels_test[:, i], pred_labels_test[:, i], zero_division=0)
            writer.add_scalar(f'F1_per_class/train_class_{i}', class_f1_train, epoch)
            writer.add_scalar(f'F1_per_class/val_class_{i}', class_f1_val, epoch)

        # Print metrics
        xstrres = f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}"
        for met, dat in train_metrics.items():
            xstrres += f' | Train {met}: {dat:.4f}'

        xstrres += f" || Val Loss: {avg_test_loss:.4f}"
        met_test = 0
        for met, dat in test_metrics.items():
            xstrres += f' | Val {met}: {dat:.4f}'
            if met == save_on:
                met_test = dat

        print(xstrres)

        # Save best model
        if met_test > met_test_best and SAVE_MODEL:
            torch.save(model.state_dict(), f"model_{NICKNAME}.pt")

            # Also save with epoch number for backup
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1_micro': met_test,
                'thresholds': best_thresholds,
            }, f"checkpoint_{NICKNAME}_epoch_{epoch}.pt")

            # Save predictions
            xfinal_pred_labels = []
            for i in range(len(pred_labels_test)):
                joined_string = ",".join(str(int(e)) for e in pred_labels_test[i])
                xfinal_pred_labels.append(joined_string)

            xdf_dset_results = xdf_dset_test.copy()
            xdf_dset_results['results'] = xfinal_pred_labels
            xdf_dset_results.to_excel(f'results_{NICKNAME}.xlsx', index=False)

            print(f"✓ Model saved! Val {save_on}: {met_test:.4f}")
            met_test_best = met_test

            # Log best score to TensorBoard
            writer.add_scalar('Best/val_f1_micro', met_test_best, epoch)

        # Update learning rate
        if SCHEDULER_TYPE == 'reduce_on_plateau':
            scheduler.step(met_test)  # ReduceLROnPlateau needs the metric
        else:
            scheduler.step()  # Cosine schedulers don't need metric

        print(f"Learning Rate: {current_lr:.6f}")

        # Check early stopping
        if early_stopping(met_test, epoch):
            print(f"Training stopped early at epoch {epoch + 1}")
            print(
                f"Best model was at epoch {early_stopping.best_epoch + 1} with {save_on}: {early_stopping.best_score:.4f}")
            break

        print()  # Empty line for readability

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(f'training_history_{NICKNAME}.csv', index=False)
    print(f"\nTraining history saved to: training_history_{NICKNAME}.csv")

    # Close TensorBoard writer
    writer.close()
    print(f"TensorBoard logs saved to: {log_dir}")

    return history


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
    # Find Excel file
    for file in os.listdir(PATH + os.path.sep + "excel"):
        if file[-5:] == '.xlsx':
            FILE_NAME = PATH + os.path.sep + "excel" + os.path.sep + file

    # Read data
    xdf_data = pd.read_excel(FILE_NAME)

    # Process targets
    class_names = process_target(target_type=2)

    # Split data
    xdf_dset = xdf_data[xdf_data["split"] == 'train'].copy()
    xdf_dset_test = xdf_data[xdf_data["split"] == 'test'].copy()

    # Create dataloaders
    train_ds, test_ds = read_data(target_type=2)

    OUTPUTS_a = len(class_names)

    print(f"\n{'=' * 60}")
    print(f"Dataset Information:")
    print(f"{'=' * 60}")
    print(f"Training samples: {len(xdf_dset)}")
    print(f"Test samples: {len(xdf_dset_test)}")
    print(f"Number of classes: {OUTPUTS_a}")
    print(f"Classes: {class_names}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: {device}")
    print(f"{'=' * 60}\n")

    list_of_metrics = ['f1_micro', 'f1_macro', 'f1_weighted']
    list_of_agg = ['avg']

    train_and_test(train_ds, test_ds, list_of_metrics, list_of_agg, save_on='f1_micro', pretrained=USE_PRETRAINED)