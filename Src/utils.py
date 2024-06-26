import os
import torchio as tio
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize


# Data Preprocessing Class
def create_subjects_list(root_dir, class_labels, preprocessing_transforms,
                         rotation_transform, flipping_transform, elastic_transform):
    subjects_list = []
    for classes in os.listdir(root_dir):
        class_path = os.path.join(root_dir, classes)
        if classes not in class_labels:
            continue
        print(classes)
        i = 0
        class_subjects_count = 0

        for subjects in os.listdir(class_path):
            subjects_path = os.path.join(class_path, subjects)
            if not os.path.isdir(subjects_path):
                continue
            for file in os.listdir(subjects_path):
                if file.endswith('t1.nii.gz') or file.endswith('resampled.nii.gz'):
                    T1_sample_path = os.path.join(subjects_path, file)
                    i+=1
                    t1_subject = tio.Subject(t1=tio.ScalarImage(T1_sample_path), label=class_labels[classes])
                    preprocessed_subject = preprocessing_transforms(t1_subject)
                    subjects_list.append(preprocessed_subject)
                    class_subjects_count += 1

                    if classes == "LGG":
                        # rotation augmentation
                        rotation_subject = rotation_transform(t1_subject)
                        preprocessed_subject = preprocessing_transforms(rotation_subject)
                        subjects_list.append(preprocessed_subject)
                        # flipping augmentation
                        flipping_subject = flipping_transform(t1_subject)
                        preprocessed_subject = preprocessing_transforms(flipping_subject)
                        subjects_list.append(preprocessed_subject)
                        # elastic transform augmentation
                        elastic_subject = elastic_transform(t1_subject)
                        preprocessed_subject = preprocessing_transforms(elastic_subject)
                        subjects_list.append(preprocessed_subject)
                        class_subjects_count += 3

        print(f"Total {classes} processed: {class_subjects_count}")

    return subjects_list


# Model Defining Class

# MobileNetV2
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1, 1, 1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1, 1, 1) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes, sample_size, width_mult):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 16
        last_channel = 640
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 2, (1, 1, 1)],
            [6, 24, 2, (2, 2, 2)],
            [6, 32, 2, (2, 2, 2)],
            [6, 64, 1, (2, 2, 2)],
            [6, 96, 3, (1, 1, 1)],
            [6, 160, 1, (2, 2, 2)],
            [6, 320, 1, (1, 1, 1)],
        ]

        # building first layer
        assert sample_size % 16 == 0.
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, (1, 2, 2))]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.avg_pool3d(x, x.data.size()[-3:])
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()



# ResNet3D-18 Model
class ResNet3D(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(ResNet3D, self).__init__()
        weights = R3D_18_Weights.KINETICS400_V1
        self.resnet3d = models.video.r3d_18(weights=weights)
        # Modified the first convolutional layer to take 1-channel(grey) input
        self.resnet3d.stem[0] = nn.Conv3d(
            1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        num_features = self.resnet3d.fc.in_features
        self.resnet3d.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(num_features, num_classes)
        )
    def forward(self, x):
        return self.resnet3d(x)


# Model Training Class
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, batch in enumerate(train_loader):
        inputs = batch['t1'][tio.DATA].to(device)

        # Check if label is already a tensor.
        if torch.is_tensor(batch['label']):
            labels = batch['label'].to(device, dtype=torch.long)
        else:
            # If the label is a list or numpy array, then convert it to tensor
            labels = torch.tensor(batch['label'], dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    return train_loss, train_accuracy

# Validation CLass
def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch['t1'][tio.DATA].to(device)
            if torch.is_tensor(batch['label']):
                labels = batch['label'].to(device, dtype=torch.long)
            else:
                labels = torch.tensor(batch['label'], dtype=torch.long).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    return val_loss, val_accuracy

# Testing class
def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs = batch['t1'][tio.DATA].to(device)
            if torch.is_tensor(batch['label']):
                labels = batch['label'].to(device, dtype=torch.long)
            else:
                labels = torch.tensor(batch['label'], dtype=torch.long).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    test_accuracy = 100 * correct_test / total_test
    return test_loss / len(test_loader), test_accuracy

# Model Prediction
def get_predictions(model, dataloader, device):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['t1'][tio.DATA].to(device)

            if torch.is_tensor(batch['label']):
                labels = batch['label'].to(device, dtype=torch.long)
            else:
                labels = torch.tensor(batch['label'], dtype=torch.long).to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()  # Use softmax for multi-class probabilities
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_probs, all_preds, all_labels

# Confusion Matrix
def plot_confusion_matrix(true_labels, predictions, class_labels):
    cm = confusion_matrix(true_labels, predictions)
    labels = [label for label in class_labels.values()]
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()



# Accuracy vs Loos Metrics
def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    # Plotting Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss', color='red')
    plt.plot(epochs, val_losses, label='Validation Loss', color='blue')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plotting Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='red')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='blue')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()



# ROC Curve
def plot_roc_curve(y_true, y_score, n_classes):
    # Binarize the output
    y_true = label_binarize(y_true, classes=list(range(n_classes)))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve - One vs Rest')
    plt.legend(loc="lower right")
    plt.show()


def compute_metrics(all_labels, all_preds):
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds,
                                average='weighted')
    recall = recall_score(all_labels, all_preds,
                          average='weighted')
    f1 = f1_score(all_labels, all_preds,
                  average='weighted')

    return accuracy, precision, recall, f1