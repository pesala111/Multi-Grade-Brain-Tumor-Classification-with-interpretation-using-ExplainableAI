import os
import torch
import torchio as tio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.models as models
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.models.video.resnet import R3D_18_Weights

# Data Preprocessing Class
def create_subjects_list(root_dir, class_labels, preprocessing_transforms):
    subjects_list = []
    for classes in os.listdir(root_dir):
        class_path = os.path.join(root_dir, classes)
        if classes not in class_labels:
            continue
        for subjects in os.listdir(class_path):
            subjects_path = os.path.join(class_path, subjects)
            if not os.path.isdir(subjects_path):
                continue
            for file in os.listdir(subjects_path):
                if file.endswith('_t1.nii.gz'):
                    T1_sample_path = os.path.join(subjects_path, file)
                    t1_subject = tio.Subject(t1=tio.ScalarImage(T1_sample_path), label=class_labels[classes])
                    preprocessed_subject = preprocessing_transforms(t1_subject)
                    subjects_list.append(preprocessed_subject)
        print(f"Total {classes} processed: {len(subjects_list)}")

    return subjects_list

# Model Defining Class
def conv_block(in_channels, out_channels, kernel_size, stride, padding='same'):
    if padding == 'same':
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU6(inplace=True)
    )

class DepthwiseConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseConv3D, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class MobileNetV2_3D(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(MobileNetV2_3D, self).__init__()
        self.input_shape = input_shape
        in_channels = input_shape[0]

        self.initial_layer = conv_block(in_channels, 32, 3, 2, padding='same')
        self.block1 = DepthwiseConv3D(32, 64, 1)

        # Add more blocks as needed following the MobileNetV2 architecture

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv2 = nn.Conv3d(64, num_classes, 1)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.block1(x)

        # Forward through additional blocks

        x = self.avg_pool(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        return F.softmax(x, dim=1)

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


class Simple3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(128 * 136800, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        print("Output size after conv2:", x.size())
        x = x.view(-1, 128 * 136800)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
"""

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool3d(output_size=(20, 20, 20))

        # Adjusted size for the fc1 layer
        self.fc1 = nn.Linear(128 * 20 * 20 * 20, 1000)  # Approximately 128 * 40 * 40 * 40
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.adaptive_pool(x)
        x = x.view(-1, 128 * 20 * 20 * 20)  # Flatten the tensor for the fc1 layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

"""

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

# Model Prediction / Evaluation
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
            probs = torch.sigmoid(outputs).squeeze()
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs.cpu().numpy())  # Store probabilities
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

# ROC Curve
def plot_roc_curve(true_labels, predictions, model_name):
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {ResNet3D}')
    plt.legend(loc='lower right')
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