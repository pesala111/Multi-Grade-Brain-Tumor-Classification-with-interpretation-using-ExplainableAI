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
"""
def conv_block(in_channels, out_channels, kernel_size, stride, padding='same'):
    if padding == 'same':
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual3D, self).__init__()
        hidden_dim = round(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(conv_block(in_channels, hidden_dim, kernel_size=1, stride=1))
        layers.extend([
            DepthwiseConv3D(hidden_dim, hidden_dim, stride=stride),
            nn.Conv3d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm3d(out_channels),
        ])

        self.layers = nn.Sequential(*layers)
        self.out_channels = out_channels

    def forward(self, x):
        if self.use_res_connect:
            return x + self.layers(x)
        else:
            return self.layers(x)

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
        self.block1 = InvertedResidual3D(32, 16, 1, expand_ratio=1)
        self.block2 = InvertedResidual3D(16, 24, 2, expand_ratio=6)  # Stride 2 for downsampling
        self.block3 = InvertedResidual3D(24, 32, 2, expand_ratio=6)  # Repeated blocks can be added here
        self.block4 = InvertedResidual3D(32, 64, 2, expand_ratio=6)
        self.block5 = InvertedResidual3D(64, 96, 1, expand_ratio=6)  # Stride 1 to maintain dimensions
        self.block6 = InvertedResidual3D(96, 160, 2, expand_ratio=6)  # New block for downsampling
        self.block7 = InvertedResidual3D(160, 320, 1, expand_ratio=6)  # New block to increase features
        self.block8 = InvertedResidual3D(320, 480, 2, expand_ratio=6)  # Additional block for more complexity
        self.block9 = InvertedResidual3D(480, 640, 1, expand_ratio=6)  # Additional block to increase features further

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv2 = nn.Conv3d(640, 1280, 1)
        self.dropout = nn.Dropout3d(p=0.05)
        self.classifier = nn.Linear(1280, num_classes)

        #self.conv2 = nn.Conv3d(96, num_classes, 1)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)

        x = self.avg_pool(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        #return x

        return F.softmax(x, dim=1)
"""


import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
            [1, 16, 1, (1, 1, 1)],
            [6, 24, 2, (2, 2, 2)],
            [6, 32, 2, (2, 2, 2)],
            [6, 64, 1, (2, 2, 2)],
            [6, 96, 1, (1, 1, 1)],
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
            nn.Dropout(0.4),
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





from torchvision.models.video import r3d_18, R3D_18_Weights

class ModifiedResNet3D(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.6):
        super(ModifiedResNet3D, self).__init__()
        # Load the pre-trained ResNet 3D-18 model
        weights = R3D_18_Weights.KINETICS400_V1
        self.resnet3d = r3d_18(weights=weights)

        # Modify the first convolutional layer to take 1-channel(grey) input
        self.resnet3d.stem[0] = nn.Conv3d(
            1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

        self.resnet3d.layer3 = nn.Sequential(*list(self.resnet3d.layer3.children())[:1])

        # Reduce layer4 to only the first block
        self.resnet3d.layer4 = nn.Sequential(*list(self.resnet3d.layer4.children())[:1])

        # Adjust the fully connected layer
        num_features = self.resnet3d.fc.in_features
        self.resnet3d.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.resnet3d(x)
"""

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
"""
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
"""
def plot_roc_curve(true_labels, predictions, model_name):
    # Assuming 'predictions' are probability scores and true_labels are one-hot encoded
    n_classes = predictions.shape[1]
    plt.figure(figsize=(10, 8))

    # Compute ROC curve and ROC area for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(true_labels[:, i], predictions[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve of class {i} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
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