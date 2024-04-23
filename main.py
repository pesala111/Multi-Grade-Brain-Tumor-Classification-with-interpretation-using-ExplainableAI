import torch
import torch.nn as nn
import torchio as tio
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import utils
# Initialize wandb (if using wandb for experiment tracking)
# import wandb
# wandb.init(project="Brain Tumor Classification", entity="your_entity_name")

# Define paths and parameters
root_dir = '/Users/pesala/Documents/brats_dataset_short/brats_dataset_short'
class_labels = {'HGG': 0, 'LGG': 1}

# Define preprocessing transforms
preprocessing_transforms = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
])

# Create subjects list
subjects_list = utils.create_subjects_list(root_dir, class_labels, preprocessing_transforms)
sample_subject = subjects_list[0]  # Get the first subject for simplicity
image = sample_subject['t1']
voxel_resolution = image.spacing  # Voxel size (resolution) in mm (width, height, depth)
volume_resolution = image.shape  # Volume size (resolution) in voxels (height, width, depth)

print(f"Voxel resolution (after preprocessing): {voxel_resolution} mm")
print(f"Volume resolution (after preprocessing): height={volume_resolution[0]}, width={volume_resolution[1]}, depth={volume_resolution[2]}")
"""
# Create dataset and split it
dataset = tio.SubjectsDataset(subjects_list)
train_size = int(0.7 * len(dataset))
val_test_size = len(dataset) - train_size
test_size = val_test_size // 2
val_size = val_test_size - test_size

train_dataset, val_test_dataset = torch.utils.data.random_split(dataset, [train_size, val_test_size])
val_dataset, test_dataset = torch.utils.data.random_split(val_test_dataset, [val_size, test_size])

# DataLoader setup
batch_size = 1
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, loss, and optimizer setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = utils.ResNet3D(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.00001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)

# Training loop
num_epochs = 45
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

for epoch in range(num_epochs):
    train_loss, train_accuracy = utils.train_model(model, train_loader, criterion, optimizer, device)
    val_loss, val_accuracy = utils.validate_model(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

    scheduler.step(val_loss)

# Testing the model
test_loss, test_accuracy = utils.test_model(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# Saving the model
torch.save(model.state_dict(), 'brain_tumor_model.pth')

# Plotting Training Curves
utils.plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

# Evaluation: Confusion Matrix and ROC Curve
predictions, true_labels = utils.get_predictions(model, test_loader, device)
utils.plot_confusion_matrix(true_labels, predictions, class_labels)
utils.plot_roc_curve(true_labels, predictions, "ResNet3D")
"""