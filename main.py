import os
import torch.utils.data
import torchio as tio
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.models as models


root_dir = '/Users/pesala/Documents/MICCAI_BraTS_2019_Data_Training 2/MICCAI_BraTS_2019_Data_Training'
#preprocessed_dir = '/Users/pesala/Documents/BraTA_dataset/preprocessed_dataset' #case_2
#the above step making transformations outside the loop is to reduce computation usage

subjects_list = []   #case_1
# #checked---tensor os 4-dim array [1,240,240,155]
#checked---labels are converted to interger format

class_labels = {'HGG': 0, 'LGG': 1}

preprocessing_transforms = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    #tio.Resize())
    ])

for classes in os.listdir(root_dir):
    class_path = os.path.join(root_dir, classes)
    if classes not in class_labels:
        continue
    for subjects in os.listdir(class_path):
        subjects_path = os.path.join(class_path, subjects)
        if not os.path.isdir(subjects_path):
            continue
        #preprocessed_subjects_path = os.path.join(preprocessed_dir, subjects)   #case_2
        for file in os.listdir(subjects_path):
            if file.endswith('_t1.nii.gz'):
                T1_sample_path = os.path.join(subjects_path, file)
                t1_subject = tio.Subject(t1=tio.ScalarImage(T1_sample_path), label=class_labels[classes])
                preprocessed_subject = preprocessing_transforms(t1_subject)
                #preprocessed_subject.save(preprocessed_subjects_path)   #case_2
                subjects_list.append(preprocessed_subject)    #case_1

    print(f"Total {classes} processed: {len(subjects_list)}")


dataset = tio.SubjectsDataset(subjects_list)
dataset_size = len(dataset)

dataset_type = type(dataset)
print(f"Type of dataset: {dataset_type}")
dataset_length = len(dataset)
print(f"Number of subjects in dataset: {dataset_length}")

#check data split consistency across each class
#check after preprocessing images and compare it with actual images

train_size = int(0.7*dataset_size)
val_test_size = dataset_size-train_size
test_size = val_test_size // 2
val_size = val_test_size-test_size

train_dataset, val_test_dataset = torch.utils.data.random_split(dataset,[train_size, val_test_size])
val_dataset, test_dataset = torch.utils.data.random_split(val_test_dataset, [val_size,test_size])

batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class ResNet3D(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.5):
        super(ResNet3D, self).__init__()

        self.resnet3d = models.video.r3d_18(pretrained=False)

        # Modify the first convolutional layer to take 1-channel input
        self.resnet3d.stem[0] = nn.Conv3d(
            1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False
        )

        num_features = self.resnet3d.fc.in_features

        self.resnet3d.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.resnet3d(x)


num_classes = 2  # actually the number of classes are 2 but since we are using binary cross entropy so we make it 1
model = ResNet3D(num_classes)

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()   #if classes are imbalanced then use focal loss
# if you change it to standard cross entropy then change the number of classes to 1 in final layer in the model
optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.00001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)

model.to(device)
num_epochs = 45
patience = 4
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(num_epochs):
    print("epoch started")
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for i, batch in enumerate(train_loader):
        inputs = batch['t1'][tio.DATA].to(device)  # Transfer image data to device

        # Check if 'label' is already a tensor. If it is, simply move it to the device.
        if torch.is_tensor(batch['label']):
            labels = batch['label'].to(device, dtype=torch.long)
        else:
            # If 'label' is a list or numpy array, convert it to a tensor and then move to the device.
            labels = torch.tensor(batch['label'], dtype=torch.long).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    print("epochmiddle")
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs = batch['t1'][tio.DATA].to(device)  # Transfer image data to device

            # Same handling for validation labels
            if torch.is_tensor(batch['label']):
                labels = batch['label'].to(device, dtype=torch.long)
            else:
                labels = torch.tensor(batch['label'], dtype=torch.long).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_val_loss += loss.item()

            # Calculate validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = running_val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print(f"Early stopping! Best validation loss: {best_val_loss:.4f}")
            break

    scheduler.step(val_loss)
