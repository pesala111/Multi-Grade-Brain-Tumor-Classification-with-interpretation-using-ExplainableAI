import torch
import torchio as tio
from utils import MobileNetV2_3D
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# Model loading
model = MobileNetV2_3D(input_shape=(1, 155, 240, 240), num_classes=3)
model_path = "/Users/pesala/Downloads/brain_tumor_model_91.30%.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Data loading and preprocessing
data_sample_HGG = "/Users/pesala/Documents/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_CBICA_ALU_1/BraTS19_CBICA_ALU_1_t1.nii.gz"
t1_subject = tio.Subject(t1=tio.ScalarImage(data_sample_HGG))
preprocessing_transforms = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
])

preprocessed_subject = preprocessing_transforms(t1_subject)
input_tensor = preprocessed_subject['t1']['data'][None]  # Add batch dimension

# Enable gradients for input
input_tensor.requires_grad = True

# Forward pass
outputs = model(input_tensor)
_, predicted_class = torch.max(outputs, 1)

# Zero gradients
model.zero_grad()

# Compute gradients
outputs[0, predicted_class.item()].backward()

# Saliency map
saliency = input_tensor.grad.data.abs().squeeze()
saliency = saliency.numpy()  # Convert to numpy array for visualization

# Plotting
original_volume = preprocessed_subject['t1']['data'].squeeze().numpy()
slice_index = 123  # For example, choose a slice index that you are interested in

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(original_volume[:, slice_index, :], cmap='gray')
plt.title('Original MRI Slice')

plt.subplot(1, 2, 2)
plt.imshow(saliency[:, slice_index, :], cmap='hot')
plt.colorbar()
plt.title('Saliency Map for MRI Slice')
plt.show()
