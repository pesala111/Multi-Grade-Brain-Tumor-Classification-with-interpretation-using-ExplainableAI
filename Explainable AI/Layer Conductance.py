import torch
import torchio as tio
from utils import MobileNetV2
import captum.attr
import matplotlib.pyplot as plt
import numpy as np

# Model loading
model = MobileNetV2(num_classes=3, sample_size=240, width_mult=1.)
model_path = "/Users/pesala/Downloads/brain tumor trained paths/brain_tumor_model_perfect_91.30%.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Data loading and preprocessing
data_sample_HGG = "/Users/pesala/Documents/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_CBICA_AXJ_1/BraTS19_CBICA_AXJ_1_t1.nii.gz"
t1_subject = tio.Subject(t1=tio.ScalarImage(data_sample_HGG))
preprocessing_transforms = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
])

preprocessed_subject = preprocessing_transforms(t1_subject)
input_tensor = preprocessed_subject['t1']['data'][None]
print(input_tensor.shape)
outputs = model(input_tensor)
_, predicted_class = torch.max(outputs.data, 1)
print(predicted_class.item())


target_layer = model.features[-1]

layer_conductance = captum.attr.LayerConductance(model, target_layer)
attributions = layer_conductance.attribute(input_tensor, target=predicted_class.item())

print("Shape of attributions:", attributions.shape)


# Layer Conductance processing
layer_conductance_map = attributions.squeeze().detach().numpy()
original_volume = preprocessed_subject['t1']['data'].squeeze().numpy()

alpha = 0.3
overlay = (1 - alpha) * original_volume + alpha * layer_conductance_map

plt.figure(figsize=(12, 6))

# Original MRI Slice
plt.subplot(1, 2, 1)
plt.imshow(original_volume[:, :, 86], cmap='gray')
plt.title('Original MRI Slice')

# Overlay Map for MRI Slice
plt.subplot(1, 2, 2)
plt.imshow(overlay[:, :, 86], cmap='hot')
plt.colorbar()
plt.title('Overlay Map for MRI Slice')

plt.tight_layout()
plt.show()

