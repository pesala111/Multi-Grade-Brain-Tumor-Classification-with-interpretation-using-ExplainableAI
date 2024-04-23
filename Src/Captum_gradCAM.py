import torch
import torchio as tio
from utils import MobileNetV2_3D
from captum.attr import LayerGradCam, LayerAttribution, LayerConductance, LayerActivation, LayerGradientXActivation
from captum.attr import visualization as viz

import matplotlib.pyplot as plt
import numpy as np

# Model loading
model = MobileNetV2_3D(input_shape=(1, 155, 240, 240), num_classes=3)
model_path = "/Users/pesala/Downloads/brain_tumor_model_91.30%.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Data loading and preprocessing
data_sample_HGG = "/Users/pesala/Documents/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_CBICA_ALU_1/BraTS19_CBICA_ALU_1_t1.nii.gz"
# HGG: tumor at (165,123,66)
data_sample_LGG = "/Users/pesala/Documents/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/LGG/BraTS19_TCIA09_620_1/BraTS19_TCIA09_620_1_t1.nii.gz"
data_sample_Healthy = "/Users/pesala/Documents/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/Healthy/healthy_T1_12/IXI531-Guys-1057-T1_brain_resampled.nii.gz"
t1_subject = tio.Subject(t1=tio.ScalarImage(data_sample_HGG))
print("data sample dimentions are (240,240,155)")
print(f"tio.subject shape is {t1_subject.shape}")
preprocessing_transforms = tio.Compose([
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ])

preprocessed_subject = preprocessing_transforms(t1_subject)
input_tensor = preprocessed_subject['t1']['data'][None]
#input_tensor = input_tensor.permute(0, 1, 4, 2, 3)
outputs = model(input_tensor)
_, predicted_class = torch.max(outputs.data, 1)
print(predicted_class.item())
#target_layer = model.block9
# computer for every layer and average those
grad_cam = LayerGradCam(model, model.block2)
attributions = grad_cam.attribute(input_tensor, target=predicted_class.item())
"""
# LAYER CONDUCTANCE METHOD
layer_conductance = LayerConductance(forward_func=model, layer=model.block2)
attributions = layer_conductance.attribute(input_tensor, target=predicted_class.item())

# LAYER ACTIVATION METHOD
layer_activation = LayerActivation(forward_func=model, layer=model.block2)
attributions = layer_activation.attribute(inputs=input_tensor, attribute_to_layer_input=False)

layer_grad_x_activation = LayerGradientXActivation(forward_func=model.forward, layer=model.block2, multiply_by_inputs=True)
attributions = layer_grad_x_activation.attribute(inputs=input_tensor, target=predicted_class.item(), additional_forward_args=None, attribute_to_layer_input=False)
"""
print("Shape of attributions:", attributions.shape)

upsampled_attributions = LayerAttribution.interpolate(attributions,(240, 240, 155),interpolate_mode='trilinear')
print("Shape of upsampled_attributions:", upsampled_attributions.shape)
heatmap = upsampled_attributions.squeeze().cpu().detach().numpy()
heatmap_normalized = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
heatmap_normalized = heatmap_normalized.squeeze() # shape is [240, 240, 155]
print("Shape of heatmap_normalized:", heatmap_normalized.shape)

original_volume = preprocessed_subject['t1']['data'].squeeze().numpy()
print("Shape of original volume:", original_volume.shape)

alpha = 0.3
overlay = (1 - alpha) * original_volume + alpha * heatmap_normalized
print("Shape of overlay:", overlay.shape)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
#plt.imshow(original_volume[:, :, 66], cmap='gray')
plt.imshow(original_volume[:, 123, :], cmap='gray')
#plt.imshow(original_volume[165, :, :], cmap='gray')
plt.title('Original Volume Slice')

plt.subplot(1, 2, 2)
#plt.imshow(overlay[:, :, 66], cmap='jet_r')
plt.imshow(overlay[:, 123, :], cmap='jet_r')
#plt.imshow(overlay[165, :, :], cmap='jet_r')
plt.title('Overlay with Heatmap')
plt.colorbar()
plt.show()



"""
# Visualization using Captum
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Visualize the original image slice
viz.visualize_image_attr(None,
                         original_volume[:, 123, ],
                         method="original_image",
                         use_pyplot=False,
                         plt_fig_axis=(fig, ax[0]))
ax[0].set_title('Original Volume Slice')
ax[0].axis('off')

# Visualize the overlay of heatmap on the original image
viz.visualize_image_attr(heatmap_normalized[:, 123, :],
                         original_volume[:, 123, :],
                         method="heat_map",
                         sign="all",
                         show_colorbar=True,
                         alpha_overlay=0.3,
                         use_pyplot=False,
                         plt_fig_axis=(fig, ax[1]))
ax[1].set_title('Overlay with Heatmap')
ax[1].axis('off')

plt.tight_layout()
plt.show()




"""




"""
main goal is to localise with high resolution fine grained detail
this model is predicting the class based on brain tumor and with more general brain charecteristics
this might result in correct predtiction but the model is not localising the tumor and taking tumor like shapes 
in the brain structure into consideration
"""


""""
PREDICTIONS OF THIS MODEL:
HGG: 
LGG: 55/76
Healthy: 
"""

