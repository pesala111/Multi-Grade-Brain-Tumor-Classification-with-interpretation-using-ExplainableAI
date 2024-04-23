"""
import modules
import data sample and preprocess it
import model
apply GradCam
visualisse the attributions
"""

from captum.attr import GuidedGradCam
import torch
import torchio as tio
import utils
from utils import MobileNetV2_3D
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F


# DATA PREPARATION
data_sample_HGG = "/Users/pesala/Documents/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/HGG/BraTS19_CBICA_ASH_1/BraTS19_CBICA_ASH_1_t1.nii.gz"
data_sample_LGG = "/Users/pesala/Documents/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/LGG/BraTS19_2013_1_1/BraTS19_2013_1_1_t1.nii.gz"
data_sample_Healthy = "/Users/pesala/Documents/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/Healthy/healthy_T1_2/IXI068-Guys-0756-T1_brain_resampled.nii.gz"
t1_subject = tio.Subject(t1=tio.ScalarImage(data_sample_Healthy))
preprocessing_transforms = tio.Compose([tio.RescaleIntensity(out_min_max=(0, 1)),
                                     tio.ZNormalization(masking_method=tio.ZNormalization.mean)])
preprocessed_subject = preprocessing_transforms(t1_subject)
dataset = tio.SubjectsDataset([preprocessed_subject])
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

class MobileNetV2_3D_GradCAM(MobileNetV2_3D):
    def __init__(self, input_shape, num_classes):
        super(MobileNetV2_3D_GradCAM, self).__init__(input_shape, num_classes)

        # Placeholder for the gradients and the activations
        self.gradients = None
        self.activations = None

    # Hook for the gradients of the activations
    def activations_hook(self, grad):
        print("Hook called.")  # Debug print to confirm the hook is called
        self.gradients = grad

    def forward_with_hooks(self, x):
        # Forward pass through initial layers or blocks
        x = self.initial_layer(x)
        # Intermediate layers or blocks
        # Make sure to iterate over blocks or layers where you want to register the hook
        for block in [self.block1, self.block2, self.block3, self.block4, self.block5, self.block6, self.block7,
                      self.block8, self.block9]:
            x = block(x)
            # Assuming you're interested in the activations from the last block for Grad-CAM
            if block == self.block9:
                self.activations = x
                h = x.register_hook(self.activations_hook)
                print("Hook registered")  # Confirm hook registration

        # Forward pass through remaining layers
        x = self.avg_pool(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return F.softmax(x, dim=1)

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations


# MODEL
input_shape = (1, 155, 240, 240)
num_classes = 3
model_path = "/Users/pesala/Downloads/brain_tumor_model_90.35%.pth"
#model = utils.MobileNetV2_3D(input_shape, num_classes)
model = MobileNetV2_3D_GradCAM(input_shape, num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

batch = next(iter(data_loader))
img_tensor = batch['t1']['data']
pred = model.forward_with_hooks(img_tensor)

#pred = model(img_tensor)
pred_class = pred.argmax(dim=1).item()

print(pred_class)

pred[:, pred_class].backward()

gradients = model.get_activations_gradient()

# Get the activations of the last convolutional layer
activations = model.get_activations().detach()  # Already captured during forward pass

# Pool the gradients across the channels
pooled_gradients = torch.mean(gradients, dim=[0, 2, 3, 4])

# Weight the channels by corresponding gradients
for i in range(activations.shape[1]):  # Iterate over the channel dimension
    activations[:, i, :, :, :] *= pooled_gradients[i]

# Average the channels of the activations to get the raw heatmap
heatmap = torch.mean(activations, dim=1).squeeze()  # Now should have the shape [4, 4, 3]

# Apply ReLU to the raw heatmap (Grad-CAM)
heatmap = torch.relu(heatmap)

# Normalize the heatmap
heatmap /= torch.max(heatmap)
print("Heatmap shape:", heatmap.shape)

# Choose a slice for 2D visualization, e.g., the middle slice along the last dimension
slice_index = heatmap.shape[2] // 2  # Middle index in the 3rd dimension
heatmap_2d = heatmap[:, :, slice_index].cpu().numpy()

# Visualize the selected slice
plt.matshow(heatmap_2d)
plt.colorbar()
plt.show()
