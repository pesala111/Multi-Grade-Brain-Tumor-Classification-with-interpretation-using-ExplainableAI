
# ===============================================================================================

# This code is to check the spatial dimentions and voxel resolution

"""
import torchio as tio
import os
import nibabel as nib
import numpy as np
i = 0

root_dir = '/Users/pesala/Documents/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/Healthy'


for t1_img_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, t1_img_folder)
        if not os.path.isdir(folder_path):
            continue
        for t1_img in os.listdir(folder_path):
            if t1_img.endswith('resampled.nii.gz'):
                T1_sample_path = os.path.join(folder_path, t1_img)

                subject = tio.Subject(image=tio.ScalarImage(T1_sample_path))
                i += 1
                voxel_resolution = subject.spacing
                volume_resolution = subject.shape

                #print(voxel_resolution)
                #print(t1_img_folder)
                print(f"{volume_resolution[0]}, {volume_resolution[1]}, {volume_resolution[2]}, {volume_resolution[3]}")

"""

# =====================================================================================================================================


# This code is to change the spational dimentions and voxel resolution (similar to previos code block but with different directories

"""

import torchio as tio
import os

root_dir = '/Users/pesala/Documents/Healthy_axial'
target_shape = (240, 240, 155)
crop_or_pad = tio.CropOrPad(target_shape)
target_spacing = (1.0, 1.0, 1.0)
resample = tio.Resample(target_spacing)
resized_folder_path = '/Users/pesala/Documents/Healthy_axial/resampled'

i = 0

for t1_img in os.listdir(root_dir):
    if t1_img.endswith('T1.nii.gz'):
        file_path = os.path.join(root_dir, t1_img)
        subject = tio.Subject(image=tio.ScalarImage(file_path))

        transformed_image = resample(subject)
        transformed_image = crop_or_pad(transformed_image)


        i += 1
        subject_id = f"healthy_T1_{i}"
        subject_dir = os.path.join(resized_folder_path, subject_id)
        os.makedirs(subject_dir, exist_ok=True)

        output_file_path = os.path.join(subject_dir, t1_img)
        transformed_image['image'].save(output_file_path)
        #print(i)
        voxel_resolution = transformed_image.spacing
        volume_resolution = transformed_image.shape
        #print(f"{i}: {voxel_resolution}")
        print(f"{volume_resolution[0]}, {volume_resolution[1]}, {volume_resolution[2]}, {volume_resolution[3]}")

"""

# ========================================================================================


# This code is to change the plane of the nifty image (coronal, axxial, sagittal)

"""
import os
import nibabel as nib
import numpy as np
import torchio as tio


def reorient_image_nibabel(file_path, output_file_path):
    img = nib.load(file_path)
    data = img.get_fdata()
    reoriented_data = np.transpose(data, (2, 1, 0))
    reoriented_img = nib.Nifti1Image(reoriented_data, img.affine)
    nib.save(reoriented_img, output_file_path)


root_dir = '/Users/pesala/Downloads/IXI-T1'
new_folder_path = '/Users/pesala/Documents/Healthy_axial'
i = 0

for t1_img in os.listdir(root_dir):
    if t1_img.endswith('T1.nii.gz'):
        i += 1
        file_path = os.path.join(root_dir, t1_img)
        subject_id = f"healthy_T1_{i}"
        subject_dir = os.path.join(new_folder_path, subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        output_file_path = os.path.join(subject_dir, t1_img)

        # Reorient the image using nibabel and save
        reorient_image_nibabel(file_path, output_file_path)

        # Now, load the reoriented image with TorchIO if needed
        subject = tio.Subject(image=tio.ScalarImage(output_file_path))
        print(f"Processed and saved: {output_file_path}")
"""

# ======================================================================================

"""
# This code is to change the spational dimentions and voxel resolution (similar to previos code block but with different directories



import torchio as tio
import nibabel as nib
import nilearn
import nilearn.image
import os

root_dir = '/Users/pesala/Documents/MICCAI_BraTS_inference_dataset/HGG'
resized_folder_path = '/Users/pesala/Documents/brats_depth_160/HGG'
#target_img = ''
i = 0

for t1_img in os.listdir(root_dir):
    if t1_img.endswith('resampled.nii.gz'):
        source_img = os.path.join(root_dir, t1_img)
        subject = tio.Subject(image=tio.ScalarImage(source_img))

        resampled_img = nilearn.image.resample_to_img(source_img, target_img, interpolation='continuous',
                                     copy=True, order='F', clip=False, fill_value=0, force_resample=False)


        i += 1
        subject_id = f"healthy_T1_{i}"
        subject_dir = os.path.join(resized_folder_path, subject_id)
        os.makedirs(subject_dir, exist_ok=True)
        output_file_path = os.path.join(subject_dir, t1_img)
        #subject.to_filename(output_file_path)
        subject['image'].save(output_file_path)

        print(i)
        #voxel_resolution = resample_img.spacing
        #volume_resolution = subject.shape
        #print(f"{i}: {voxel_resolution}")
        #print(f"{volume_resolution[0]}, {volume_resolution[1]}, {volume_resolution[2]}, {volume_resolution[3]}")

"""
# =========================================================================================================================================

# This is the inference code to check the predicted class

"""
import torch
import torchio as tio
from utils import MobileNetV2_3D
import os
from captum.attr import LayerGradCam, LayerAttribution
from captum.attr import visualization as viz

import matplotlib.pyplot as plt
import numpy as np

# Model loading
model = MobileNetV2_3D(input_shape=(1, 155, 240, 240), num_classes=3)
model_path = "/Users/pesala/Downloads/brain_tumor_model_91.30%.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

root_dir = "/Users/pesala/Documents/MICCAI_BraTS_inference_dataset/LGG"
i = 0
j = 0
for t1_img_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, t1_img_folder)
        if not os.path.isdir(folder_path):
            continue
        for t1_img in os.listdir(folder_path):
            if t1_img.endswith('t1.nii.gz'):
                j+=1
                T1_sample_path = os.path.join(folder_path, t1_img)

                t1_subject = tio.Subject(t1=tio.ScalarImage(T1_sample_path))
                preprocessing_transforms = tio.Compose([
                    tio.RescaleIntensity(out_min_max=(0, 1)),
                    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
                ])

                preprocessed_subject = preprocessing_transforms(t1_subject)
                input_tensor = preprocessed_subject['t1']['data'][None]
                # input_tensor = input_tensor.permute(0, 1, 4, 2, 3)
                outputs = model(input_tensor)
                _, predicted_class = torch.max(outputs.data, 1)
                if predicted_class.item() == 1:
                    i+=1
                print(predicted_class.item())

print(f"correct: {i}")
print(f"incorrect: {j}")
"""


# =========================================================================================================================================
import torch
"""
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=3, stride=stride, padding=(1,1,1), bias=False),
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
        self.use_res_connect = self.stride == (1,1,1) and inp == oup

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
    def __init__(self, num_classes=2, input_shape=(1, 155, 240, 240), width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            [1, 16, 1, (1, 1, 1)],
            [6, 24, 2, (2, 2, 2)],
            [6, 32, 3, (2, 2, 2)],
            [6, 64, 4, (2, 2, 2)],
            [6, 96, 3, (1, 1, 1)],
            [6, 160, 3, (2, 2, 2)],
            [6, 320, 1, (1, 1, 1)],
        ]

        print("Input shape:", input_shape)

        # Extract the depth dimension from the input shape tuple
        sample_size = input_shape[1]

        assert sample_size % 16 == 0, "Input depth must be divisible by 16."

        print("Sample size:", sample_size)

        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(input_shape[0], input_channel, (1, 2, 2))]

        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.features.append(conv_1x1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
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

model = MobileNetV2(3, (1, 155, 240, 240), 1.)
print(model)
"""
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
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, (1, 1, 1)],
            [6, 24, 2, (2, 2, 2)],
            [6, 32, 3, (2, 2, 2)],
            [6, 64, 4, (2, 2, 2)],
            [6, 96, 3, (1, 1, 1)],
            [6, 160, 3, (2, 2, 2)],
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
            nn.Dropout(0.5),
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


model = MobileNetV2(num_classes=3, sample_size=240, width_mult=1.)
print(model)
"""

#=======================================================================================================================================

# This code is to output and download the augmented NiFTY files
"""
import os
import torchio as tio


rotation_transform = tio.transforms.RandomAffine(
    scales=(1, 1),
    degrees=(0, 0, 15),
    translation=(0, 0, 0),)
flipping_transform = tio.transforms.RandomFlip(axes=0, flip_probability=1.0)
elastic_transform = tio.transforms.RandomElasticDeformation(num_control_points=7, max_displacement=6)


preprocessing_transforms = tio.Compose([
    tio.RescaleIntensity(out_min_max=(0, 1)),
    tio.ZNormalization(masking_method=tio.ZNormalization.mean),
])



root_dir = "/Users/pesala/Documents/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training"
class_labels = {"LGG":0}
class_names = list(class_labels)
save_dir = "/Users/pesala/Documents/Augmented_brats/elastic"




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
                if classes == "LGG":
                    i+=1
                    T1_sample_path = os.path.join(subjects_path, file)
                    t1_subject = tio.Subject(t1=tio.ScalarImage(T1_sample_path), label=class_labels[classes])
                    # rotation augmentation
                    i+=1
                    rotation_subject = rotation_transform(t1_subject)
                    #preprocessed_subject = preprocessing_transforms(rotation_subject)
                    save_path = os.path.join(save_dir, f"{subjects}_rotation.nii.gz")
                    rotation_subject.t1.save(save_path)
                    # flipping augmentation
                    flipping_subject = flipping_transform(t1_subject)
                    save_path = os.path.join(save_dir, f"{subjects}_flip.nii.gz")
                    flipping_subject.t1.save(save_path)
                    # elastic transform augmentation
                    """
"""
                    elastic_subject = elastic_transform(t1_subject)
                    save_path = os.path.join(save_dir, f"{subjects}_elastic.nii.gz")
                    elastic_subject.t1.save(save_path)
"""
"""

    print(f"Total {classes} processed: {class_subjects_count}")
    print(i)

"""

#=======================================================================================================================================

# This code is to get the MobileNetV2 model structure with layerwise input and output values
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
            [6, 96, 2, (1, 1, 1)],
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
        for i, layer in enumerate(self.features):
            x = layer(x)
            print(f"Layer {i}: {x.shape}")
        x = F.avg_pool3d(x, x.data.size()[-3:])
        print(f"After avg_pool3d: {x.shape}")
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        print(f"Final output: {x.shape}")
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

# Example usage:
model = MobileNetV2(num_classes=3, sample_size=240, width_mult=1.0)
input_tensor = torch.randn(1, 1, 250, 240, 155)  # batch size of 1, 1 input channel, 250x240x155 spatial dimensions
output = model(input_tensor)
"""

#=======================================================================================================================================

# This code is to get the Resnet model structure with layerwise input and output values
"""
import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

class ModifiedResNet3D(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.6):
        super(ModifiedResNet3D, self).__init__()
        # Load the pre-trained ResNet 3D-18 model
        weights = R3D_18_Weights.KINETICS400_V1
        self.resnet3d = r3d_18(weights=weights)

        # Modify the first convolutional layer to take 1-channel (grey) input
        self.resnet3d.stem[0] = nn.Conv3d(
            1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

        # Adjust the fully connected layer
        num_features = self.resnet3d.fc.in_features
        self.resnet3d.fc = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.resnet3d(x)

def print_layer_shapes(model, input_size):
    def hook(module, input, output):
        class_name = module.__class__.__name__
        layer_name = module_names.get(module, 'layer')
        print(f"{layer_name: <15} | {str(output.shape): <30}")

    module_names = {}
    for name, module in model.named_modules():
        module_names[module] = name
        module.register_forward_hook(hook)

    # Create a dummy input tensor with the specified size
    dummy_input = torch.randn(*input_size)
    model(dummy_input)

# Define the model and input size
num_classes = 3
input_size = (1, 1, 155, 240, 240)  # Batch size of 1, 1 channel, depth 155, height 240, width 240
model = ModifiedResNet3D(num_classes)

# Print the output shapes of each layer
print_layer_shapes(model, input_size)
"""
