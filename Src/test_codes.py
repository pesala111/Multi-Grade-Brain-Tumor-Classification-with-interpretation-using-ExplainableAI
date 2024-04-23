
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


# This code is to change the spational dimentions and voxel resolution (similar to previos code block but with different directories

"""


import torchio as tio
import nibabel as nib
import nilearn
import nilearn.image
import os

root_dir = '/Users/pesala/Documents/IXI (original+resampled)'
resized_folder_path = '/Users/pesala/Documents/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/Healthy'
#target_img = ''
i = 0

for t1_img in os.listdir(root_dir):
    if t1_img.endswith('resampled.nii.gz'):
        source_img = os.path.join(root_dir, t1_img)
        subject = tio.Subject(image=tio.ScalarImage(source_img))

        #resampled_img = nilearn.image.resample_to_img(source_img, target_img, interpolation='continuous',
        #                             copy=True, order='F', clip=False, fill_value=0, force_resample=False)


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
