
import nibabel as nib
import os
i = 0
sub_folder = '/Users/pesala/Documents/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/Healthy'
for img in os.listdir(sub_folder):
    i += 1
    nifti_img = os.path.join(sub_folder, img)
    if img.endswith('T1.nii.gz'):
        final_img = nib.load(nifti_img)
        voxel_size = final_img.header.get_zooms()
        print(f"{i} Resolution (voxel size) of the image: {voxel_size}")


"""
import nibabel as nib
import os

root_dir = '/Users/pesala/Documents/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/LGG'

for category in os.listdir(root_dir):
    sub_folder = os.path.join(root_dir, category)
    for img in os.listdir(sub_folder):
        nifti_img = os.path.join(sub_folder, img)
        final_img = nib.load(nifti_img)
        voxel_size = final_img.header.get_zooms()
        print(f"Resolution (voxel size) of the image: {voxel_size}")
"""