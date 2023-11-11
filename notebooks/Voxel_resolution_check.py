import nibabel as nib
import os

root_dir = '/home/pesala/Downloads/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/LGG'
for category in os.listdir(root_dir):
    sub_folder = os.path.join(root_dir, category)
    for img in os.listdir(sub_folder):
        nifti_img = os.path.join(sub_folder, img)
        final_img = nib.load(nifti_img)
        voxel_size = final_img.header.get_zooms()
        print(f"Resolution (voxel size) of the image: {voxel_size}")