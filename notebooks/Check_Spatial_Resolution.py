import os
import nibabel as nib

root_dir = '/Users/pesala/Documents/MICCAI_BraTS_2019_Data_Training/MICCAI_BraTS_2019_Data_Training/Healthy'
for t1_MRI in os.listdir(root_dir):
    if t1_MRI.endswith('T1.nii.gz'):
        t1_MRI_path = os.path.join(root_dir, t1_MRI)
        img = nib.load(t1_MRI_path)
        height, width, depth = img.shape
        print(f'height={height}, width={width}, depth={depth}')
        #data_type = img.get_data_dtype()
        #print('Data type:', data_type)




"""
for classes in os.listdir(root_dir):
    classes_path = os.path.join(root_dir, classes)
    if not os.path.isdir(classes_path):
        continue
    for subjects in os.listdir(classes_path):
        subjects_path = os.path.join(classes_path, subjects)
        if not os.path.isdir(subjects_path):
            continue
        for t1_MRI in os.listdir(subjects_path):
            if t1_MRI.endswith('_t1.nii.gz'):
                t1_MRI_path = os.path.join(subjects_path, t1_MRI)
                img = nib.load(t1_MRI_path)
                height, width, depth = img.shape
                #print(f'height={height}, width={width}, depth={depth}')
                data_type = img.get_data_dtype()
                print('Data type:', data_type)
"""