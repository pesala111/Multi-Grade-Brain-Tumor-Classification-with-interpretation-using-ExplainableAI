import torchio as tio
import os

root_dir = '/Users/pesala/Documents/MICCAI_BraTS_inference_dataset/HGG'
target_shape = (240, 240, 160)
crop_or_pad = tio.CropOrPad(target_shape)

resized_folder_path = '/Users/pesala/Documents/brats_depth_160/HGG'

i = 0
for subjects in os.listdir(root_dir):
    subjects_path = os.path.join(root_dir, subjects)
    if not os.path.isdir(subjects_path):
        continue
    for file in os.listdir(subjects_path):
        if file.endswith('t1.nii.gz') or file.endswith('resampled.nii.gz'):
            T1_sample_path = os.path.join(subjects_path, file)

            subject = tio.Subject(image=tio.ScalarImage(T1_sample_path))

            transformed_image = crop_or_pad(subject) # changing the spatial resolution

            i += 1
            subject_id = f"HGG_T1_{i}"
            subject_dir = os.path.join(resized_folder_path, subject_id)
            os.makedirs(subject_dir, exist_ok=True)

            output_file_path = os.path.join(subject_dir, os.path.basename(T1_sample_path))
            transformed_image['image'].save(output_file_path)

            volume_resolution = transformed_image['image'].data.shape
            print(f"{volume_resolution[0]}, {volume_resolution[1]}, {volume_resolution[2]}, {volume_resolution[3]}")
