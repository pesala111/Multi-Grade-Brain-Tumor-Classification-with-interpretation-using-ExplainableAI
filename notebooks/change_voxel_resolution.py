import torchio as tio
import os

root_dir = '/Users/pesala/Documents/Healthy_axial'

target_spacing = (1.0, 1.0, 1.0)
resample = tio.Resample(target_spacing)
resized_folder_path = '/Users/pesala/Documents/Healthy_axial/resampled'

i = 0

for t1_img in os.listdir(root_dir):
    if t1_img.endswith('T1.nii.gz'):
        file_path = os.path.join(root_dir, t1_img)
        subject = tio.Subject(image=tio.ScalarImage(file_path))

        transformed_image = resample(subject) # changing the voxel resolution


        i += 1
        subject_id = f"healthy_T1_{i}"
        subject_dir = os.path.join(resized_folder_path, subject_id)
        os.makedirs(subject_dir, exist_ok=True)

        output_file_path = os.path.join(subject_dir, t1_img)
        transformed_image['image'].save(output_file_path)
        #print(i)
        voxel_resolution = transformed_image.spacing
        print(f"{i}: {voxel_resolution}")
