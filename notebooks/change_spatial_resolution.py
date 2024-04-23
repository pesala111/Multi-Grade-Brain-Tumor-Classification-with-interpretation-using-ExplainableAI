import torchio as tio
import os

root_dir = '/Users/pesala/Documents/Healthy_axial'
target_shape = (240, 240, 155)
crop_or_pad = tio.CropOrPad(target_shape)

resized_folder_path = '/Users/pesala/Documents/Healthy_axial/resampled'

i = 0

for t1_img in os.listdir(root_dir):
    if t1_img.endswith('T1.nii.gz'):
        file_path = os.path.join(root_dir, t1_img)
        subject = tio.Subject(image=tio.ScalarImage(file_path))

        transformed_image = crop_or_pad(subject) # changing the spatial resolution


        i += 1
        subject_id = f"healthy_T1_{i}"
        subject_dir = os.path.join(resized_folder_path, subject_id)
        os.makedirs(subject_dir, exist_ok=True)

        output_file_path = os.path.join(subject_dir, t1_img)
        transformed_image['image'].save(output_file_path)
        #print(i)
        volume_resolution = transformed_image.shape
        print(f"{volume_resolution[0]}, {volume_resolution[1]}, {volume_resolution[2]}, {volume_resolution[3]}")
