import torch
import torchio as tio
from utils import MobileNetV2
import os


# Model loading
model = MobileNetV2(num_classes=3, sample_size=240, width_mult=1.)
model_path = "/Users/pesala/Downloads/brain tumor trained paths/brain_tumor_model_87.83%_final.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

root_dir = "/Users/pesala/Documents/MICCAI_BraTS_inference_dataset/LGG"
original_class = "LGG"

for t1_img_folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, t1_img_folder)
        if not os.path.isdir(folder_path):
            continue
        for t1_img in os.listdir(folder_path):
            if t1_img.endswith('t1.nii.gz') or t1_img.endswith('resampled.nii.gz'):
                T1_sample_path = os.path.join(folder_path, t1_img)

                t1_subject = tio.Subject(t1=tio.ScalarImage(T1_sample_path))
                preprocessing_transforms = tio.Compose([
                    tio.RescaleIntensity(out_min_max=(0, 1)),
                    tio.ZNormalization(masking_method=tio.ZNormalization.mean)])

                preprocessed_subject = preprocessing_transforms(t1_subject)
                input_tensor = preprocessed_subject['t1']['data'][None]
                print(input_tensor.shape)
                # input_tensor = input_tensor.permute(0, 1, 4, 2, 3)
                outputs = model(input_tensor)
                _, predicted_class = torch.max(outputs.data, 1)
                print(f"original class: {original_class} and predicted class: {predicted_class.item()}")


