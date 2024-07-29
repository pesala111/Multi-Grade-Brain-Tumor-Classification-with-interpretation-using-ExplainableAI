# Multi-Grade-Brain-Tumor-Classification-with-interpretation-using-ExplainableAI
This project focuses on classifying brain tumors using deep learning models and includes explainable AI (XAI) methods to interpret the model's predictions.
This repository contains the code for preprocessing data, models implemented using the Hydra configuration framework, results, and explainable methods for this project on brain tumor classification interpreting with explainable AI techniques. The project leverages deep learning models, particularly MobileNetV2, and utilizes the Captum library for model interpretability.


The top 5 best-performing model checkpoints are included in the Git repository. 
For the Explainable AI (XAI) methods, the winning model checkpoint 'MobileNetV2_2221311_0.3' was used.

# Repository Structure

- `Explainable_AI/`: Contains interpretation methods for the model.
  
- `Model_Checkpoints/`: Contains the top 5 best-performing model checkpoints.
- `Src/`: Contains the training pipeline, inference code, configuration files, and utility scripts.
  - `Model_Train.py`: Script for training the model.
  - `config.yaml`: Configuration files for the project.
  - `model_inference.py`: Script for running inference on new data.
  - `utils.py`: Utility functions used across the training pipeline.
- `notebooks/`: Contains preprocessing scripts used for preparing the dataset before training.
  - `Change_MRI_Orientation.py`: Script to change the orientation of MRI images.
  - `Change_Orientation_using_FSL.sh`: Shell script for changing orientation using FSL tools.
  - `Check_Spatial_Resolution.py`: Script to check the spatial resolution of images.
  - `Check_Voxel_Resolution.py`: Script to check the voxel resolution of images.
  - `change_spatial_resolution.py`: Script to change the spatial resolution.
  - `change_voxel_resolution.py`: Script to change the voxel resolution.
  - `skull_strip_using_FSL.sh`: Shell script for skull stripping using FSL tools.
- `README.md`: This file.
- `requirements.txt`: List of dependencies required to run the project.

# Requirements
```
python==3.11.6
numpy==1.26.4
scikit-learn==1.4.2
seaborn==0.13.2
torch==2.3.1
torchvision==0.18.1
torchio==0.19.7
captum==0.7.0
matplotlib==3.8.4
scipy==1.11.3
hydra-core==1.3.2
omegaconf==2.3.0
```
# Dataset
The dataset used in this project is stored in Google Drive. Follow the link below to download the data.

Download the dataset:
```bash
https://drive.google.com/drive/folders/1nWwmsVyuU2A5eQVBD9IrsJ857PsIclXA
```
This dataset consists of three classes:

- High-Grade Glioma (HGG)

- Low-Grade Glioma (LGG)

- Healthy Brains




