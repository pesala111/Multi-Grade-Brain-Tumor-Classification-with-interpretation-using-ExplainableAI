import torch
import torch.nn as nn
import torchio as tio
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
import utils
from omegaconf import DictConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import numpy as np
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose

with initialize(config_path="", job_name="my_app"):
    cfg = compose(config_name="config")

def main(cfg:DictConfig):
    #root_dir = cfg.data.root_dir
    num_classes= 2
    input_shape= (1, 155, 240, 240)
    #num_classes = int(cfg.model.num_classes)
    #input_shape = tuple(cfg.model.input_shape)
    seed = cfg.data.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    class_labels = cfg.data.class_labels
    class_names = list(class_labels)
    batch_size = cfg.data.batch_size
    arch = cfg.model.architecture
    optimizer = cfg.training.optimizer
    num_epochs = cfg.training.num_epochs
    learning_rate = cfg.training.learning_rate
    momentum = cfg.training.momentum
    weight_decay = cfg.training.weight_decay
    momentum = cfg.training.momentum
    patience = cfg.training.patience

    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity, name=cfg.wandb.run_name, tags=["MobileNetV2"],
               config={ "learning_rate": learning_rate,
                        "momentum": momentum,
                        "weight_decay": weight_decay,
                        "batch_size": batch_size,
                        "num_epochs": num_epochs,
                        "patience" : patience,
                        "optimizer" : optimizer,
                        "architecture": arch,
                        "transforms": ["RescaleIntensity", "ZNormalization"],
                        "random_seed": 42
                    })

    preprocessing_transforms = tio.Compose([
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ])

    subjects_list = utils.create_subjects_list(root_dir, class_labels, preprocessing_transforms)
    dataset = tio.SubjectsDataset(subjects_list)
    train_size = int(0.7 * len(dataset))
    val_test_size = len(dataset) - train_size
    test_size = val_test_size // 2
    val_size = val_test_size - test_size
    train_dataset, val_test_dataset = torch.utils.data.random_split(dataset, [train_size, val_test_size])
    val_dataset, test_dataset = torch.utils.data.random_split(val_test_dataset, [val_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if arch == "Resnet3D":
      model = utils.ResNet3D(num_classes=num_classes).to(device)
    elif arch == "Simple3dCNN":
      model = utils.Simple3DCNN(num_classes=num_classes).to(device)
    elif arch == "MobilenetV2_3D":
      model = utils.MobileNetV2_3D(input_shape, num_classes).to(device)

    criterion = cfg.model.criterion

    if optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4, verbose=True)

    model_description = str(model)
    wandb.log({"model_description": model_description})
    wandb.watch(model, criterion, log="all", log_freq=10)

    num_epochs = cfg.training.num_epochs
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(num_epochs):
        train_loss, train_accuracy = utils.train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = utils.validate_model(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                   "val_loss": val_loss, "val_accuracy": val_accuracy})
        scheduler.step(val_loss)

    # Testing the model
    test_loss, test_accuracy = utils.test_model(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})

    # Saving the model
    model_path = f"brain_tumor_model_{test_accuracy:.2f}%.pth"
    torch.save(model.state_dict(), model_path)
    wandb.save(model_path)


    # Plotting Training Curves
    utils.plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    # Evaluation: Confusion Matrix and ROC Curve
    probabilities, predictions, ground_truth = utils.get_predictions(model, test_loader, device)
    confusion_matrix = utils.plot_confusion_matrix(ground_truth, predictions, class_labels)
    roc_curve = utils.plot_roc_curve(ground_truth, predictions, "MobileNetV2")


    wandb.log({"confusion_mat" : wandb.plot.confusion_matrix(probs=None,
                        y_true=ground_truth, preds=predictions,
                        class_names=class_names)})
    wandb.log({"roc" : wandb.plot.roc_curve( ground_truth, probabilities,
                        labels=None, classes_to_plot=None)})

    wandb.finish()
if __name__=='__main__':
    main(cfg)


