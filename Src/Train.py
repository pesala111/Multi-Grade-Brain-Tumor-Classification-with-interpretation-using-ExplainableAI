import torch
import torch.nn as nn
import torchio as tio
from torch.optim.lr_scheduler import ReduceLROnPlateau
import utils
import wandb
import hydra
from omegaconf import DictConfig

@hydra.main(config_path="config.yaml", config_name="config")
def main(cfg:DictConfig):
    # Hydra configuration
    batch_size = cfg.training.batch_size
    root_dir = cfg.data.root_dir
    class_labels = cfg.data.class_labels
    batch_size = cfg.data.batch_size
    num_classes = cfg.model.num_classes
    num_epochs = cfg.training.num_epochs
    learning_rate = cfg.training.learning_rate
    momentum = cfg.training.momentum
    weight_decay = cfg.training.weight_decay
    patience = cfg.training.patience
    wandb.init(project="cfg.wandb.project", entity="cfg.wandb.entity")
    wandb.config = {
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "batch_size": batch_size}
    #class_labels = {'HGG': 0, 'LGG': 1}

    # Transforms
    preprocessing_transforms = tio.Compose([
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.ZNormalization(masking_method=tio.ZNormalization.mean),
    ])

    subjects_list = utils.create_subjects_list(root_dir, class_labels, preprocessing_transforms)

    # Data Splitting
    dataset = tio.SubjectsDataset(subjects_list)
    train_size = int(0.7 * len(dataset))
    val_test_size = len(dataset) - train_size
    test_size = val_test_size // 2
    val_size = val_test_size - test_size
    train_dataset, val_test_dataset = torch.utils.data.random_split(dataset, [train_size, val_test_size])
    val_dataset, test_dataset = torch.utils.data.random_split(val_test_dataset, [val_size, test_size])

    # DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model, loss, and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = utils.ResNet3D(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=patience, verbose=True)

    # Training loop
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

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                   "val_loss": val_loss, "val_accuracy": val_accuracy})
        scheduler.step(val_loss)

    # Testing the model
    test_loss, test_accuracy = utils.test_model(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})

    # Saving the model
    torch.save(model.state_dict(), 'brain_tumor_model.pth')

    # Plotting Training Curves
    utils.plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies)

    # Evaluation: Confusion Matrix and ROC Curve
    predictions, true_labels = utils.get_predictions(model, test_loader, device)
    utils.plot_confusion_matrix(true_labels, predictions, class_labels)
    utils.plot_roc_curve(true_labels, predictions, "ResNet3D")
    """
    there is error in the logic:
    wandb.log({"confusion_matrix": wandb.Image(plot_confusion_matrix),
               "roc_curve": wandb.Image(plot_roc_curve)})
    """
    wandb.finish()