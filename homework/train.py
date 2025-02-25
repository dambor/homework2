import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import ClassificationLoss, load_model, save_model
from .utils import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    **kwargs,
):
    # Determine device: cuda, mps, or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create a log directory for TensorBoard logs and checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = tb.SummaryWriter(log_dir)

    # Load model using defaults expected by the grader, and move it to device
    model = load_model(model_name, **kwargs)
    model = model.to(device)

    # Create loss function and optimizer
    loss_func = ClassificationLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Load training and validation data (expects directories "classification_data/train" and "classification_data/val")
    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("classification_data/val", shuffle=False, batch_size=batch_size, num_workers=2)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # Main training loop
    for epoch in range(num_epoch):
        # Clear metrics at the start of each epoch
        metrics["train_acc"].clear()
        metrics["val_acc"].clear()

        model.train()
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()
            outputs = model(img)
            loss = loss_func(outputs, label)
            loss.backward()
            optimizer.step()

            # Calculate training accuracy for the current batch
            _, preds = torch.max(outputs, dim=1)
            batch_acc = (preds == label).float().mean().item()
            metrics["train_acc"].append(batch_acc)

            global_step += 1

        # Evaluate on validation set
        model.eval()
        with torch.inference_mode():
            for img, label in val_data:
                img, label = img.to(device), label.to(device)
                outputs = model(img)
                _, preds = torch.max(outputs, dim=1)
                batch_acc = (preds == label).float().mean().item()
                metrics["val_acc"].append(batch_acc)

        # Compute average accuracies for the epoch
        epoch_train_acc = np.mean(metrics["train_acc"])
        epoch_val_acc = np.mean(metrics["val_acc"])

        # Log accuracies to TensorBoard
        logger.add_scalar("train/accuracy", epoch_train_acc, epoch)
        logger.add_scalar("val/accuracy", epoch_val_acc, epoch)

        # Print progress at epoch 1, last epoch, and every 10 epochs
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # Save the final model using the provided save_model function.
    # This will write a file (e.g., "linear.th") in the same directory as models.py.
    save_model(model)

    # Optionally, also save a copy of the model state in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # Optional: add any additional model hyperparameters here (e.g., --num_layers)
    train(**vars(parser.parse_args()))
