import numpy as np
import matplotlib.pyplot as plt

def remove_outliers(data):
    data = np.array(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered = np.where((data >= lower_bound) & (data <= upper_bound), data, np.nan)
    return filtered

def plot_losses(train_losses, val_losses):
    train_losses_filtered = remove_outliers(train_losses)
    val_losses_filtered = remove_outliers(val_losses)

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses_filtered, label='Train Loss', marker='o')
    plt.plot(val_losses_filtered, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss (Outliers Removed)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_acc(train_acc, val_acc):
    plt.figure(figsize=(8, 6))
    plt.plot(train_acc, label='Train Acc', marker='o')
    plt.plot(val_acc, label='Validation Acc', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_map(train_maps, val_maps):
    plt.figure(figsize=(8, 6))
    plt.plot(train_maps, label='Train mAP', marker='o')
    plt.plot(val_maps, label='Validation mAP', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('mAP (mean Average Precision)')
    plt.title('Validation mAP')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
