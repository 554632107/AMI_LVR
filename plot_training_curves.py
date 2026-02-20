import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = 'results_class2'
epoch_number = 50
show_result_epoch = 1

def plot_training_curves(output_dir, train_acc_list, train_loss_list, val_acc_list, val_loss_list):

    # Create unified canvas
    plt.figure(figsize=(12, 5))

    # Accuracy curve subplot
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_list, color='#2ca02c', marker='o', linestyle='-', linewidth=1.5, markersize=4, label='Training acc')
    plt.plot(val_acc_list, color='#1f77b4', marker='^', linestyle='--', linewidth=1.5, markersize=5, label='Validation acc')

    plt.xlim(0, epoch_number)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training & Validation Accuracy Curve')
    plt.legend(loc='lower right')

    # Loss curve subplot
    plt.subplot(1, 2, 2)
    plt.plot(train_loss_list, color='#d62728', marker='o', linestyle='-', linewidth=1.5, markersize=4, label='Training loss')
    plt.plot(val_loss_list, color='#9467bd', marker='^', linestyle='--', linewidth=1.5, markersize=5, label='Validation loss')

    plt.xlim(0, epoch_number)
    plt.title('Training & Validation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')  
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'ACC_Loss_Curve.png'))
    plt.close()