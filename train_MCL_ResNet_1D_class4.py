import os
import sys 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  
sys.path.append(os.path.dirname(os.path.abspath(__file__))) 
import numpy as np
import torch
import torch.nn as nn  
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ECG_dataset import load_and_process_data, package_dataset, split_dataset_by_patients, set_seed
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score , auc
import seaborn as sns
from model.MCL_ResNet1D_4 import ResNet1D, EXPERIMENTS
from para_define import LabelSmoothingLoss
from plot_training_curves import plot_training_curves, epoch_number, show_result_epoch
from tqdm import tqdm
import json
"""
Multi-class Classification Training Script for AMI and LVR Detection
"""

set_seed(42)
data, labels, patient_ids = load_and_process_data()
print(f"Dataset size: {data.shape[0]}")
print(f"Channels: {data.shape[1]}")
print(f"Data length: {data.shape[2]}")
print(f"Number of classes: {len(np.unique(labels))}")

X_train, y_train, X_val, y_val, X_test, y_test, patient_ids_test = split_dataset_by_patients(
    data, labels, patient_ids, test_size=0.1, val_size=0.1, random_state=42
)

train_dataset = [[X_train[i], y_train[i]] for i in range(len(X_train))]
val_dataset = [[X_val[i], y_val[i]] for i in range(len(X_val))]
test_dataset = [[X_test[i], y_test[i]] for i in range(len(X_test))]

print(f"Training set: {len(train_dataset)} ({len(train_dataset)/data.shape[0]:.1%})")
print(f"Validation set: {len(val_dataset)} ({len(val_dataset)/data.shape[0]:.1%})") 
print(f"Test set: {len(test_dataset)} ({len(test_dataset)/data.shape[0]:.1%})")

dataset, channels, data_length, classes = package_dataset(data, labels)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

experiment = 'ca_ms_v2'
cfg = EXPERIMENTS[experiment]

class CustomModel(ResNet1D):
    def __init__(self, in_channels, experiment):
        super().__init__(in_channels=in_channels, experiment=experiment)


model = CustomModel(in_channels=channels, experiment=experiment).to(device)


optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='max', 
    factor=0.5, 
    patience=3, 
)
criterion = LabelSmoothingLoss(smoothing=0.05)  



train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32,
                        num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32,
                        num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32,
                        num_workers=0)

output_dir = r'D:\AMI_LVR_Project\results_class4'

import shutil
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

train_acc_list = []
val_acc_list = []
test_acc_list = []  

train_loss_list = [] 
val_loss_list = []
test_loss_list = []


def train(epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}', unit='batch')
    for batch_data in progress_bar:
        data, target = batch_data
        data = data.to(device).float()
        target = target.to(device).long()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    progress_bar.set_postfix({
        'loss': f'{loss.item():.4f}',
        'acc': f'{100.*correct/total:.2f}%'
    })
    
    avg_loss = total_loss / len(train_loader)
    acc = 100. * correct / total
    train_acc_list.append(acc)
    train_loss_list.append(avg_loss)
    if epoch % show_result_epoch == 0:
        print(f'Epoch {epoch}:')
        print(f'Train Loss: {avg_loss:.4f} | Acc: {acc:.2f}%')
        val()
    

best_acc = 0
def val():
    global best_acc
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels = []
    probs = []

    with torch.no_grad():
        for val_data in val_loader:
            val_data_value, val_data_label = val_data
            val_data_value = val_data_value.to(device).float()
            val_data_label = val_data_label.to(device).long()
            outputs = model(val_data_value)
            
            loss = criterion(outputs, val_data_label)
            val_loss += loss.item()
            
            prob = F.softmax(outputs, dim=1)
            probs.extend(prob.cpu().numpy())
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(val_data_label.cpu().numpy())
            
            val_total += val_data_label.size(0)
            val_correct += (preds == val_data_label).sum().item()

    val_acc = round(100 * val_correct / val_total, 3)
    avg_val_loss = val_loss / len(val_loader)
    val_acc_list.append(val_acc)
    val_loss_list.append(avg_val_loss)
    scheduler.step(val_acc)
    print(f'Validation accuracy: {val_acc}%')
    
    if val_acc > best_acc:
        best_acc = val_acc
        model_path = os.path.join(output_dir, 'best_model.pth')
        os.makedirs(output_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f'Saved new best model with acc: {val_acc}%')


def final_test():
    class_names = ['NSTEMI non-LVR', 'NSTEMI LVR', 'STEMI non-LVR', 'STEMI LVR']
    model.load_state_dict(torch.load(
        os.path.join(output_dir, 'best_model.pth'),
        map_location=device
    ))
    model.eval()
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    probs = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            
            prob = F.softmax(outputs, dim=1)
            probs.extend(prob.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())
            
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
    
    test_acc = 100. * test_correct / test_total
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0,1,2,3])
    fpr, tpr, roc_auc = [], [], {}
    for i in range(4):
        fpr_i, tpr_i, _ = roc_curve((np.array(all_labels) == i).astype(int), np.array(probs)[:, i])
        fpr.append(fpr_i.tolist())
        tpr.append(tpr_i.tolist())
        roc_auc[f'class_{i}'] = auc(fpr_i, tpr_i)
    
    class_metrics = {}
    for i, name in enumerate(class_names):
        class_labels = (np.array(all_labels) == i).astype(int)
        class_preds = (np.array(all_preds) == i).astype(int)
        
        class_correct = np.sum((np.array(all_labels) == i) & (np.array(all_preds) == i))
        class_total = np.sum(np.array(all_labels) == i)
        class_accuracy = float(class_correct / class_total) if class_total > 0 else 0.0
        
        class_precision = float(precision_score(class_labels, class_preds, zero_division=0))
        class_recall = float(recall_score(class_labels, class_preds, zero_division=0))
        class_f1 = float(f1_score(class_labels, class_preds, zero_division=0))
        
        class_metrics[f'class_{i}'] = {
            'name': name,
            'accuracy': class_accuracy,
            'precision': class_precision,
            'recall': class_recall,
            'f1': class_f1,
            'auc': float(roc_auc[f'class_{i}']),
            'correct_predictions': int(class_correct),
            'total_samples': int(class_total)
        }
    
    test_eval_results = {
        'accuracy': float(test_acc / 100),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'fpr': [[float(x) for x in inner_list] for inner_list in fpr],
        'tpr': [[float(x) for x in inner_list] for inner_list in tpr],
        'roc_auc': {
            'macro': float(roc_auc_score(all_labels, probs, multi_class='ovr')),
            **{k: float(v) for k, v in roc_auc.items()}
        },
        'class_metrics': class_metrics
    }
    
    print(f"\nTest Evaluation:")
    print(f"\nTest Evaluation:")
    print(f"Accuracy: {test_acc:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print(f"Macro AUC: {test_eval_results['roc_auc']['macro']:.4f}")
    print(f"Per-class AUC:")
    for i, name in enumerate(class_names):
        print(f"{name}: {test_eval_results['roc_auc'][f'class_{i}']:.4f}")
    
    print(f"\nPer-class metrics:")
    for i, name in enumerate(class_names):
        class_labels = (np.array(all_labels) == i).astype(int)
        class_preds = (np.array(all_preds) == i).astype(int)
        
        class_correct = np.sum((np.array(all_labels) == i) & (np.array(all_preds) == i))
        class_total = np.sum(np.array(all_labels) == i)
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        
        class_precision = precision_score(class_labels, class_preds, zero_division=0)
        class_recall = recall_score(class_labels, class_preds, zero_division=0)
        class_f1 = f1_score(class_labels, class_preds, zero_division=0)
        
        print(f"{name}:")
        print(f"  Accuracy: {class_accuracy:.4f} ({class_correct}/{class_total})")
        print(f"  Precision: {class_precision:.4f}")
        print(f"  Recall: {class_recall:.4f}")
        print(f"  F1: {class_f1:.4f}")
  
    plt.figure(figsize=(10, 8))
    for i in range(4):
        fpr_i = test_eval_results['fpr'][i]
        tpr_i = test_eval_results['tpr'][i]
        roc_auc_i = test_eval_results['roc_auc'][f'class_{i}']
        plt.plot(fpr_i, tpr_i, lw=2.5, label=f'Class {i} (AUC = {roc_auc_i:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('4-class ROC Curve', fontsize=20)
    plt.legend(loc="lower right", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.savefig(os.path.join(output_dir, 'test_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Final Test Accuracy: {test_acc:.2f}%')

    
    with open(os.path.join(output_dir, 'test_eval_results.json'), 'w') as f:
        json.dump(test_eval_results, f, indent=4)

    plt.figure(figsize=(10,8))
    cm = np.array(test_eval_results['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            annot_kws={'size': 14})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix (Four-Class)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_confusion_matrix.png'))
    plt.close()
    


    train_samples = set([np.ascontiguousarray(x[0]).tobytes() for x in train_dataset])
    test_samples = set([np.ascontiguousarray(x[0]).tobytes() for x in test_dataset])
    overlap = len(train_samples & test_samples)
    print(f"\nOverlap between training and test sets: {overlap}/{len(test_samples)}")

    class_dist = np.bincount(all_labels)
    print(f"Test set class distribution: Class 0 - {class_dist[0]} | Class 1 - {class_dist[1]} | Class 2 - {class_dist[2]} | Class 3 - {class_dist[3]}")

for epoch in range(1, epoch_number+1):
    train(epoch)
final_test()

np.save(os.path.join(output_dir, 'train_acc_list.npy'), train_acc_list)
np.save(os.path.join(output_dir, 'val_acc_list.npy'), val_acc_list)
np.save(os.path.join(output_dir, 'train_loss_list.npy'), train_loss_list)
np.save(os.path.join(output_dir, 'val_loss_list.npy'), val_loss_list)


plot_training_curves(output_dir, train_acc_list, train_loss_list, val_acc_list, val_loss_list)
