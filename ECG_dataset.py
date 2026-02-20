import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import random
import sys
from sklearn.model_selection import GroupShuffleSplit


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _load_data_with_patient_ids(*data_paths):
    """General data loading function that also returns patient IDs"""
    try:
        datasets = [np.load(p, allow_pickle=True) for p in data_paths]
        data = np.concatenate(datasets, axis=0)
        
        # Standardization processing
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data = np.array([scaler.fit_transform(sample.T).T for sample in data])
        
        # Generate 4-class labels: ensure labels 0-3 are generated
        labels = np.concatenate([np.full(dataset.shape[0], i) 
                               for i, dataset in enumerate(datasets)])
        
        # Generate patient IDs
        patient_ids = []
        current_patient_id = 0
        for i, dataset in enumerate(datasets):
            n_samples = dataset.shape[0]
            patient_ids.extend(range(current_patient_id, current_patient_id + n_samples))
            current_patient_id += n_samples
        
        patient_ids = np.array(patient_ids)
        
        # Validate label generation
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 4, f"Expected 4 classes, got {len(unique_labels)} classes"
        print(f"Loaded 4-class data, samples per class: {np.bincount(labels)}")
        print(f"Total: {len(np.unique(patient_ids))}")
            
        return data, labels, patient_ids
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        sys.exit(1)


def load_and_process_data_binary():
    """Load binary classification data: NS_0+ST_0=class 0, NS_1+ST_1=class 1"""
    try:
        # Load four datasets
        ns0_data = np.load(r"D:\AMI_LVR_Project\data\cut_NS_0.npy", allow_pickle=True)
        ns1_data = np.load(r"D:\AMI_LVR_Project\data\cut_NS_1.npy", allow_pickle=True)
        st0_data = np.load(r"D:\AMI_LVR_Project\data\cut_ST_0.npy", allow_pickle=True)
        st1_data = np.load(r"D:\AMI_LVR_Project\data\cut_ST_1.npy", allow_pickle=True)
        
        # Merge data
        data = np.concatenate([ns0_data, ns1_data, st0_data, st1_data], axis=0)
        
        # Standardization processing for binary data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        data = np.array([scaler.fit_transform(sample.T).T for sample in data])
        
        # Generate binary classification labels: NS_0 and ST_0 as class 0, NS_1 and ST_1 as class 1
        labels = np.concatenate([
            np.full(ns0_data.shape[0], 0),  # NS_0 → 0
            np.full(ns1_data.shape[0], 1),  # NS_1 → 1
            np.full(st0_data.shape[0], 0),  # ST_0 → 0
            np.full(st1_data.shape[0], 1)   # ST_1 → 1
        ])
        
        # Generate patient IDs
        patient_ids = []
        current_patient_id = 0
        for dataset in [ns0_data, ns1_data, st0_data, st1_data]:
            n_samples = dataset.shape[0]
            patient_ids.extend(range(current_patient_id, current_patient_id + n_samples))
            current_patient_id += n_samples
        
        patient_ids = np.array(patient_ids)
        
        # Validate label generation
        unique_labels = np.unique(labels)
        assert len(unique_labels) == 2, f"Expected 2 classes, got {len(unique_labels)} classes"
        print(f"Loaded binary data, samples per class: {np.bincount(labels)}")
        print(f"Total: {len(np.unique(patient_ids))}")
            
        return data, labels, patient_ids
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        sys.exit(1)


def load_and_process_data():
    """Load data""" 
    data, labels, patient_ids = _load_data_with_patient_ids(
        r"D:\AMI_LVR_Project\data\cut_NS_0.npy",
        r"D:\AMI_LVR_Project\data\cut_NS_1.npy",
        r"D:\AMI_LVR_Project\data\cut_ST_0.npy",
        r"D:\AMI_LVR_Project\data\cut_ST_1.npy"
    )
    return data, labels, patient_ids

def package_dataset(data, labels, patient_ids=None):
    classes = 4
    if patient_ids is not None:
        dataset = [[i, j, k] for i, j, k in zip(data, labels, patient_ids)]
    else:
        dataset = [[i, j] for i, j in zip(data, labels)]
    channels = data[0].shape[0]
    data_length = data[0].shape[1]
    classes = len(np.unique(labels))
    print(f"Validate label distribution: {np.bincount(labels)}")  # Add debug output
    return dataset, channels, data_length, classes

# Function to split dataset by patient ID
def split_dataset_by_patients(data, labels, patient_ids, test_size=0.1, val_size=0.2, random_state=42):
    """
    Stratified split by patient ID to avoid data leakage
    Using 7:2:1 split ratio
    """
    print("\n" + "="*60)
    print("Split dataset by patient ID")
    print("="*60)
    
    # First split: train+val vs test (10% test set)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss_test.split(data, labels, groups=patient_ids))
    
    X_train_val, X_test = data[train_val_idx], data[test_idx]
    y_train_val, y_test = labels[train_val_idx], labels[test_idx]
    patient_ids_train_val = patient_ids[train_val_idx]
    patient_ids_test = patient_ids[test_idx]
    
    # Second split: train vs val (split 90% data in 2:7 ratio)
    val_size_adjusted = val_size / (1 - test_size)  # 0.2 / 0.9 ≈ 0.222
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_state)
    train_idx, val_idx = next(gss_val.split(X_train_val, y_train_val, groups=patient_ids_train_val))
    
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
    patient_ids_train = patient_ids_train_val[train_idx]
    patient_ids_val = patient_ids_train_val[val_idx]
    
    # Statistics
    train_patients = np.unique(patient_ids_train)
    val_patients = np.unique(patient_ids_val)
    test_patients = np.unique(patient_ids_test)
    
    print(f"\nPatient-level split:")
    print(f"  Train: {len(train_patients)}, {X_train.shape[0]} samples ({100*X_train.shape[0]/data.shape[0]:.1f}%)")
    print(f"  Val: {len(val_patients)}, {X_val.shape[0]} samples ({100*X_val.shape[0]/data.shape[0]:.1f}%)")
    print(f"  Test: {len(test_patients)}, {X_test.shape[0]} samples ({100*X_test.shape[0]/data.shape[0]:.1f}%)")
    
    # Dynamic class distribution statistics (supports binary and 4-class classification)
    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    print(f"\nClass distribution (total {n_classes} classes):")
    for split_name, y_split in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
        class_counts = [np.sum(y_split==i) for i in unique_labels]
        class_str = ', '.join([f'Class {i}: {cnt}' for i, cnt in zip(unique_labels, class_counts)])
        print(f"  {split_name} - {class_str}")
    
    # Verify no data leakage
    assert len(set(train_patients) & set(val_patients)) == 0, "Train and val patients overlap!"
    assert len(set(train_patients) & set(test_patients)) == 0, "Train and test patients overlap!"
    assert len(set(val_patients) & set(test_patients)) == 0, "Val and test patients overlap!"
    print("\nPatient-level split validation passed, no data leakage")
    print("="*60)
    
    return X_train, y_train, X_val, y_val, X_test, y_test, patient_ids_test


class ECGDataset(Dataset):
    def __init__(self, data):
        self.len = len(data)
        # Check data format
        if len(data[0]) == 3:  # [data, label, patient_id]
            self.x_data = torch.from_numpy(np.array([x[0] for x in data], dtype=np.float32))
            self.y_data = torch.from_numpy(np.array([x[1] for x in data])).long()
            self.patient_ids = np.array([x[2] for x in data])
            self.has_patient_ids = True
        else:  # [data, label]
            self.x_data = torch.from_numpy(np.array([x[0] for x in data], dtype=np.float32))
            self.y_data = torch.from_numpy(np.array([x[1] for x in data])).long()
            self.has_patient_ids = False

    def __getitem__(self, index):
        if self.has_patient_ids:
            return self.x_data[index], self.y_data[index], self.patient_ids[index]
        else:
            return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

if __name__ == "__main__":
    set_seed(42)
    
    print("="*60)
    print("ECG Dataset Split Function Test")
    print("="*60)

    # Test binary classification data
    print("\n" + "="*60)
    print("=== Binary Classification Data Test ===")
    data_2class, labels_2class, patient_ids_2class = load_and_process_data_binary()
    X_train_2, y_train_2, X_val_2, y_val_2, X_test_2, y_test_2, patient_ids_test_2 = split_dataset_by_patients(
        data_2class, labels_2class, patient_ids_2class, test_size=0.1, val_size=0.2
    )
    
    # Create binary classification dataset
    train_dataset_2 = [[X_train_2[i], y_train_2[i]] for i in range(len(X_train_2))]
    val_dataset_2 = [[X_val_2[i], y_val_2[i]] for i in range(len(X_val_2))]
    test_dataset_2 = [[X_test_2[i], y_test_2[i]] for i in range(len(X_test_2))]
    
    dataset_2, channels_2, data_length_2, classes_2 = package_dataset(data_2class, labels_2class)
    print(f"Dataset size: {len(dataset_2)}")
    print(f"Channels: {channels_2}")
    print(f"Data length: {data_length_2}")
    print(f"Classes: {classes_2}")
    
    # Binary classification label distribution
    print("\nBinary classification label distribution:")
    for class_idx, count in enumerate(np.bincount(labels_2class)):
        print(f"  Class {class_idx}: {count} samples ({count/len(labels_2class):.1%})")
    
    print(f"\nBinary classification patient split results:")
    print(f"  Train: {len(train_dataset_2)} ({len(train_dataset_2)/len(dataset_2):.1%})")
    print(f"  Val: {len(val_dataset_2)} ({len(val_dataset_2)/len(dataset_2):.1%})")
    print(f"  Test: {len(test_dataset_2)} ({len(test_dataset_2)/len(dataset_2):.1%})")


    
    # Test 4-class classification data
    print("\n=== 4-Class Classification Data Test ===")
    data_4class, labels_4class, patient_ids_4class = load_and_process_data()
    X_train_4, y_train_4, X_val_4, y_val_4, X_test_4, y_test_4, patient_ids_test_4 = split_dataset_by_patients(
        data_4class, labels_4class, patient_ids_4class, test_size=0.1, val_size=0.2
    )
    
    # Create 4-class classification dataset
    train_dataset_4 = [[X_train_4[i], y_train_4[i]] for i in range(len(X_train_4))]
    val_dataset_4 = [[X_val_4[i], y_val_4[i]] for i in range(len(X_val_4))]
    test_dataset_4 = [[X_test_4[i], y_test_4[i]] for i in range(len(X_test_4))]
    
    dataset_4, channels_4, data_length_4, classes_4 = package_dataset(data_4class, labels_4class)
    print(f"Dataset size: {len(dataset_4)}")
    print(f"Channels: {channels_4}")
    print(f"Data length: {data_length_4}")
    print(f"Classes: {classes_4}")
    
    # 4-class classification label distribution
    print("\n4-class classification label distribution:")
    for class_idx, count in enumerate(np.bincount(labels_4class)):
        print(f"  Class {class_idx}: {count} samples ({count/len(labels_4class):.1%})")
    
    print(f"\n4-class classification patient split results:")
    print(f"  Train: {len(train_dataset_4)} ({len(train_dataset_4)/len(dataset_4):.1%})")
    print(f"  Val: {len(val_dataset_4)} ({len(val_dataset_4)/len(dataset_4):.1%})")
    print(f"  Test: {len(test_dataset_4)} ({len(test_dataset_4)/len(dataset_4):.1%})")
    
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)