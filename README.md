# ECG-based Left Ventricular Remodeling (LVR) Detection

This repository contains the implementation of a deep learning framework for detecting left ventricular remodeling (LVR) from 12-lead electrocardiogram (ECG) signals, along with saliency analysis for model interpretability.

## Overview

The project includes:
- **Multi-task Classification**: Binary (LVR vs. non-LVR) and 4-class (NSTEMI/STEMI with/without LVR) classification
- **Saliency Analysis**: Gradient-based visualization using SmoothGrad for model interpretability
- **Quantitative Analysis**: ICC (Intraclass Correlation Coefficient) and consistency metrics for lead importance evaluation

## Project Structure

```
.
├── model/
│   ├── MCL_ResNet1D_2.py      # 2-class ResNet1D model
│   └── MCL_ResNet1D_4.py      # 4-class ResNet1D model
├── data/                       # Data directory (user-provided)
├── ECG_dataset.py             # Data loading and preprocessing
├── para_define.py             # Loss functions and utilities
├── plot_training_curves.py    # Training visualization
├── train_MCL_ResNet_1D_class2.py   # 2-class training script
├── train_MCL_ResNet_1D_class4.py   # 4-class training script
└── Saliency_ana2_sys.py       # Saliency analysis and visualization
```

## Requirements

```
torch>=1.9.0
numpy>=1.19.0
matplotlib>=3.3.0
scipy>=1.5.0
scikit-learn>=0.24.0
seaborn>=0.11.0
tqdm>=4.50.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Format

The system expects preprocessed ECG data in `.npy` format:
- Four data files required:
  - `cut_NS_0.npy`: NSTEMI without LVR
  - `cut_NS_1.npy`: NSTEMI with LVR
  - `cut_ST_0.npy`: STEMI without LVR
  - `cut_ST_1.npy`: STEMI with LVR

Place data files in the `data/` directory or update paths in `ECG_dataset.py`.

## Usage

### 1. Training

**2-class classification (LVR detection):**
```bash
python train_MCL_ResNet_1D_class2.py
```

**4-class classification (fine-grained diagnosis):**
```bash
python train_MCL_ResNet_1D_class4.py
```

Trained models will be saved to `output_dir` specified in each training script.

### 2. Saliency Analysis

Run saliency analysis on the trained model:
```bash
python Saliency_ana2_sys.py
```

This will:
- Generate saliency maps for test samples
- Perform quantitative consistency analysis
- Output visualizations to `output_saliencyana2/`

## Key Features

### Patient-level Data Splitting
Prevents data leakage by ensuring samples from the same patient are not split across train/validation/test sets.

### SmoothGrad Saliency
Reduces noise in gradient-based saliency maps by averaging over multiple noise-perturbed inputs.

### Quantitative Metrics
- **ICC**: Measures consistency of lead importance across samples
- **ANOVA**: Tests significance of top-3 leads vs. others
- **Consistency Rate**: Percentage of samples with key leads in top-3


## Citation

If you use this code in your research, please cite:

```
[To be added upon publication]
```

## License

[To be specified]

## Contact

For questions or issues, please open an issue in this repository.
