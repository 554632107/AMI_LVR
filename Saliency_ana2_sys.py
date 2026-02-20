import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from pathlib import Path

from model.MCL_ResNet1D_2 import ResNet1D, EXPERIMENTS
from ECG_dataset import load_and_process_data, split_dataset_by_patients, set_seed
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 30  
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 25
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['legend.fontsize'] = 20

# Standard 12-lead names
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

LVR_FEATURES = {
    'qrs_complex': (130, 170)
}

def compute_saliency_optimized(model: nn.Module,
                               inputs: torch.Tensor,
                               targets: torch.Tensor,
                               use_smoothgrad: bool = True,
                               num_samples: int = 10,
                               noise_level: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Saliency implementation with SmoothGrad support.
    Returns: channel_importance (B, 12), time_importance (B, T), saliency (B, 12, T)
    """
    model.eval()
    inputs = inputs.to(device)
    targets = targets.to(device)
    inputs.requires_grad_(True)

    if use_smoothgrad:
        saliency_acc = torch.zeros_like(inputs)
        for _ in range(num_samples):
            inputs.grad = None
            noise = torch.randn_like(inputs) * noise_level
            outputs = model(inputs + noise)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            model.zero_grad(set_to_none=True)
            loss.backward()
            assert inputs.grad is not None, "Gradient computation failed"
            saliency_acc += inputs.grad.abs()
        saliency = saliency_acc / num_samples
    else:
        inputs.grad = None
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        model.zero_grad(set_to_none=True)
        loss.backward()
        assert inputs.grad is not None, "Gradient computation failed"
        saliency = inputs.grad.abs()

    ch_imp = saliency.mean(dim=2)     # (B, 12)
    t_imp = saliency.mean(dim=1)      # (B, T)
    return ch_imp.detach(), t_imp.detach(), saliency.detach()

def visualize_saliency_for_sample(inputs: torch.Tensor,
                                  saliency_tuple: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                  sample_idx: int,
                                  pred_class: int,
                                  true_class: int,
                                  data_type: str,
                                  topk: int = 6,
                                  out_dir: str = 'results/saliencyana',
                                  vmin: Optional[float] = None,
                                  vmax: Optional[float] = None,
                                  save_idx: Optional[int] = None):
    """
    Generate visualization for a single sample using Saliency:
    1. Top-6 Leads waveforms with importance percentages
    2. Time-channel heatmap with standard lead names on y-axis
    3. Best Lead with R peak and temporal saliency
    
    Args:
        vmin, vmax: Colorbar min/max values for consistent scaling across samples
        save_idx: Global index for filename (if provided), otherwise uses sample_idx
    """
    ch_imp, t_imp, saliency = saliency_tuple

    x = inputs[sample_idx].detach().cpu().numpy()
    sal = saliency[sample_idx].detach().cpu().numpy()
    ch_imp_sample = ch_imp[sample_idx].detach().cpu().numpy()
    t_imp_sample = t_imp[sample_idx].detach().cpu().numpy()

    top_indices = torch.topk(ch_imp[sample_idx], k=min(topk, ch_imp_sample.shape[0])).indices.tolist()
    sorted_indices = sorted(top_indices, key=lambda idx: float(ch_imp_sample[idx]))
    best_idx: int = int(torch.topk(ch_imp[sample_idx], k=1).indices.item())
    best_lead_name = LEAD_NAMES[best_idx] if 0 <= best_idx < len(LEAD_NAMES) else str(best_idx)

    denom = ch_imp_sample.sum() + 1e-12
    ch_imp_pct = ch_imp_sample / denom
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    legend_items = []
    for i, lead_idx in enumerate(sorted_indices):
        lead_signal = x[lead_idx]
        norm_signal = (lead_signal - np.mean(lead_signal)) / (np.std(lead_signal) + 1e-6)
        lead_name = LEAD_NAMES[lead_idx] if 0 <= lead_idx < len(LEAD_NAMES) else str(lead_idx)
        line, = axes[0].plot(
            norm_signal + i * 3,
            label=f'Lead {lead_name} (imp: {ch_imp_pct[lead_idx]*100:.2f}%)',
            linewidth=1.5, alpha=0.85
        )
        legend_items.append((line, float(ch_imp_sample[lead_idx])))
    y_ticks = [i * 3 for i in range(len(sorted_indices))]
    y_labels = [LEAD_NAMES[idx] if 0 <= idx < len(LEAD_NAMES) else str(idx) for idx in sorted_indices]
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels(y_labels, fontsize=30, fontweight='bold')
    axes[0].set_ylabel('Lead', fontsize=30, fontweight='bold')
    axes[0].set_title(f'Top-6 Leads - Sample {sample_idx} (Pred: {pred_class}, True: {true_class}, Type: {data_type})', fontsize=30, fontweight='bold')
    handles_sorted = [h for (h, _) in sorted(legend_items, key=lambda t: t[1], reverse=True)]
    labels_sorted = [h.get_label() for h in handles_sorted]
    axes[0].legend(handles_sorted, labels_sorted, loc='upper right', fontsize=20)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, x.shape[1])
    axes[0].tick_params(axis='both', labelsize=25)
    for tick in axes[0].get_xticklabels() + axes[0].get_yticklabels():
        tick.set_fontweight('bold')

    im = axes[1].imshow(sal, aspect='auto', cmap='viridis', extent=[0, x.shape[1], 0, x.shape[0]],
                        vmin=vmin, vmax=vmax)
    axes[1].set_xlim(0, x.shape[1])
    
    phase_colors = {
        'p_wave': 'cyan',
        'pr_segment': 'magenta',
        'qrs_complex': 'lime',
        'st_segment': 'yellow',
        't_wave': 'white'
    }
    phase_display = {
        'p_wave': 'P',
        'qrs_complex': 'QRS',
        'st_segment': 'ST',
        't_wave': 'T',
        'pathological_q': 'Q'
    }
    axes[1].set_title('Time-Channel Importance', fontsize=30, fontweight='bold')
    tick_ms = [0, 200, 400, 600, 800, 1000]
    tick_pos = np.linspace(0, x.shape[1], len(tick_ms))
    axes[1].set_xticks(tick_pos)
    axes[1].set_xticklabels([str(v) for v in tick_ms], fontsize=30, fontweight='bold')
    axes[1].set_xlabel('Time (ms)', fontsize=30, fontweight='bold')
    axes[1].set_ylabel('Lead', fontsize=30, fontweight='bold')
    axes[1].set_yticks(list(range(12)))
    axes[1].set_yticklabels(LEAD_NAMES, fontsize=30, fontweight='bold')
    axes[1].set_ylim(-0.5, x.shape[0] + 1.5)
    cax = axes[1].inset_axes([1.02, 0.05, 0.02, 0.9])
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.yaxis.get_offset_text().set_visible(False)  
    cbar.set_label('Importance', fontsize=30, fontweight='bold')
    cbar.ax.tick_params(labelsize=25)
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight('bold')
    axes[1].tick_params(axis='both', labelsize=25)
    for tick in axes[1].get_xticklabels() + axes[1].get_yticklabels():
        tick.set_fontweight('bold')

    lead_signal = x[best_idx]
    lead_norm = lead_signal / (np.std(lead_signal) + 1e-6) if np.std(lead_signal) > 0 else lead_signal
    qrs_start, qrs_end = LVR_FEATURES['qrs_complex']
    r_local: int = int(np.argmax(np.abs(lead_norm[qrs_start:qrs_end])).item())
    r_idx: int = qrs_start + r_local
    axes[2].axvline(x=int(r_idx), color='black', linestyle='--', label='_nolegend_')

    sal_single = sal[best_idx, :]
    smin, smax = sal_single.min(), sal_single.max()
    sal_norm = (sal_single - smin) / (smax - smin + 1e-12)
    axes[2].plot(sal_norm, label='Saliency', color='C2', alpha=0.9)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel('Importance', fontsize=30, fontweight='bold', color='black')
    axes[2].set_xlim(0, x.shape[1])
    axes[2].tick_params(axis='both', labelsize=25)
    for tick in axes[2].get_xticklabels() + axes[2].get_yticklabels():
        tick.set_fontweight('bold')

    ax2 = axes[2].twinx()
    ax2.plot(lead_norm, label=f'Best Lead {best_lead_name}', color='C0', alpha=0.85)
    ax2.set_ylabel('')
    ax2.tick_params(axis='y', right=False, labelright=False)
    ax2.spines['right'].set_visible(False)
    ax2.set_xlim(0, x.shape[1])

    axes[2].set_title(f'Best Lead {best_lead_name}: R Peak - Saliency', fontsize=30, fontweight='bold')
    h1, l1 = axes[2].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[2].legend(h1 + h2, l1 + l2, loc='upper right', fontsize=20)

    axes[2].grid(True, alpha=0.3)
    y_min, y_max = axes[2].get_ylim()
    axes[2].text(int(r_idx) + 3, y_max * 0.95, 'R peak', color='black', fontsize=30, va='top')

    plt.tight_layout()
    idx_for_filename = save_idx if save_idx is not None else sample_idx
    base_name = f'saliency_analysis_{data_type}_sample_{idx_for_filename}'
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_path / f"{base_name}.png"
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close()
    print(f'[Saved] {out_path}')

def main(num_samples_per_class=20, batch_size=32, use_smoothgrad=True):
    print(f'Using device: {device}')
    os.makedirs('output_saliencyana2', exist_ok=True)

    set_seed(42)
    data, labels_four, patient_ids = load_and_process_data()
    
    # Convert to binary labels for LVR classification
    # labels_four: 0=NS_0, 1=NS_1, 2=ST_0, 3=ST_1
    # labels_binary: 0=no LVR (0,2), 1=has LVR (1,3)
    labels_binary = np.where((labels_four == 1) | (labels_four == 3), 1, 0)
    
    X_train, y_train, X_val, y_val, X_test, y_test_binary, patient_ids_test = split_dataset_by_patients(
        data, labels_binary, patient_ids, test_size=0.1, val_size=0.2, random_state=42
    )
    
    test_mask = np.isin(patient_ids, patient_ids_test)
    y_test_four = labels_four[test_mask]
    
    unique, counts = np.unique(y_test_four, return_counts=True)
    print("\nTest set distribution:")
    label_names = {0: 'NSTEMI_non_LVR', 1: 'NSTEMI_LVR', 2: 'STEMI_non_LVR', 3: 'STEMI_LVR'}
    for u, c in zip(unique, counts):
        print(f"  {label_names.get(u, u)}: {c}")
    print(f"  Total: {len(y_test_four)}")
    
    test_dataset = [[X_test[i], y_test_binary[i]] for i in range(len(X_test))]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ResNet1D(in_channels=12, experiment='ca_ms_v2').to(device)
    if os.path.exists('best_model2.pth'):
        model.load_state_dict(torch.load('best_model2.pth', map_location=device))
        print('Loaded weights: best_model2.pth')
    else:
        print('Warning: best_model2.pth not found. Using randomly initialized model.')

    global_idx = 0
    lead_importance_dict = {}
    
    for batch_idx, batch in enumerate(test_loader):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)

        ch_imp, t_imp, saliency = compute_saliency_optimized(
            model, inputs, targets, use_smoothgrad=use_smoothgrad, num_samples=10, noise_level=0.1
        )

        with torch.no_grad():
            logits = model(inputs)
            pred_classes = logits.argmax(dim=1)

        label_to_subtype = {0: 'NSTEMI_0', 1: 'NSTEMI_1', 2: 'STEMI_0', 3: 'STEMI_1'}
        subgroup_map = {0: 'NSTEMI_non_LVR', 1: 'NSTEMI_LVR', 2: 'STEMI_non_LVR', 3: 'STEMI_LVR'}
        
        for i in range(inputs.shape[0]):
            true_class_four = int(y_test_four[global_idx])
            true_class_binary = int(targets[i].item())
            pred_class = int(pred_classes[i].item())
            
            dtype_str = label_to_subtype.get(true_class_four, f'Class_{true_class_four}')
            subgroup = subgroup_map.get(true_class_four, 'Unknown')
            
            if true_class_four in [1, 3]:
                lead_imp_raw = ch_imp[i].cpu().numpy()
                lead_imp_normalized = lead_imp_raw / (lead_imp_raw.sum() + 1e-12)
                lead_importance_dict[global_idx] = {
                    'subgroup': subgroup,
                    'lead_importance': lead_imp_normalized,
                    'true_class': true_class_binary
                }
            
            sub_dir = os.path.join('output_saliencyana2', dtype_str)
            visualize_saliency_for_sample(
                inputs, (ch_imp, t_imp, saliency), i, pred_class, true_class_binary, dtype_str,
                topk=6, out_dir=sub_dir, save_idx=global_idx
            )
            global_idx += 1
    
    if lead_importance_dict:
        print(f"\nCollected {len(lead_importance_dict)} positive samples for quantitative analysis")
        analyze_lead_importance_consistency(lead_importance_dict, 'output_saliencyana2')
    else:
        print("\nWarning: No positive samples collected, skipping quantitative analysis")

def analyze_lead_importance_consistency(lead_importance_dict, output_dir='output_saliencyana2'):
    """
    Quantitative consistency analysis: analyze key lead importance distribution in STEMI and NSTEMI subgroups.
    
    Args:
        lead_importance_dict: Dictionary containing lead importance for each sample {sample_idx: {subgroup: str, lead_importance: np.array(12), true_class: int}}
        output_dir: Output directory
    """
    from scipy import stats
    from scipy.stats import f_oneway
    import json
    
    print("\n" + "="*60)
    print("Quantitative Consistency Analysis - Key Lead Importance Distribution in LVR Samples")
    print("="*60)
    
    stemi_samples = []
    nstemi_samples = []
    
    subgroup_counts = {'NSTEMI_LVR': 0, 'STEMI_LVR': 0}
    
    for sample_idx, data in lead_importance_dict.items():
        subgroup = data['subgroup']
        lead_imp = data['lead_importance']
        
        if subgroup in subgroup_counts:
            subgroup_counts[subgroup] += 1
        
        if subgroup == 'STEMI_LVR':
            stemi_samples.append(lead_imp)
        elif subgroup == 'NSTEMI_LVR':
            nstemi_samples.append(lead_imp)
    
    print(f"\nLVR Sample Distribution:")
    print(f"  NSTEMI_LVR (label 1): {subgroup_counts['NSTEMI_LVR']}")
    print(f"  STEMI_LVR (label 3): {subgroup_counts['STEMI_LVR']}")
    print(f"  Total LVR samples: {sum(subgroup_counts.values())}")
    
    stemi_array = np.array(stemi_samples) if stemi_samples else np.array([])
    nstemi_array = np.array(nstemi_samples) if nstemi_samples else np.array([])
    
    results = {}
    
    if len(stemi_array) > 0:
        print(f"\n[STEMI LVR Subgroup] Samples: {len(stemi_array)}")
        
        stemi_mean_importance = np.mean(stemi_array, axis=0)
        stemi_top3_indices = np.argsort(stemi_mean_importance)[-3:][::-1]
        stemi_top3_names = [LEAD_NAMES[i] for i in stemi_top3_indices]
        stemi_top3_scores = stemi_mean_importance[stemi_top3_indices]
        
        print(f"  Top 3 leads by mean importance: {stemi_top3_names}")
        print(f"  Importance scores: {stemi_top3_scores}")
        
        other_indices = [i for i in range(12) if i not in stemi_top3_indices]
        top3_data = [stemi_array[:, i] for i in stemi_top3_indices]
        other_data = [stemi_array[:, i] for i in other_indices]
        
        f_stat, p_value = f_oneway(*top3_data, *other_data)
        print(f"  One-way ANOVA: F={f_stat:.4f}, P={p_value:.6f}")
        
        loose_top3_count = 0
        avg_key_in_top3 = 0
        
        for sample_imp in stemi_array:
            sample_top3 = np.argsort(sample_imp)[-3:]
            key_in_top3 = sum(1 for idx in stemi_top3_indices if idx in sample_top3)
            avg_key_in_top3 += key_in_top3
            if key_in_top3 >= 2:
                loose_top3_count += 1
        
        avg_key_in_top3 /= len(stemi_array)
        loose_top3_percentage = (loose_top3_count / len(stemi_array)) * 100
        
        print(f"  Loose consistency (at least 2 key leads in top 3): {loose_top3_percentage:.1f}%")
        print(f"  Average key leads in top 3: {avg_key_in_top3:.2f}/3")
        
        icc_value, ci_lower, ci_upper = compute_icc(stemi_array)
        print(f"  ICC(12 leads)={icc_value:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
        
        results['STEMI'] = {
            'top3_leads': stemi_top3_names,
            'top3_scores': stemi_top3_scores.tolist(),
            'anova_f': float(f_stat),
            'anova_p': float(p_value),
            'strict_consistency_percentage': float(strict_top3_percentage),
            'loose_consistency_percentage': float(loose_top3_percentage),
            'avg_key_in_top3': float(avg_key_in_top3),
            'icc': float(icc_value),
            'icc_ci_lower': float(ci_lower),
            'icc_ci_upper': float(ci_upper),
            'sample_count': len(stemi_array)
        }
    
    if len(nstemi_array) > 0:
        print(f"\n[NSTEMI LVR Subgroup] Samples: {len(nstemi_array)}")
        
        nstemi_mean_importance = np.mean(nstemi_array, axis=0)
        nstemi_top3_indices = np.argsort(nstemi_mean_importance)[-3:][::-1]
        nstemi_top3_names = [LEAD_NAMES[i] for i in nstemi_top3_indices]
        nstemi_top3_scores = nstemi_mean_importance[nstemi_top3_indices]
        
        print(f"  Top 3 leads by mean importance: {nstemi_top3_names}")
        print(f"  Importance scores: {nstemi_top3_scores}")
        
        other_indices = [i for i in range(12) if i not in nstemi_top3_indices]
        top3_data = [nstemi_array[:, i] for i in nstemi_top3_indices]
        other_data = [nstemi_array[:, i] for i in other_indices]
        
        f_stat, p_value = f_oneway(*top3_data, *other_data)
        print(f"  One-way ANOVA: F={f_stat:.4f}, P={p_value:.6f}")
        
        loose_top3_count = 0
        avg_key_in_top3 = 0
        
        for sample_imp in nstemi_array:
            sample_top3 = np.argsort(sample_imp)[-3:]
            key_in_top3 = sum(1 for idx in nstemi_top3_indices if idx in sample_top3)
            avg_key_in_top3 += key_in_top3
            if key_in_top3 >= 2:
                loose_top3_count += 1
        
        avg_key_in_top3 /= len(nstemi_array)
        loose_top3_percentage = (loose_top3_count / len(nstemi_array)) * 100
        
        print(f"  Loose consistency (at least 2 key leads in top 3): {loose_top3_percentage:.1f}%")
        print(f"  Average key leads in top 3: {avg_key_in_top3:.2f}/3")
        
        icc_value, ci_lower, ci_upper = compute_icc(nstemi_array)
        print(f"  ICC(12 leads)={icc_value:.3f} (95% CI: {ci_lower:.3f}-{ci_upper:.3f})")
        
        results['NSTEMI'] = {
            'top3_leads': nstemi_top3_names,
            'top3_scores': nstemi_top3_scores.tolist(),
            'anova_f': float(f_stat),
            'anova_p': float(p_value),
            'loose_consistency_percentage': float(loose_top3_percentage),
            'avg_key_in_top3': float(avg_key_in_top3),
            'icc': float(icc_value),
            'icc_ci_lower': float(ci_lower),
            'icc_ci_upper': float(ci_upper),
            'sample_count': len(nstemi_array)
        }
    
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, 'quantitative_consistency_analysis.json')
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nAnalysis results saved: {result_path}")
    
    # Generate visualization plots
    plot_lead_importance_comparison(stemi_array, nstemi_array, output_dir)
    plot_lead_importance_heatmap(stemi_array, nstemi_array, output_dir)
    
    return results

def compute_icc(data_array):
    """
    Compute Intraclass Correlation Coefficient (ICC) using single-factor random effects model ICC(2,1).
    
    Args:
        data_array: (n_samples, n_leads) array
    
    Returns:
        icc_value: ICC value
        ci_lower: 95% confidence interval lower bound
        ci_upper: 95% confidence interval upper bound
    """
    from scipy import stats
    
    n_samples, n_leads = data_array.shape
    
    if n_samples < 2 or n_leads < 2:
        return 0.0, 0.0, 0.0
    
    lead_means = np.mean(data_array, axis=0)
    grand_mean = np.mean(data_array)
    
    ss_between = n_samples * np.sum((lead_means - grand_mean) ** 2)
    df_between = n_leads - 1
    ms_between = ss_between / df_between if df_between > 0 else 0
    
    ss_within = np.sum((data_array - lead_means) ** 2)
    df_within = n_leads * (n_samples - 1)
    ms_within = ss_within / df_within if df_within > 0 else 0
    
    if ms_between + (n_samples - 1) * ms_within == 0:
        icc_value = 0.0
    else:
        icc_value = (ms_between - ms_within) / (ms_between + (n_samples - 1) * ms_within)
    
    if ms_within > 0 and ms_between > 0:
        f_value = ms_between / ms_within
        df1, df2 = df_between, df_within
        
        f_lower = stats.f.ppf(0.025, df1, df2)
        f_upper = stats.f.ppf(0.975, df1, df2)
        
        ci_lower = (f_value / f_upper - 1) / (f_value / f_upper + n_samples - 1)
        ci_upper = (f_value / f_lower - 1) / (f_value / f_lower + n_samples - 1)
        
        ci_lower = max(-1.0, min(1.0, ci_lower))
        ci_upper = max(-1.0, min(1.0, ci_upper))
    else:
        ci_lower, ci_upper = 0.0, 0.0
    
    return icc_value, ci_lower, ci_upper

def plot_lead_importance_comparison(stemi_array, nstemi_array, output_dir):
    """Plot lead importance comparison between STEMI and NSTEMI groups."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    label_fontsize = 12
    title_fontsize = 14
    tick_fontsize = 12
    
    if len(stemi_array) > 0:
        stemi_mean = np.mean(stemi_array, axis=0)
        stemi_std = np.std(stemi_array, axis=0)
        axes[0].bar(range(12), stemi_mean, yerr=stemi_std, capsize=5, 
                   color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_xticks(range(12))
        axes[0].set_xticklabels(LEAD_NAMES, fontsize=tick_fontsize)
        axes[0].set_ylabel('Mean Importance', fontsize=label_fontsize)
        axes[0].set_title(f'STEMI LVR (n={len(stemi_array)})', fontsize=title_fontsize)
        axes[0].tick_params(axis='y', labelsize=tick_fontsize)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        top3_idx = np.argsort(stemi_mean)[-3:][::-1]
        for idx in top3_idx:
            axes[0].bar(idx, stemi_mean[idx], color='red', alpha=0.8, edgecolor='black')
    
    if len(nstemi_array) > 0:
        nstemi_mean = np.mean(nstemi_array, axis=0)
        nstemi_std = np.std(nstemi_array, axis=0)
        axes[1].bar(range(12), nstemi_mean, yerr=nstemi_std, capsize=5,
                   color='forestgreen', alpha=0.7, edgecolor='black')
        axes[1].set_xticks(range(12))
        axes[1].set_xticklabels(LEAD_NAMES, fontsize=tick_fontsize)
        axes[1].set_ylabel('Mean Importance', fontsize=label_fontsize)
        axes[1].set_title(f'NSTEMI LVR (n={len(nstemi_array)})', fontsize=title_fontsize)
        axes[1].tick_params(axis='y', labelsize=tick_fontsize)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        top3_idx = np.argsort(nstemi_mean)[-3:][::-1]
        for idx in top3_idx:
            axes[1].bar(idx, nstemi_mean[idx], color='red', alpha=0.8, edgecolor='black')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'lead_importance_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved: {plot_path}")

def plot_lead_importance_heatmap(stemi_array, nstemi_array, output_dir):
    """Plot lead importance heatmap (samples × leads)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    label_fontsize = 12
    title_fontsize = 14
    tick_fontsize = 12
    
    if len(stemi_array) > 0:
        im1 = axes[0].imshow(stemi_array, aspect='auto', cmap='YlOrRd', 
                             interpolation='nearest')
        axes[0].set_xticks(range(12))
        axes[0].set_xticklabels(LEAD_NAMES, fontsize=tick_fontsize, rotation=0)
        axes[0].set_ylabel('Sample Index', fontsize=label_fontsize)
        axes[0].set_title(f'STEMI LVR Lead Importance Heatmap (n={len(stemi_array)})', fontsize=title_fontsize)
        axes[0].set_xlabel('Lead', fontsize=label_fontsize)
        axes[0].tick_params(axis='both', labelsize=tick_fontsize)
        cbar1 = plt.colorbar(im1, ax=axes[0], label='Importance')
        cbar1.ax.tick_params(labelsize=tick_fontsize)
    
    if len(nstemi_array) > 0:
        im2 = axes[1].imshow(nstemi_array, aspect='auto', cmap='YlGn',
                             interpolation='nearest')
        axes[1].set_xticks(range(12))
        axes[1].set_xticklabels(LEAD_NAMES, fontsize=tick_fontsize, rotation=0)
        axes[1].set_ylabel('Sample Index', fontsize=label_fontsize)
        axes[1].set_title(f'NSTEMI LVR Lead Importance Heatmap (n={len(nstemi_array)})', fontsize=title_fontsize)
        axes[1].set_xlabel('Lead', fontsize=label_fontsize)
        axes[1].tick_params(axis='both', labelsize=tick_fontsize)
        cbar2 = plt.colorbar(im2, ax=axes[1], label='Importance')
        cbar2.ax.tick_params(labelsize=tick_fontsize)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'lead_importance_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved: {plot_path}")


if __name__ == '__main__':
    main()