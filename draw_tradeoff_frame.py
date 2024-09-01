import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_bias_accuracy_comparison(
    epsilon, 
    tlfc_mitigations, tlfc_acclosses, 
    emfc_mitigations, emfc_acclosses, 
    save_path, file_name):
    # Ensure the directory exists
    Path(save_path).mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))

    # Plot for TLFC
    ax1.plot(epsilon, tlfc_mitigations, label='Bias Mitigation (%)', marker='D', color='blue', linestyle='--')
    ax1.plot(epsilon, tlfc_acclosses, label='Accuracy Loss (%)', marker='*', color='firebrick', linestyle='--')
    net_effectiveness = np.array(tlfc_mitigations) + np.array(tlfc_acclosses)
    ax1.plot(epsilon, net_effectiveness, label='Net Gain (%)', marker='o', color='darkorange')
    ax1.fill_between(epsilon, 0, net_effectiveness, where=(net_effectiveness > 0), color='darkorange', alpha=0.3)
    ax1.set_title('TLFC')
    ax1.set_xlabel('Thickness')
    ax1.grid(True, linestyle=':') 
    ax1.axhline(0, color='silver', linewidth=1.25)
    ax1.legend(loc='upper right', fontsize='small')
    # ax1.set_xticks(epsilon)

    # Plot for EMFC
    ax2.plot(epsilon, emfc_mitigations, label='Bias Mitigation (%)', marker='D', color='blue', linestyle='--')
    ax2.plot(epsilon, emfc_acclosses, label='Accuracy Loss (%)', marker='*', color='firebrick', linestyle='--')
    net_effectiveness = np.array(emfc_mitigations) + np.array(emfc_acclosses)
    ax2.plot(epsilon, net_effectiveness, label='Net Gain (%)', marker='o', color='darkorange')
    ax2.fill_between(epsilon, 0, net_effectiveness, where=(net_effectiveness > 0), color='darkorange', alpha=0.3)
    ax2.set_title('EMFC')
    ax2.set_xlabel('Thickness')
    ax2.grid(True, linestyle=':') 
    ax2.axhline(0, color='silver', linewidth=1.25)
    ax2.legend(loc='upper right', fontsize='small')
    # ax2.set_xticks(epsilon)

    ax1.set_ylim(-10, 20)
    ax2.set_ylim(-10, 20)

    plt.tight_layout()
    # Save the plot
    full_path = Path(save_path) / file_name
    plt.savefig(full_path, dpi=300)
    plt.close()

    return f"Plot saved to {full_path}"

def compute_bias_accuracy_metrics(baseline_accuracy, baseline_unfairness, accuracies, unfairnesses):
    bias_mitigations = []
    accuracy_losses = []

    for acc, unf in zip(accuracies, unfairnesses):
        bias_mitigation = (baseline_unfairness - unf) 
        accuracy_loss = (baseline_accuracy - acc)
        
        bias_mitigations.append(bias_mitigation)
        accuracy_losses.append(-accuracy_loss)  # Negate to show loss as positive when accuracy decreases

    return bias_mitigations, accuracy_losses

epsilon = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225]
baseline_accuracy, baseline_unfairness = 82.28, 18.52
# TLFC
tlfc_accuracies = [80.84, 79.61, 80.44, 80.40, 79.73, 78.68, 79.51, 79.01, ]
tlfc_unfairnesses = [7.85, 7.39, 7.39, 7.26, 7.01, 5.78, 7.74, 5.73, ]
tlfc_mitigations, tlfc_acclosses = compute_bias_accuracy_metrics(baseline_accuracy, baseline_unfairness, tlfc_accuracies, tlfc_unfairnesses)

# EMFC
emfc_accuracies = [78.76, 78.02, 78.25, 78.27, 79.22, 79.39, 78.8, 78.14, ]
emfc_unfairnesses = [5.86, 5.54, 6.62, 6.45, 7.31, 7.69, 7.42, 6.87, ]
emfc_mitigations, emfc_acclosses = compute_bias_accuracy_metrics(baseline_accuracy, baseline_unfairness, emfc_accuracies, emfc_unfairnesses)

file_name = 'bae2.png'
save_path = '.'

plot_bias_accuracy_comparison(
    epsilon, 
    tlfc_mitigations, tlfc_acclosses, 
    emfc_mitigations, emfc_acclosses, 
    save_path, file_name)

