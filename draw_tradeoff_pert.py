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
    ax1.set_xlabel('Epsilon')
    ax1.grid(True, linestyle=':') 
    ax1.axhline(0, color='silver', linewidth=1.25)
    ax1.legend(loc='center right', fontsize='small')
    ax1.set_xticks(epsilon)

    # Plot for EMFC
    ax2.plot(epsilon, emfc_mitigations, label='Bias Mitigation (%)', marker='D', color='blue', linestyle='--')
    ax2.plot(epsilon, emfc_acclosses, label='Accuracy Loss (%)', marker='*', color='firebrick', linestyle='--')
    net_effectiveness = np.array(emfc_mitigations) + np.array(emfc_acclosses)
    ax2.plot(epsilon, net_effectiveness, label='Net Gain (%)', marker='o', color='darkorange')
    ax2.fill_between(epsilon, 0, net_effectiveness, where=(net_effectiveness > 0), color='darkorange', alpha=0.3)
    ax2.set_title('EMFC')
    ax2.set_xlabel('Epsilon')
    ax2.grid(True, linestyle=':') 
    ax2.axhline(0, color='silver', linewidth=1.25)
    ax2.legend(loc='center right', fontsize='small')
    ax2.set_xticks(epsilon)

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

# epsilon = [4, 6, 8, 10, 12, 14, 16, 18]
epsilon = [4, 8, 12, 16, 20, 24, 28, 32]


baseline_accuracy, baseline_unfairness = 82.28, 18.52
# TLFC
# tlfc_accuracies = [81.08, 80.43, 80.47, 79.71, 80.11, 78.61, 78.61, 76.53, ]
# tlfc_unfairnesses = [11.76, 8.19, 6.22, 2.77, 3.01, 2.05, 2.48, 0.33, ]
tlfc_accuracies = [81.08, 80.47, 80.11, 78.61, 81.06, 81.20, 80.65, 81.04, ]
tlfc_unfairnesses = [11.76, 6.22, 3.01, 2.48, 5.86, 5.22, 3.6, 4.6, ]

tlfc_mitigations, tlfc_acclosses = compute_bias_accuracy_metrics(baseline_accuracy, baseline_unfairness, tlfc_accuracies, tlfc_unfairnesses)

# EMFC
# emfc_accuracies = [81.19, 79.3, 80.33, 79.48, 78.71, 79.76, 79.4, 80.47, ]
# emfc_unfairnesses = [11.96, 7.87, 7.25, 4.8, 4.51, 5.11, 3.03, 4.11, ]
emfc_accuracies = [81.19, 80.33, 78.71, 79.4, 76.56, 79.46, 79.38, 79.62, ]
emfc_unfairnesses = [11.96, 7.25, 4.51, 3.03, 0.78, 3.88, 2.51, 3.15, ]

emfc_mitigations, emfc_acclosses = compute_bias_accuracy_metrics(baseline_accuracy, baseline_unfairness, emfc_accuracies, emfc_unfairnesses)

file_name = 'bae.png'
save_path = '.'

plot_bias_accuracy_comparison(
    epsilon, 
    tlfc_mitigations, tlfc_acclosses, 
    emfc_mitigations, emfc_acclosses, 
    save_path, file_name)

