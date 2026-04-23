
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Paths ---
ORIG_REPORT = "visualizations/classification_report.csv"
AUG_REPORT = "visualizations_physionet_augmented/classification_report_augmented.csv"
OUTPUT_DIR = "visualizations_physionet_augmented/model_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Loading ---
df_orig = pd.read_csv(ORIG_REPORT, index_col=0).drop(['accuracy', 'macro avg', 'weighted avg'])
df_aug = pd.read_csv(AUG_REPORT, index_col=0).drop(['accuracy', 'macro avg', 'weighted avg'])

df_orig['Model'] = 'Original'
df_aug['Model'] = 'Augmented'

df_combined = pd.concat([df_orig.reset_index(), df_aug.reset_index()], axis=0)

# --- Plotting ---
metrics = ['precision', 'recall', 'f1-score']

for metric in metrics:
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_combined, x='index', y=metric, hue='Model', palette='muted')
    plt.title(f'Model Comparison: {metric.capitalize()} by Class')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Tremor Exercise Class')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'compare_{metric}.png'))
    plt.close()

# Overall Accuracy Comparison
df_acc_orig = pd.read_csv(ORIG_REPORT, index_col=0).loc['accuracy', 'precision'] # Accuracy is stored in precision col for accuracy row
df_acc_aug = pd.read_csv(AUG_REPORT, index_col=0).loc['accuracy', 'precision']

plt.figure(figsize=(8, 6))
sns.barplot(x=['Original Model', 'Augmented Model'], y=[df_acc_orig, df_acc_aug], palette='coolwarm')
plt.title('Overall Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1.1)
for i, v in enumerate([df_acc_orig, df_acc_aug]):
    plt.text(i, v + 0.02, f"{v:.2%}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'compare_accuracy.png'))
plt.close()

print(f"Model comparison visualizations generated in {OUTPUT_DIR}")
