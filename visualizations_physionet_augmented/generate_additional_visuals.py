
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the classification report
report_path = 'visualizations_physionet_augmented/classification_report_augmented.csv'
df_report = pd.read_csv(report_path, index_col=0)

# Filter out averages
df_classes = df_report.drop(['accuracy', 'macro avg', 'weighted avg'])

# Plotting Performance Metrics per Class
plt.figure(figsize=(14, 8))
df_metrics = df_classes[['precision', 'recall', 'f1-score']]
df_metrics_melted = df_metrics.reset_index().melt(id_vars='index', var_name='Metric', value_name='Score')

sns.barplot(data=df_metrics_melted, x='index', y='Score', hue='Metric', palette='viridis')
plt.title('Model Performance Metrics by Class (PhysioNet Augmented Model)')
plt.ylabel('Score')
plt.xlabel('Tremor Exercise Class')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Metric', loc='lower right')
plt.tight_layout()
plt.savefig('visualizations_physionet_augmented/performance_by_class.png')
plt.close()

# Plotting Support (Class Distribution in Test Set)
plt.figure(figsize=(12, 6))
sns.barplot(x=df_classes.index, y=df_classes['support'], palette='magma')
plt.title('Class Support in Evaluation Dataset')
plt.ylabel('Number of Samples')
plt.xlabel('Tremor Exercise Class')
plt.xticks(rotation=45, ha='right')
plt.yscale('log') # Use log scale because of extreme imbalance
plt.grid(axis='y', which="both", linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('visualizations_physionet_augmented/class_support_log.png')
plt.close()

print("New visualizations saved: performance_by_class.png, class_support_log.png")
