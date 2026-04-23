
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- Paths ---
TRAIN_CSV = "assests/dataset/train.csv"
PHYSIONET_CSV = "assests/dataset/physionet_pronation_supination_features.csv"
OUTPUT_DIR = "visualizations_physionet_augmented/dataset_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Data Loading ---
df_original = pd.read_csv(TRAIN_CSV)
df_physionet = pd.read_csv(PHYSIONET_CSV)

# --- Preprocess Metadata ---
# Original columns: age, gender, patient_off_on (on/off), doctor_diagnosis_0_5, folder_path (label)
# PhysioNet columns: age, gender (1/0), off_on (1/0), doctor_diagnosis, label

# Standardize Original
df_orig_meta = df_original[['age', 'gender', 'patient_off_on', 'doctor_diagnosis_0_5', 'folder_path']].copy()
df_orig_meta.columns = ['age', 'gender', 'off_on', 'diagnosis', 'label']
df_orig_meta['off_on'] = df_orig_meta['off_on'].map({'on': 1, 'off': 0})
df_orig_meta['source'] = 'Original'

# Standardize PhysioNet
df_phys_meta = df_physionet[['age', 'gender', 'off_on', 'doctor_diagnosis', 'label']].copy()
df_phys_meta.columns = ['age', 'gender', 'off_on', 'diagnosis', 'label']
df_phys_meta['source'] = 'PhysioNet'

# Combined
df_combined = pd.concat([df_orig_meta, df_phys_meta], axis=0)

# --- Distribution Plots ---

def plot_distribution(df, col, title, filename, kind='count'):
    plt.figure(figsize=(10, 6))
    if kind == 'count':
        sns.countplot(data=df, x=col, hue='source', palette='viridis')
    elif kind == 'hist':
        sns.histplot(data=df, x=col, hue='source', bins=20, multiple="stack", palette='viridis')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

plot_distribution(df_combined, 'age', 'Age Distribution by Data Source', 'age_dist_combined.png', kind='hist')
plot_distribution(df_combined, 'gender', 'Gender Distribution (0=Female, 1=Male, 2=Other)', 'gender_dist_combined.png')
plot_distribution(df_combined, 'off_on', 'DBS Status (0=Off, 1=On)', 'dbs_status_combined.png')
plot_distribution(df_combined, 'diagnosis', 'UPDRS Diagnosis Score (0-5)', 'diagnosis_dist_combined.png')
plot_distribution(df_combined, 'label', 'Class Distribution (Final Dataset)', 'class_dist_combined.png')

# --- Feature Analysis ---
# Load features
df_features_orig = pd.read_csv("assests/dataset/our_features.csv")
# PhysioNet features are in the same file as metadata
feature_cols = [str(i) for i in range(92)]
df_features_phys = df_physionet[feature_cols + ['label']]
df_features_phys.columns = list(range(92)) + ['label']
df_features_orig.columns = list(range(92)) + ['label']

df_features_all = pd.concat([df_features_orig, df_features_phys], axis=0)

# Correlation Heatmap (Subset of features)
plt.figure(figsize=(12, 10))
# Pick 20 features with highest variance
top_var_features = df_features_all[list(range(92))].var().sort_values(ascending=False).head(20).index
corr = df_features_all[top_var_features].corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Top 20 High-Variance Engineered Features')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_correlation.png'))
plt.close()

# Feature Distribution per Class (Boxplot for top 3 features)
top_3 = top_var_features[:3]
for i, feat in enumerate(top_3):
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_features_all, x='label', y=feat, palette='Set2')
    plt.title(f'Distribution of Feature {feat} by Tremor Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'feature_{feat}_dist.png'))
    plt.close()

print(f"Dataset visualizations generated in {OUTPUT_DIR}")
