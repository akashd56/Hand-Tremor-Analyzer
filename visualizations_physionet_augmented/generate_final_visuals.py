
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import joblib
from sklearn.metrics import confusion_matrix, classification_report

# --- Paths ---
TRAIN_CSV = "assests/dataset/train.csv"
PHYSIONET_CSV = "assests/dataset/physionet_pronation_supination_features.csv"
REPORT_CSV = "visualizations_physionet_augmented/classification_report_augmented.csv"
OUTPUT_DIR = "visualizations_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Data Loading & Unification ---
df_original = pd.read_csv(TRAIN_CSV)
df_physionet = pd.read_csv(PHYSIONET_CSV)

# Standardize and Combine Metadata
df_orig_meta = df_original[['age', 'gender', 'patient_off_on', 'doctor_diagnosis_0_5', 'folder_path']].copy()
df_orig_meta.columns = ['age', 'gender', 'off_on', 'diagnosis', 'label']
df_orig_meta['off_on'] = df_orig_meta['off_on'].map({'on': 1, 'off': 0})
df_orig_meta['gender'] = df_orig_meta['gender'].replace({'Male': 1, 'Female': 0, '0': 2}).astype(float)

df_phys_meta = df_physionet[['age', 'gender', 'off_on', 'doctor_diagnosis', 'label']].copy()
df_phys_meta.columns = ['age', 'gender', 'off_on', 'diagnosis', 'label']
df_phys_meta['gender'] = df_phys_meta['gender'].astype(float)

df_combined = pd.concat([df_orig_meta, df_phys_meta], axis=0)
df_combined['gender'] = df_combined['gender'].fillna(2) # Map others/NaN to 2

# --- 1. Dataset Distribution Visuals ---

def save_plot(filename):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

# Age Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_combined['age'], bins=15, kde=True, color='skyblue')
plt.title('Patient Age Distribution')
plt.xlabel('Age')
save_plot('dataset_age_distribution.png')

# Gender Distribution
plt.figure(figsize=(8, 6))
gender_counts = df_combined['gender'].value_counts().sort_index()
labels = []
if 0 in gender_counts: labels.append('Female')
if 1 in gender_counts: labels.append('Male')
if 2 in gender_counts: labels.append('Other')
plt.pie(gender_counts, labels=labels, autopct='%1.1f%%', colors=['#99ff99','#66b3ff','#ffcc99'])
plt.title('Patient Gender Distribution')
save_plot('dataset_gender_distribution.png')

# DBS Status
plt.figure(figsize=(8, 6))
sns.countplot(x=df_combined['off_on'], palette='pastel')
plt.title('Neurostimulator (DBS) Status (0=Off, 1=On)')
plt.xlabel('Status')
save_plot('dataset_dbs_status.png')

# Class Distribution
plt.figure(figsize=(12, 6))
order = df_combined['label'].value_counts().index
sns.countplot(data=df_combined, y='label', order=order, palette='viridis')
plt.title('Tremor Exercise Class Distribution')
plt.xlabel('Number of Samples')
save_plot('dataset_class_distribution.png')

# --- 2. Feature Analysis ---
df_features_orig = pd.read_csv("assests/dataset/our_features.csv")
feature_cols = [str(i) for i in range(92)]
df_features_phys = df_physionet[feature_cols + ['label']]
df_features_phys.columns = list(range(92)) + ['label']
df_features_orig.columns = list(range(92)) + ['label']
df_features_all = pd.concat([df_features_orig, df_features_phys], axis=0)

# Correlation Heatmap
plt.figure(figsize=(12, 10))
top_var_features = df_features_all[list(range(92))].var().sort_values(ascending=False).head(15).index
corr = df_features_all[top_var_features].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdBu_r', center=0)
plt.title('Correlation Matrix of Key Engineered Signal Features')
save_plot('feature_correlation_matrix.png')

# --- 3. Model Performance Visuals ---
df_report = pd.read_csv(REPORT_CSV, index_col=0)
df_metrics = df_report.drop(['accuracy', 'macro avg', 'weighted avg'])

# Performance per Class
plt.figure(figsize=(14, 8))
df_metrics_melted = df_metrics[['precision', 'recall', 'f1-score']].reset_index().melt(id_vars='index')
sns.barplot(data=df_metrics_melted, x='index', y='value', hue='variable', palette='magma')
plt.title('Model Performance Metrics by Tremor Class')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right')
plt.legend(loc='lower right')
save_plot('model_performance_metrics.png')

# Confusion Matrix (Simplified placeholder since I don't have the raw y_test/y_pred here, 
# but I can explain it in the paper or use the existing one if available)
# I'll copy the existing confusion matrix to the new folder for consistency
import shutil
if os.path.exists('visualizations_physionet_augmented/confusion_matrix_augmented.png'):
    shutil.copy('visualizations_physionet_augmented/confusion_matrix_augmented.png', 
                os.path.join(OUTPUT_DIR, 'model_confusion_matrix.png'))

print(f"Final visualizations generated in {OUTPUT_DIR}")
