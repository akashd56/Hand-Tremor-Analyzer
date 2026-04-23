
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import OneHotEncoder
from itertools import cycle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# --- Configuration ---
MODEL_PATH = 'trained_model/best_physionet_augmented.keras'
TRAIN_CSV_PATH = 'assests/dataset/train.csv'
DATA_DIR = 'assests/dataset/data/'
FEATURES_CSV = 'assests/dataset/our_features.csv'
PHYSIONET_CSV = 'assests/dataset/physionet_pronation_supination_features.csv'
OUTPUT_DIR = 'visualizations_physionet_augmented'
RANDOM_SEED = 42

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Data Loading ---
print("Loading data for evaluation (Original + PhysioNet)...")
meta_data = pd.read_csv(TRAIN_CSV_PATH)
df_features = pd.read_csv(FEATURES_CSV)
df_physionet = pd.read_csv(PHYSIONET_CSV)

gender_map = {"Male": 1, "Female": 0, "0": 2}
class_labels = sorted(meta_data["folder_path"].unique())
class_num = {class_labels[i]: i for i in range(len(class_labels))}
num_to_class = {i: class_labels[i] for i in range(len(class_labels))}

moves = []
metadata_list = []
engineered_features = []
classes = []

print("Preprocessing original data...")
for i in tqdm(range(meta_data.shape[0])):
    data_file = os.path.join(DATA_DIR, meta_data["data_file_name"][i])
    if not os.path.exists(data_file):
        continue

    temporary_data = pd.read_csv(data_file)
    temporary_meta = meta_data.iloc[i]

    gender = gender_map.get(temporary_meta['gender'], 2)
    age = temporary_meta['age']
    off_on = 1 if temporary_meta['patient_off_on'] == 'on' else 0
    diagnosis = temporary_meta['doctor_diagnosis_0_5']
    
    classes.append(class_num[temporary_meta["folder_path"]])

    # Landmarks
    data_points = temporary_data.drop("TIME", axis=1).values
    if len(data_points) < 100:
        padding = np.zeros((100 - len(data_points), data_points.shape[1]))
        data_points = np.vstack([data_points, padding])
    else:
        data_points = data_points[:100]

    moves.append(data_points)
    metadata_list.append([gender, age, off_on, diagnosis])
    engineered_features.append(df_features.iloc[i, :-1].to_numpy(dtype=np.float32))

print("Preprocessing PhysioNet data...")
for i in tqdm(range(df_physionet.shape[0])):
    row = df_physionet.iloc[i]
    classes.append(class_num[row["label"]])
    moves.append(np.zeros((100, 63))) # Dummy landmarks
    metadata_list.append([row['gender'], row['age'], row['off_on'], row['doctor_diagnosis']])
    eng_feat = row[[str(j) for i in range(1) for j in range(92)]].to_numpy(dtype=np.float32)
    engineered_features.append(eng_feat)

X_l = np.array(moves)
X_m = np.array(metadata_list)
X_e = np.nan_to_num(np.array(engineered_features))
y = np.array(classes)

# Split (Ensure the same split as training)
indices = np.arange(len(y))
_, test_idx = train_test_split(indices, train_size=0.75, random_state=RANDOM_SEED, stratify=y)

X_l_test, X_m_test, X_e_test, y_test = X_l[test_idx], X_m[test_idx], X_e[test_idx], y[test_idx]

# --- Prediction ---
print("Loading augmented model and predicting...")
model = tf.keras.models.load_model(MODEL_PATH)
y_pred_proba = model.predict([X_l_test, X_m_test, X_e_test])
y_pred = np.argmax(y_pred_proba, axis=1)

# --- Results ---
class_names = [num_to_class[i] for i in sorted(num_to_class)]
print("\nPhysioNet Augmented Model Classification Report:")
report_dict = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
df_report.to_csv(os.path.join(OUTPUT_DIR, "classification_report_augmented.csv"))
print(classification_report(y_test, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('PhysioNet Augmented Model Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix_augmented.png'))
plt.close()

# ROC Curves
print("Generating ROC curves for Augmented Model...")
y_test_bin = OneHotEncoder(sparse_output=False).fit_transform(y_test.reshape(-1, 1))
n_classes = len(class_names)

fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(12, 10))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'deeppink', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC for {class_names[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves (PhysioNet Augmented Model)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curves_augmented.png'))
plt.close()

print(f"Evaluation complete. Results in {OUTPUT_DIR}/")
