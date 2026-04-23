import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import joblib

# Set random seed for reproducibility
RANDOM_SEED = 42
NUM_CLASSES = 5

# Ensure output directory exists
os.makedirs("trained_model", exist_ok=True)

# Data paths
TRAIN_CSV = "assests/dataset/train.csv"
DATA_DIR = "assests/dataset/data/"
FEATURES_CSV = "assests/dataset/our_features.csv"
PHYSIONET_CSV = "assests/dataset/physionet_pronation_supination_features.csv"

# Load metadata and landmarks
meta_data = pd.read_csv(TRAIN_CSV)
df_features = pd.read_csv(FEATURES_CSV)
df_physionet = pd.read_csv(PHYSIONET_CSV)

# Define mappings
gender_map = {"Male": 1, "Female": 0, "0": 2}
class_labels = sorted(meta_data["folder_path"].unique())
class_num = {class_labels[i]: i for i in range(len(class_labels))}

moves = []
metadata_list = []
engineered_features = []
classes = []

print("Preprocessing original data for Ultimate Model...")
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
    
    # Engineered features
    # our_features.csv has 92 features + 'label' column
    eng_feat = df_features.iloc[i, :-1].to_numpy(dtype=np.float32)
    engineered_features.append(eng_feat)

print("Preprocessing PhysioNet data...")
# PhysioNet data has features in columns '0' to '91'
# Metadata in 'gender', 'age', 'off_on', 'doctor_diagnosis'
# Label in 'label'
for i in tqdm(range(df_physionet.shape[0])):
    row = df_physionet.iloc[i]
    
    # Classes
    classes.append(class_num[row["label"]])
    
    # Landmarks (DUMMY/ZERO padding for PhysioNet)
    data_points = np.zeros((100, 63))
    moves.append(data_points)
    
    # Metadata
    metadata_list.append([row['gender'], row['age'], row['off_on'], row['doctor_diagnosis']])
    
    # Engineered features
    # Explicitly pick columns '0' to '91' to ensure order
    eng_feat = row[[str(j) for i in range(1) for j in range(92)]].to_numpy(dtype=np.float32)
    engineered_features.append(eng_feat)

# Convert to numpy arrays
X_landmarks = np.array(moves)
X_metadata = np.array(metadata_list)
X_engineered = np.array(engineered_features)
X_engineered = np.nan_to_num(X_engineered)
y = np.array(classes)

print(f"Total samples: {len(y)}")
print(f"Shape of X_engineered: {X_engineered.shape}")

# Split
indices = np.arange(len(y))
train_idx, test_idx = train_test_split(indices, train_size=0.75, random_state=RANDOM_SEED, stratify=y)

X_l_train, X_l_test = X_landmarks[train_idx], X_landmarks[test_idx]
X_m_train, X_m_test = X_metadata[train_idx], X_metadata[test_idx]
X_e_train, X_e_test = X_engineered[train_idx], X_engineered[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

from sklearn.utils import class_weight

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights))

print(f"Class weights: {class_weight_dict}")

# Model definition (Triple Input)
landmark_input = tf.keras.layers.Input(shape=(100, 63), name="landmark_input")
x = tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu')(landmark_input)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.GlobalAveragePooling1D()(x)

metadata_input = tf.keras.layers.Input(shape=(4,), name="metadata_input")
y_meta = tf.keras.layers.Dense(16, activation='relu')(metadata_input)

engineered_input = tf.keras.layers.Input(shape=(X_engineered.shape[1],), name="engineered_input")
z = tf.keras.layers.Dense(64, activation='relu')(engineered_input)
z = tf.keras.layers.Dropout(0.2)(z)
z = tf.keras.layers.Dense(32, activation='relu')(z)

combined = tf.keras.layers.concatenate([x, y_meta, z])
combined = tf.keras.layers.Dense(64, activation='relu')(combined)
combined = tf.keras.layers.Dropout(0.3)(combined)
output = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(combined)

model = tf.keras.models.Model(inputs=[landmark_input, metadata_input, engineered_input], outputs=output)

# Compilation
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Callbacks
best_model_save_path = "trained_model/best_physionet_augmented.keras"

es_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=50, verbose=1, restore_best_weights=True
)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_save_path,
    save_best_only=True,
    monitor="val_accuracy",
    mode="max"
)

print("Starting training (PhysioNet Augmented Model)...")
history = model.fit(
    [X_l_train, X_m_train, X_e_train],
    y_train,
    epochs=200, # Reduced epochs for faster iteration, but with early stopping
    batch_size=64,
    validation_data=([X_l_test, X_m_test, X_e_test], y_test),
    callbacks=[es_callback, model_checkpoint_callback],
    class_weight=class_weight_dict,
    verbose=1
)

# Save training history
with open("trained_model/history_physionet_augmented.pkl", "wb") as f:
    import pickle
    pickle.dump(history.history, f)

print(f"Training completed. Best augmented model saved to {best_model_save_path}")
print(f"History saved to trained_model/history_physionet_augmented.pkl")
