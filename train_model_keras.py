
import warnings 
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
# from matplotlib import pyplot as plt  # Not required by project, used for EDA
import tensorflow as tf
from tqdm import tqdm
from glob import glob
# import seaborn as sns  # Not required by project, used for EDA
import pandas as pd
import numpy as np
import os

# Set random seed for reproducibility
RANDOM_SEED = 42
NUM_CLASSES = 5

# Ensure output directory exists
os.makedirs('trained_model', exist_ok=True)

# Data paths
TRAIN_CSV = 'assests/dataset/train.csv'
DATA_DIR = 'assests/dataset/data/'

# Load metadata
meta_data = pd.read_csv(TRAIN_CSV)

# Define mappings
gender_map = {'Male': 1, 'Female': 0, '0': 2}
class_labels = meta_data['folder_path'].unique()
class_num = {class_labels[i]: i for i in range(len(class_labels))}

moves = []
classes = []

print("Preprocessing data...")
for i in tqdm(range(meta_data.shape[0])):
    data_file = os.path.join(DATA_DIR, meta_data['data_file_name'][i])
    if not os.path.exists(data_file):
        continue
    
    temporary_data = pd.read_csv(data_file)
    temporary_meta = meta_data[['gender', 'age', 'patient_off_on', 'doctor_diagnosis_0_5', 'folder_path']].iloc[i].values
    
    temporary_gender = gender_map.get(temporary_meta[0], 2)
    classes.append(class_num[temporary_meta[4]])
    
    # Take first 100 samples from data, padding if necessary
    data_points = temporary_data.drop('TIME', axis=1).values
    if len(data_points) < 100:
        # Pad with zeros if less than 100 points
        padding = np.zeros((100 - len(data_points), data_points.shape[1]))
        data_points = np.vstack([data_points, padding])
    else:
        data_points = data_points[:100]
    
    # Combine features
    features = np.append(data_points.reshape(-1), [
        temporary_gender, 
        temporary_meta[1], 
        int(temporary_meta[2] == 'on'), 
        temporary_meta[3]
    ])
    moves.append(features)

# Convert to numpy arrays
X = np.array(moves)
y = np.array(classes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=RANDOM_SEED)

# Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 3 * 100 + 4, )),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(48, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compilation
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
model_save_path = 'trained_model/last.keras'
best_model_save_path = 'trained_model/best.keras'

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)

es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=150, verbose=1)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=best_model_save_path,
    save_weights_only=False,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

print("Starting training...")
model.fit(
    X_train,
    y_train,
    epochs=1500,
    batch_size=1024,
    validation_data=(X_test, y_test),
    callbacks=[cp_callback, es_callback, model_checkpoint_callback]
)

print(f"Training completed. Best model saved to {best_model_save_path}")
