
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob
from tqdm import tqdm
import os

# Create output directory if it doesn't exist
output_dir = "visualizations"
os.makedirs(output_dir, exist_ok=True)

# --- Data Loading ---
try:
    df_meta = pd.read_csv('assests/dataset/train.csv')
    paths_to_tables = glob.glob('assests/dataset/data/*')
except FileNotFoundError:
    print("Dataset files not found. Please ensure 'assests/dataset/train.csv' and the data files in 'assests/dataset/data/' exist.")
    exit()

# --- 1. Recording Length Distribution ---
print("Analyzing recording lengths...")
lengths = []
for table_path in tqdm(paths_to_tables):
    df = pd.read_csv(table_path)
    lengths.append(df.shape[0])

plt.style.use("bmh")
plt.figure(figsize=(12, 7))
plt.hist(lengths, bins=50, edgecolor='black')
plt.title('Distribution of Recording Lengths (Number of Frames)')
plt.xlabel('Number of Frames per Recording')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig(os.path.join(output_dir, "recording_length_distribution.png"))
plt.close()

# --- 2. Class Distribution ---
print("Analyzing class distribution...")
plt.figure(figsize=(12, 8))
ax = sns.countplot(data=df_meta, y='folder_path', order=df_meta['folder_path'].value_counts().index, palette="viridis")
plt.title('Class Distribution of Tremor Types')
plt.xlabel('Number of Recordings')
plt.ylabel('Tremor Type')
# Add percentage labels
total = len(df_meta)
for p in ax.patches:
    percentage = f'{100 * p.get_width() / total:.1f}%'
    x = p.get_x() + p.get_width() + 0.02 * total
    y = p.get_y() + p.get_height() / 2
    ax.text(x, y, percentage, ha='center', va='center')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "class_distribution.png"))
plt.close()

# --- 3. Age Distribution ---
print("Analyzing age distribution...")
plt.figure(figsize=(14, 7))
sns.countplot(data=df_meta, x='age', order=sorted(df_meta['age'].unique()), palette="plasma")
plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "age_distribution.png"))
plt.close()

# --- 4. Gender Distribution ---
print("Analyzing gender distribution...")
gender_counts = df_meta['gender'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140, colors=['#66b3ff','#ff9999','#99ff99'])
plt.title('Gender Distribution')
plt.ylabel('') # Hides the 'gender' label on the y-axis for pies
plt.savefig(os.path.join(output_dir, "gender_distribution.png"))
plt.close()

# --- 5. Neurostimulator Status ---
print("Analyzing neurostimulator status...")
status_counts = df_meta['patient_off_on'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ffcc99','#c2c2f0'])
plt.title('Neurostimulator Status (Deep Brain Stimulation)')
plt.ylabel('')
plt.savefig(os.path.join(output_dir, "neurostimulator_status.png"))
plt.close()


# --- 6. UPDRS Diagnosis Score Distribution ---
print("Analyzing UPDRS diagnosis score distribution...")
plt.figure(figsize=(10, 6))
sns.countplot(data=df_meta, x='doctor_diagnosis_0_5', order=sorted(df_meta['doctor_diagnosis_0_5'].unique()), palette="magma")
plt.title('Distribution of UPDRS Diagnosis Scores')
plt.xlabel('UPDRS Score (0-5)')
plt.ylabel('Number of Recordings')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "diagnosis_score_distribution.png"))
plt.close()

# --- FPS Calculation ---
print("Calculating mean FPS...")
td = []
td_mean = []
for table in tqdm(paths_to_tables):
    df = pd.read_csv(table)
    if df.shape[0] > 1:
        time_diffs = df.TIME.diff().dropna()
        td.extend(time_diffs.to_list())

if td:
    mean_time_diff = np.mean(td)
    mean_fps = 1 / mean_time_diff if mean_time_diff > 0 else 0
    print(f'Mean FPS calculated from data: {mean_fps:.2f}')
else:
    print("Could not calculate Mean FPS.")

print("All visualizations have been generated in the 'visualizations/' directory.")
