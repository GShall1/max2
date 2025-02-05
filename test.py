import numpy as np  
import pandas as pd 
from pathlib import Path

# Read feature info
print('Reading feature info...')
data_info = pd.read_csv(r"dataset\NUSW-NB15_features.csv", encoding="ISO-8859-1", header=None).values

# Extract features correctly
features = data_info[1:, :]  # Keep all rows after the first one
feature_names = features[:, 1]  # 49 names
print(f"Extracted {len(feature_names)} feature names.")  # Debugging check

feature_types = np.array([item.lower() for item in features[:, 2]])  

# Index arrays for different types of features
nominal_cols = np.where(feature_types == "nominal")[0]
integer_cols = np.where(feature_types == "integer")[0]
binary_cols = np.where(feature_types == "binary")[0]
float_cols = np.where(feature_types == "float")[0]

# Load dataset
dataset1 = pd.read_csv(r'dataset\UNSW-NB15_1.csv', header=None, dtype=str)
dataset2 = pd.read_csv(r'dataset\UNSW-NB15_2.csv', header=None, dtype=str)

# Combine datasets
dataset = pd.concat([dataset1, dataset2], ignore_index=True)

# Convert columns to numeric where applicable
for col in dataset.columns:
    dataset[col] = pd.to_numeric(dataset[col], errors='ignore')  # Convert only if possible

dataset[integer_cols] = dataset[integer_cols].apply(pd.to_numeric, errors='coerce')
dataset[binary_cols] = dataset[binary_cols].apply(pd.to_numeric, errors='coerce')
dataset[float_cols] = dataset[float_cols].apply(pd.to_numeric, errors='coerce')
dataset[48] = pd.to_numeric(dataset[48], errors='coerce')

# Replace NaN values with 'normal'
dataset = dataset.fillna('normal')

# Ensure feature_names matches dataset columns
if dataset.shape[1] != len(feature_names):
    print(f"Mismatch! Dataset columns: {dataset.shape[1]}, Feature names: {len(feature_names)}")
else:
    dataset.columns = feature_names  # Assign column names only if they match

# Copy dataset for training/testing
train = dataset.copy()
test = dataset.copy()
combine = pd.concat([train], ignore_index=True)
combine = combine.reset_index(drop=True)

# Drop specified columns only if they exist
columns_to_drop = ['Label', 'attack_cat']
combine = combine.drop(columns=[col for col in columns_to_drop if col in combine.columns], errors='ignore')

# Fix 'is_ftp_login' column
combine['is_ftp_login'] = pd.to_numeric(combine['is_ftp_login'], errors='coerce')
combine['is_ftp_login'] = np.where(combine['is_ftp_login'] > 1, 1, combine['is_ftp_login'])

# Fix 'service' column
combine['service'] = combine['service'].apply(lambda x: "None" if x == "-" else x)

# Check for null values
print(combine.isnull().sum())

# Save processed data to CSV
combine.to_csv("processed_dataset.csv", index=False)
print("Processed dataset saved as 'processed_dataset.csv'.")

# Print dataset
print(dataset)
