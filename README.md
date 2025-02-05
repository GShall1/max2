# max2
#import numpy as np  
import pandas as pd 
from sklearn.preprocessing import normalize  

# Read feature info
print('Reading feature info...')
data_info = pd.read_csv('C:\\Users\\Greg\\OneDrive\\Documents\\Documents\\Uni\\year 4\\Final Year Project\\datasets\\NUSW-NB15_features.csv', encoding="ISO-8859-1", header=None).values

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

dataset1 = pd.read_csv('C:\\Users\\Greg\\OneDrive\\Documents\\Documents\\Uni\\year 4\\Final Year Project\\datasets\\UNSW-NB15_1.csv', header=None, dtype=str)
dataset2 = pd.read_csv('C:\\Users\\Greg\\OneDrive\\Documents\\Documents\\Uni\\year 4\\Final Year Project\\datasets\\UNSW-NB15_2.csv', header=None, dtype=str)
dataset3 = pd.read_csv('C:\\Users\\Greg\\OneDrive\\Documents\\Documents\\Uni\\year 4\\Final Year Project\\datasets\\UNSW-NB15_3.csv', header=None, dtype=str)
dataset4 = pd.read_csv('C:\\Users\\Greg\\OneDrive\\Documents\\Documents\\Uni\\year 4\\Final Year Project\\datasets\\UNSW-NB15_4.csv', header=None, dtype=str)


dataset = pd.concat([dataset1, dataset2, dataset3, dataset4], ignore_index=True)

# Clean up memory
del dataset1, dataset2, dataset3, dataset4

# Convert columns that should be numeric
for col in dataset.columns:
    try:
        dataset[col] = pd.to_numeric(dataset[col])  # Convert only if possible
    except ValueError:
        pass  # Ignore errors for text columns
dataset[integer_cols] = dataset[integer_cols].apply(pd.to_numeric, errors='coerce')
dataset[binary_cols] = dataset[binary_cols].apply(pd.to_numeric, errors='coerce')
dataset[float_cols] = dataset[float_cols].apply(pd.to_numeric, errors='coerce')
dataset[48] = pd.to_numeric(dataset[48], errors='coerce')

# Replace NaN values with normal
dataset = dataset.fillna('normal')

# Ensure feature_names matches dataset
if dataset.shape[1] != len(feature_names):
    print(f"Mismatch! Dataset columns: {dataset.shape[1]}, Feature names: {len(feature_names)}")
else:
    dataset.columns = feature_names  # Assign only if they match



train = dataset.copy()
test = dataset.copy()
combine = pd.concat([train], ignore_index=True)
combine = combine.reset_index(drop=True)
combine = combine.drop(['Label', 'attack_cat'], axis=1)
combine['is_ftp_login'] = np.where(combine['is_ftp_login'] > 1, 1,
combine['is_ftp_login'])
combine['service'] = combine['service'].apply(lambda x: "None" if x == "-" else x)

combine.isnull().sum()
