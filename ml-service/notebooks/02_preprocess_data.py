"""
Step 2: Data Preprocessing for Production
We'll create preprocessing pipelines that can be saved and reused
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# Get the project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_SERVICE = PROJECT_ROOT
DATA_DIR = os.path.join(ML_SERVICE, 'data')
MODEL_DIR = os.path.join(ML_SERVICE, 'model')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("=" * 60)
print("DATA PREPROCESSING - Factory Machine Failure Prediction")
print("=" * 60)

# ============================================
# 1. LOAD THE DATASET
# ============================================
print("\n[1] Loading dataset...")

# Try different possible locations
data_file = os.path.join(DATA_DIR, 'ai4i2020.xls')
if not os.path.exists(data_file):
    data_file = os.path.join(DATA_DIR, 'ai4i2020.csv')

df = pd.read_csv(data_file, encoding='utf-8')
print(f"✓ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Clean column names
df.columns = df.columns.str.replace('ï»¿', '').str.strip()

# ============================================
# 2. DROP UNNECESSARY COLUMNS
# ============================================
print("\n[2] Dropping unnecessary columns...")

# UDI is just an index, Product ID has too many unique values
columns_to_drop = ['UDI', 'Product ID']
df = df.drop(columns=columns_to_drop)
print(f"✓ Dropped: {columns_to_drop}")
print(f"  Remaining columns: {list(df.columns)}")

# ============================================
# 3. ENCODE CATEGORICAL VARIABLES
# ============================================
print("\n[3] Encoding categorical variables...")

# 'Type' column: L, M, H -> numerical values
label_encoder = LabelEncoder()
df['Type'] = label_encoder.fit_transform(df['Type'])
print(f"✓ Type encoding mapping:")
for i, type_name in enumerate(label_encoder.classes_):
    print(f"  {type_name} -> {i}")

# ============================================
# 4. FEATURE ENGINEERING
# ============================================
print("\n[4] Creating engineered features...")

# Temperature difference (how much the machine heats up)
df['Temp_Difference'] = df['Process temperature [K]'] - df['Air temperature [K]']
print(f"✓ Created: Temp_Difference (process - air)")

# Power = Torque × Rotational speed (actual machine power)
df['Power'] = df['Torque [Nm]'] * df['Rotational speed [rpm]']
print(f"✓ Created: Power (Torque × Rotational speed)")

# Normalized tool wear (percentage of max possible)
# Assuming max tool wear is 250 minutes based on data
df['Tool_Wear_Percent'] = (df['Tool wear [min]'] / 250) * 100
print(f"✓ Created: Tool_Wear_Percent (wear % of max)")

print(f"\nTotal features now: {len(df.columns)}")

# ============================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================
print("\n[5] Separating features and target...")

# Define feature columns (excluding target and failure type columns)
failure_type_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
feature_columns = [col for col in df.columns if col not in failure_type_columns and col != 'Machine failure']

X = df[feature_columns]
y = df['Machine failure']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature columns:")
for i, col in enumerate(feature_columns):
    print(f"  {i+1}. {col}")

# ============================================
# 6. SPLIT DATA (STRATIFIED TO MAINTAIN FAILURE RATIO)
# ============================================
print("\n[6] Splitting data into train and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # VERY IMPORTANT! Maintains the 3.4% failure rate in both sets
)

print(f"Training set size: {len(X_train):,} records")
print(f"  - Failures in training: {y_train.sum():,} ({y_train.sum()/len(y_train)*100:.2f}%)")
print(f"Test set size: {len(X_test):,} records")
print(f"  - Failures in test: {y_test.sum():,} ({y_test.sum()/len(y_test)*100:.2f}%)")

# ============================================
# 7. SCALE FEATURES
# ============================================
print("\n[7] Scaling features...")

# Important: Fit scaler ONLY on training data, then transform both
scaler = StandardScaler()

# Fit on training data only
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)

print(f"✓ Features scaled")
print(f"  Mean of scaled features: {X_train_scaled.mean(axis=0).round(3)[:3]}...")
print(f"  Std of scaled features: {X_train_scaled.std(axis=0).round(3)[:3]}...")

# ============================================
# 8. VERIFY CLASS IMBALANCE
# ============================================
print("\n[8] Class imbalance check...")

failure_rate_train = y_train.sum() / len(y_train) * 100
failure_rate_test = y_test.sum() / len(y_test) * 100

print(f"Training set failure rate: {failure_rate_train:.2f}%")
print(f"Test set failure rate: {failure_rate_test:.2f}%")

if failure_rate_train < 5:
    print("⚠️ Class imbalance detected! Will use 'class_weight=balanced' during training")

# ============================================
# 9. SAVE PREPROCESSING ARTIFACTS
# ============================================
print("\n[9] Saving preprocessing artifacts...")

# Save scaler
scaler_path = os.path.join(MODEL_DIR, 'scaler.pkl')
joblib.dump(scaler, scaler_path)
print(f"✓ Scaler saved to: {scaler_path}")

# Save label encoder
encoder_path = os.path.join(MODEL_DIR, 'label_encoder.pkl')
joblib.dump(label_encoder, encoder_path)
print(f"✓ Label encoder saved to: {encoder_path}")

# Save feature names (important for prediction API)
feature_names_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
joblib.dump(feature_columns, feature_names_path)
print(f"✓ Feature names saved to: {feature_names_path}")

# Save processed data for training
train_data = {
    'X_train': X_train_scaled,
    'X_test': X_test_scaled,
    'y_train': y_train.values,
    'y_test': y_test.values,
    'feature_names': feature_columns
}

processed_data_path = os.path.join(PROCESSED_DIR, 'processed_data.pkl')
joblib.dump(train_data, processed_data_path)
print(f"✓ Processed data saved to: {processed_data_path}")

# Save raw splits for reference
X_train.to_csv(os.path.join(PROCESSED_DIR, 'X_train_raw.csv'), index=False)
X_test.to_csv(os.path.join(PROCESSED_DIR, 'X_test_raw.csv'), index=False)
pd.Series(y_train).to_csv(os.path.join(PROCESSED_DIR, 'y_train.csv'), index=False)
pd.Series(y_test).to_csv(os.path.join(PROCESSED_DIR, 'y_test.csv'), index=False)
print(f"✓ Raw splits saved to: {PROCESSED_DIR}/")

# ============================================
# 10. PREPROCESSING SUMMARY
# ============================================
print("\n" + "=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)

print(f"""
✅ Completed:
   1. Dropped: UDI, Product ID
   2. Encoded: Type (L→{label_encoder.transform(['L'])[0]}, 
                      M→{label_encoder.transform(['M'])[0]}, 
                      H→{label_encoder.transform(['H'])[0]})
   3. Engineered features:
      - Temp_Difference (process - air temp)
      - Power (torque × rotational speed)
      - Tool_Wear_Percent (wear as percentage)
   4. Train/Test split: {len(X_train):,} / {len(X_test):,}
   5. Features scaled using StandardScaler

📊 Final Feature Set ({len(feature_columns)} features):
""")
for i, col in enumerate(feature_columns, 1):
    print(f"   {i:2d}. {col}")

print(f"""
🎯 Next Step: Train model with:
   - Features: {len(feature_columns)} columns
   - Training samples: {len(X_train):,}
   - Test samples: {len(X_test):,}
   - Class imbalance: Will use class_weight='balanced'
""")

print("=" * 60)
print("✅ PREPROCESSING COMPLETE!")
print("=" * 60)
print("\nProceed to: 03_train_model.py")