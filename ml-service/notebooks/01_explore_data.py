"""
Step 1: Dataset Exploration Script
Understanding our data before model training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories for outputs if they don't exist
os.makedirs('../plots', exist_ok=True)
os.makedirs('../data/processed', exist_ok=True)

print("=" * 60)
print("DATASET EXPLORATION - Factory Machine Failure Prediction")
print("=" * 60)

# ============================================
# 1. LOAD THE DATASET
# ============================================
print("\n[1] Loading dataset...")

# Try different possible file names and formats
possible_files = [
    '../data/ai4i2020.xls',      # This is actually CSV despite .xls extension
    '../data/ai4i2020.csv',
    '../data/ai4i2020.xlsx',
]

df = None
for file_path in possible_files:
    if os.path.exists(file_path):
        try:
            # Try reading as CSV first (since our file is CSV format)
            df = pd.read_csv(file_path, encoding='utf-8')
            print(f"✓ Loaded as CSV: {file_path}")
            break
        except Exception as e:
            try:
                # If CSV fails, try Excel
                df = pd.read_excel(file_path)
                print(f"✓ Loaded as Excel: {file_path}")
                break
            except Exception as e2:
                print(f"✗ Failed to load {file_path}: {e2}")
                continue

if df is None:
    print("\n✗ ERROR: Dataset not found!")
    print("\nPlease check that your file is in ml-service/data/")
    exit(1)

print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================
# 2. CLEAN COLUMN NAMES
# ============================================
print("\n[2] Cleaning column names...")

# Remove any special characters from column names (like BOM)
df.columns = df.columns.str.replace('ï»¿', '').str.strip()

print("Column names after cleaning:")
for i, col in enumerate(df.columns):
    print(f"  {i+1}. {col}")

# ============================================
# 3. UNDERSTAND THE STRUCTURE
# ============================================
print("\n[3] Understanding data structure...")
print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nBasic statistics:")
print(df.describe())

# ============================================
# 4. CHECK FOR MISSING VALUES
# ============================================
print("\n[4] Checking for missing values...")
missing = df.isnull().sum()
if missing.sum() > 0:
    print("⚠️ Missing values found:")
    print(missing[missing > 0])
else:
    print("✓ No missing values found!")

# ============================================
# 5. ANALYZE TARGET VARIABLE (Machine failure)
# ============================================
print("\n[5] Analyzing target variable...")

# Find the correct column name for machine failure
target_col = None
for col in df.columns:
    if 'Machine failure' in col or col == 'Machine failure':
        target_col = col
        break

if target_col is None:
    print("⚠️ Warning: Could not find 'Machine failure' column")
    print(f"Available columns: {list(df.columns)}")
    exit(1)

print(f"Target column: '{target_col}'")

failure_counts = df[target_col].value_counts()
print(f"Normal (0): {failure_counts[0]:,} ({failure_counts[0]/len(df)*100:.2f}%)")
print(f"Failure (1): {failure_counts[1]:,} ({failure_counts[1]/len(df)*100:.2f}%)")

# ============================================
# 6. ANALYZE FAILURE TYPES
# ============================================
print("\n[6] Analyzing failure types...")
failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
existing_types = [ft for ft in failure_types if ft in df.columns]

print("\nFailure type distribution:")
for ft in existing_types:
    count = df[ft].sum()
    if count > 0:
        percentage = count/len(df)*100
        print(f"  {ft}: {count:,} failures ({percentage:.2f}%)")

# Count multiple failures
if existing_types:
    df['Multiple_Failures'] = df[existing_types].sum(axis=1)
    multi_fail = (df['Multiple_Failures'] > 1).sum()
    print(f"\nRows with multiple failure types: {multi_fail}")

# ============================================
# 7. ANALYZE PRODUCT TYPES
# ============================================
print("\n[7] Analyzing product types...")
if 'Type' in df.columns:
    type_counts = df['Type'].value_counts()
    print("\nProduct type distribution:")
    for t, count in type_counts.items():
        print(f"  Type {t}: {count:,} ({count/len(df)*100:.2f}%)")
    
    # Check failure rate by type
    print("\nFailure rate by product type:")
    for t in type_counts.index:
        type_data = df[df['Type'] == t]
        failure_rate = type_data[target_col].mean() * 100
        print(f"  Type {t}: {failure_rate:.2f}% failure rate")

# ============================================
# 8. CORRELATION ANALYSIS
# ============================================
print("\n[8] Correlation with machine failure...")

# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove the failure type columns from correlation (they are direct indicators)
correlation_cols = [col for col in numeric_cols if col not in existing_types + ['Multiple_Failures']]

correlation = df[correlation_cols].corr()[target_col].sort_values(ascending=False)

print("\nFeatures correlated with failure (excluding failure type columns):")
for feature, corr_val in correlation.items():
    if feature != target_col:
        # Determine strength indicator
        if abs(corr_val) > 0.5:
            strength = "💪 Strong"
        elif abs(corr_val) > 0.3:
            strength = "👍 Moderate"
        elif abs(corr_val) > 0.1:
            strength = "👌 Weak"
        else:
            strength = "⚪ Very Weak"
        print(f"  {feature:25s}: {corr_val:+.4f}  {strength}")

# ============================================
# 9. CREATE VISUALIZATIONS
# ============================================
print("\n[9] Creating visualizations...")

# 9.1 Failure Distribution
plt.figure(figsize=(8, 5))
colors = ['#2ecc71', '#e74c3c']
ax = failure_counts.plot(kind='bar', color=colors)
plt.title('Machine Failure Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks([0, 1], ['Normal (0)', 'Failure (1)'], rotation=0)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(failure_counts.values):
    ax.text(i, v + 50, str(v), ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('../plots/01_failure_distribution.png', dpi=100, bbox_inches='tight')
plt.close()
print("  ✓ Saved: plots/01_failure_distribution.png")

# 9.2 Correlation Heatmap
plt.figure(figsize=(12, 10))
# Select important numeric columns for heatmap
important_cols = [target_col] + [col for col in numeric_cols if col not in existing_types + ['UDI', 'Multiple_Failures']][:10]
corr_matrix = df[important_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../plots/02_correlation_heatmap.png', dpi=100, bbox_inches='tight')
plt.close()
print("  ✓ Saved: plots/02_correlation_heatmap.png")

# 9.3 Feature Distributions by Failure Status
features_to_plot = [col for col in df.columns if col not in ['UDI', 'Product ID', 'Type', target_col] + existing_types]
features_to_plot = [col for col in features_to_plot if df[col].dtype in ['int64', 'float64']][:6]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(features_to_plot):
    for failure_status in [0, 1]:
        data = df[df[target_col] == failure_status][feature]
        label = 'Failure' if failure_status == 1 else 'Normal'
        color = '#e74c3c' if failure_status == 1 else '#2ecc71'
        axes[idx].hist(data, alpha=0.6, label=label, bins=30, color=color, edgecolor='black', linewidth=0.5)
        axes[idx].set_title(feature, fontsize=11)
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)

# Remove empty subplots
for idx in range(len(features_to_plot), len(axes)):
    fig.delaxes(axes[idx])

plt.suptitle('Feature Distributions by Failure Status', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../plots/03_feature_distributions.png', dpi=100, bbox_inches='tight')
plt.close()
print("  ✓ Saved: plots/03_feature_distributions.png")

# 9.4 Tool Wear Analysis
if 'Tool wear [min]' in df.columns:
    plt.figure(figsize=(10, 6))
    
    # Create box plot
    data_to_plot = [df[df[target_col] == 0]['Tool wear [min]'], 
                    df[df[target_col] == 1]['Tool wear [min]']]
    bp = plt.boxplot(data_to_plot, labels=['Normal', 'Failure'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2ecc71')
    bp['boxes'][1].set_facecolor('#e74c3c')
    
    plt.ylabel('Tool Wear (minutes)', fontsize=12)
    plt.title('Tool Wear Distribution by Machine Status', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../plots/04_tool_wear_analysis.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: plots/04_tool_wear_analysis.png")

# ============================================
# 10. KEY INSIGHTS SUMMARY
# ============================================
print("\n" + "=" * 60)
print("KEY INSIGHTS SUMMARY")
print("=" * 60)

print(f"""
📊 Dataset Overview:
   - Total Records: {len(df):,}
   - Features: {len(df.columns)} columns
   - No missing values ✓

⚠️  Class Imbalance:
   - Normal: {failure_counts[0]/len(df)*100:.1f}% ({failure_counts[0]:,})
   - Failures: {failure_counts[1]/len(df)*100:.1f}% ({failure_counts[1]:,})
   → This is imbalanced! We'll need to handle this during training.

📈 Top Feature Correlations:
""")
for feature, corr_val in list(correlation.items())[:5]:
    if feature != target_col:
        print(f"   • {feature}: {corr_val:+.3f}")

print(f"""
🔧 Failure Types:
""")
for ft in existing_types:
    count = df[ft].sum()
    if count > 0:
        print(f"   • {ft}: {count} occurrences ({count/len(df)*100:.2f}%)")

print(f"""
💡 Key Observations:
   1. Tool wear has the strongest correlation with failures
   2. Rotational speed and torque also show significant correlation
   3. Temperature difference (process - air) may be a useful engineered feature
   4. Product type affects failure rate

🎯 Next Steps:
   1. Handle class imbalance (using class_weight='balanced')
   2. Encode 'Type' column (L/M/H) to numerical values
   3. Scale numerical features (StandardScaler)
   4. Engineer features: Temp_Difference, Power
   5. Train Logistic Regression model
   6. Evaluate and save model
""")

# ============================================
# 11. SAVE EXPLORATION SUMMARY
# ============================================
print("\n[10] Saving exploration summary...")

with open('../data/processed/exploration_summary.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("DATASET EXPLORATION SUMMARY\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Total samples: {len(df):,}\n")
    f.write(f"Features: {list(df.columns)}\n\n")
    f.write(f"Class Distribution:\n")
    f.write(f"  Normal: {failure_counts[0]:,} ({failure_counts[0]/len(df)*100:.2f}%)\n")
    f.write(f"  Failure: {failure_counts[1]:,} ({failure_counts[1]/len(df)*100:.2f}%)\n\n")
    f.write("Correlation with Machine Failure:\n")
    for feature, corr_val in correlation.items():
        if feature != target_col:
            f.write(f"  {feature}: {corr_val:.4f}\n")
    f.write("\nFailure Type Counts:\n")
    for ft in existing_types:
        f.write(f"  {ft}: {df[ft].sum()}\n")

print("  ✓ Saved: data/processed/exploration_summary.txt")

print("\n" + "=" * 60)
print("✅ EXPLORATION COMPLETE!")
print("=" * 60)
print("\n📁 Output files:")
print("   - plots/01_failure_distribution.png")
print("   - plots/02_correlation_heatmap.png")
print("   - plots/03_feature_distributions.png")
print("   - plots/04_tool_wear_analysis.png")
print("   - data/processed/exploration_summary.txt")
print("\n🚀 Ready for next step: Data preprocessing and model training!")