"""
Step 3: Train Logistic Regression Model
We'll train, evaluate, and save the model for production
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_auc_score, roc_curve, precision_recall_curve,
                            accuracy_score, precision_score, recall_score, f1_score)
import os

# Get paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ML_SERVICE = PROJECT_ROOT
MODEL_DIR = os.path.join(ML_SERVICE, 'model')
PLOTS_DIR = os.path.join(ML_SERVICE, 'plots')
DATA_DIR = os.path.join(ML_SERVICE, 'data', 'processed')

# Create plots directory if needed
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 60)
print("MODEL TRAINING - Factory Machine Failure Prediction")
print("=" * 60)

# ============================================
# 1. LOAD PREPROCESSED DATA
# ============================================
print("\n[1] Loading preprocessed data...")

# Load the processed data we saved in Step 2
processed_data = joblib.load(os.path.join(DATA_DIR, 'processed_data.pkl'))

X_train = processed_data['X_train']
X_test = processed_data['X_test']
y_train = processed_data['y_train']
y_test = processed_data['y_test']
feature_names = processed_data['feature_names']

print(f"✓ Training data: {X_train.shape[0]:,} samples")
print(f"✓ Test data: {X_test.shape[0]:,} samples")
print(f"✓ Features: {len(feature_names)}")
print(f"\nFeatures used for training:")
for i, f in enumerate(feature_names, 1):
    print(f"  {i:2d}. {f}")

# ============================================
# 2. TRAIN LOGISTIC REGRESSION MODEL
# ============================================
print("\n[2] Training Logistic Regression model...")

model = LogisticRegression(
    class_weight='balanced',  # Handles class imbalance
    max_iter=1000,            # More than enough iterations
    random_state=42,          # Reproducible results
    C=1.0                     # Regularization strength
)

# Train the model
model.fit(X_train, y_train)
print("✓ Model training completed")

# ============================================
# 3. MAKE PREDICTIONS
# ============================================
print("\n[3] Making predictions...")

# Predict class (0 or 1)
y_pred = model.predict(X_test)

# Predict probability (0 to 1) - we'll use this for threshold tuning
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"✓ Predictions made for {len(y_pred):,} test samples")

# ============================================
# 4. CALCULATE METRICS
# ============================================
print("\n[4] Calculating performance metrics...")

# Basic metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n📊 Model Performance:")
print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-Score:  {f1:.4f}")
print(f"   ROC-AUC:   {roc_auc:.4f}")

# ============================================
# 5. CONFUSION MATRIX
# ============================================
print("\n[5] Confusion Matrix:")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n   Actual: NO FAILURE  → Predicted NO: {tn:4d} (Correct)")
print(f"   Actual: NO FAILURE  → Predicted YES: {fp:4d} (False Alarm)")
print(f"   Actual: FAILURE     → Predicted NO: {fn:4d} (Missed Failure) ⚠️")
print(f"   Actual: FAILURE     → Predicted YES: {tp:4d} (Caught Failure) ✓")

# Calculate rates
false_alarm_rate = fp / (tn + fp) * 100 if (tn + fp) > 0 else 0
missed_failure_rate = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0

print(f"\n   False Alarm Rate: {false_alarm_rate:.2f}% ({fp} false alarms)")
print(f"   Missed Failure Rate: {missed_failure_rate:.2f}% ({fn} missed)")

# ============================================
# 6. DETAILED CLASSIFICATION REPORT
# ============================================
print("\n[6] Detailed Classification Report:")
print(classification_report(y_test, y_pred, 
                           target_names=['Normal (0)', 'Failure (1)']))

# ============================================
# 7. MODEL COEFFICIENTS (Feature Importance)
# ============================================
print("\n[7] Model Coefficients (Feature Importance):")

# Get coefficients (m values) for each feature
coefficients = model.coef_[0]
intercept = model.intercept_[0]

print(f"\n   Intercept (c): {intercept:.4f}")
print(f"\n   Coefficients (m values):")
print(f"   {'Feature':<30} {'Coefficient':>12} {'Impact':>12}")
print(f"   {'-'*30} {'-'*12} {'-'*12}")

# Sort features by absolute coefficient (importance)
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients,
    'abs_coef': np.abs(coefficients)
}).sort_values('abs_coef', ascending=False)

for _, row in coef_df.iterrows():
    impact = "⬆️ Increases risk" if row['coefficient'] > 0 else "⬇️ Decreases risk"
    print(f"   {row['feature']:<30} {row['coefficient']:>+12.4f} {impact:>12}")

# ============================================
# 8. VISUALIZATIONS
# ============================================
print("\n[8] Creating visualizations...")

# 8.1 Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Failure'],
            yticklabels=['Normal', 'Failure'])
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('Actual', fontsize=12)
plt.xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '05_confusion_matrix.png'), dpi=100)
plt.close()
print("  ✓ Saved: plots/05_confusion_matrix.png")

# 8.2 ROC Curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, 'b-', label=f'Logistic Regression (AUC = {roc_auc:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlabel('False Positive Rate (False Alarms)', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '06_roc_curve.png'), dpi=100)
plt.close()
print("  ✓ Saved: plots/06_roc_curve.png")

# 8.3 Precision-Recall Curve
plt.figure(figsize=(8, 6))
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall_vals, precision_vals, 'g-', linewidth=2)
plt.xlabel('Recall (Failure Detection Rate)', fontsize=12)
plt.ylabel('Precision (Prediction Accuracy)', fontsize=12)
plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '07_precision_recall_curve.png'), dpi=100)
plt.close()
print("  ✓ Saved: plots/07_precision_recall_curve.png")

# 8.4 Feature Importance Bar Chart
plt.figure(figsize=(10, 6))
colors = ['green' if c > 0 else 'red' for c in coef_df['coefficient'].head(10)]
plt.barh(coef_df['feature'].head(10), coef_df['coefficient'].head(10), color=colors)
plt.xlabel('Coefficient Value', fontsize=12)
plt.title('Top 10 Feature Importance', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, '08_feature_importance.png'), dpi=100)
plt.close()
print("  ✓ Saved: plots/08_feature_importance.png")

# ============================================
# 9. THRESHOLD TUNING (Find Best Balance)
# ============================================
print("\n[9] Finding optimal prediction threshold...")

# Test different thresholds
thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
best_recall = 0
best_threshold = 0.3
best_precision = 0

print(f"\n   {'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print(f"   {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

for threshold in thresholds_to_test:
    y_pred_tuned = (y_pred_proba >= threshold).astype(int)
    p = precision_score(y_test, y_pred_tuned, zero_division=0)
    r = recall_score(y_test, y_pred_tuned)
    f = f1_score(y_test, y_pred_tuned)
    print(f"   {threshold:>10.1f} {p:>10.4f} {r:>10.4f} {f:>10.4f}")

# Recommended threshold
recommended_threshold = 0.3  # Based on typical factory needs
y_pred_recommended = (y_pred_proba >= recommended_threshold).astype(int)
recall_recommended = recall_score(y_test, y_pred_recommended)

print(f"\n📌 Recommended threshold: 0.3")
print(f"   At threshold 0.3, recall = {recall_recommended:.2%}")

# ============================================
# 10. SAVE MODEL AND METADATA
# ============================================
print("\n[10] Saving model and metadata...")

# Save the trained model
model_path = os.path.join(MODEL_DIR, 'failure_prediction_model.pkl')
joblib.dump(model, model_path)
print(f"✓ Model saved to: {model_path}")

# Save model metadata
model_metadata = {
    'model_type': 'LogisticRegression',
    'features': feature_names.tolist() if hasattr(feature_names, 'tolist') else feature_names,
    'class_weight': 'balanced',
    'roc_auc': roc_auc,
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'threshold_recommended': recommended_threshold,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'failure_rate_train': y_train.mean() * 100,
    'failure_rate_test': y_test.mean() * 100,
    'coefficients': dict(zip(feature_names, coefficients))
}

metadata_path = os.path.join(MODEL_DIR, 'model_metadata.pkl')
joblib.dump(model_metadata, metadata_path)
print(f"✓ Model metadata saved to: {metadata_path}")

# ============================================
# 11. TEST WITH SAMPLE DATA (FIXED)
# ============================================
print("\n[11] Testing with sample data...")

# Load scaler and feature names for verification
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
feature_names_loaded = joblib.load(os.path.join(MODEL_DIR, 'feature_names.pkl'))

print(f"   Expected features ({len(feature_names_loaded)}):")
for i, f in enumerate(feature_names_loaded):
    print(f"      {i+1}. {f}")

# Create sample with ALL 9 features (in the correct order!)
print("\n   Sample 1: High Risk Machine (likely to fail)")
high_risk = np.array([[
    1,       # Type (M = 1)
    300.0,   # Air temperature [K]
    310.0,   # Process temperature [K]
    1500.0,  # Rotational speed [rpm]
    55.0,    # Torque [Nm] (HIGH)
    220.0,   # Tool wear [min] (HIGH)
    10.0,    # Temp_Difference (Process - Air)
    82500.0, # Power (Torque × Speed)
    88.0     # Tool_Wear_Percent (220/250 × 100)
]])

# Scale the sample
high_risk_scaled = scaler.transform(high_risk)
probability = model.predict_proba(high_risk_scaled)[0][1]
prediction = "⚠️ FAILURE" if probability > recommended_threshold else "✓ NORMAL"

print(f"   Failure Probability: {probability:.2%}")
print(f"   Prediction: {prediction}")

print("\n   Sample 2: Low Risk Machine (likely normal)")
low_risk = np.array([[
    1,       # Type (M = 1)
    300.0,   # Air temperature [K]
    310.0,   # Process temperature [K]
    1400.0,  # Rotational speed [rpm]
    35.0,    # Torque [Nm] (normal)
    50.0,    # Tool wear [min] (new tool)
    10.0,    # Temp_Difference
    49000.0, # Power
    20.0     # Tool_Wear_Percent (50/250 × 100)
]])

low_risk_scaled = scaler.transform(low_risk)
probability = model.predict_proba(low_risk_scaled)[0][1]
prediction = "⚠️ FAILURE" if probability > recommended_threshold else "✓ NORMAL"

print(f"   Failure Probability: {probability:.2%}")
print(f"   Prediction: {prediction}")

print("\n   Sample 3: Medium Risk Machine (borderline)")
medium_risk = np.array([[
    0,       # Type (L = 0)
    301.0,   # Air temperature [K]
    311.0,   # Process temperature [K]
    1600.0,  # Rotational speed [rpm]
    42.0,    # Torque [Nm]
    150.0,   # Tool wear [min] (worn but not extreme)
    10.0,    # Temp_Difference
    67200.0, # Power
    60.0     # Tool_Wear_Percent (150/250 × 100)
]])

medium_risk_scaled = scaler.transform(medium_risk)
probability = model.predict_proba(medium_risk_scaled)[0][1]
prediction = "⚠️ FAILURE" if probability > recommended_threshold else "✓ NORMAL"

print(f"   Failure Probability: {probability:.2%}")
print(f"   Prediction: {prediction}")
# ============================================
# 12. SUMMARY
# ============================================
print("\n" + "=" * 60)
print("TRAINING SUMMARY")
print("=" * 60)

print(f"""
✅ Model Training Complete!

📊 Model Performance:
   - Recall (Caught Failures): {recall:.2%}
   - Precision (Accurate Warnings): {precision:.2%}
   - F1-Score: {f1:.3f}
   - ROC-AUC: {roc_auc:.3f}

🎯 Key Findings:
   - Most important features:
     1. {coef_df.iloc[0]['feature']} (coefficient: {coef_df.iloc[0]['coefficient']:+.3f})
     2. {coef_df.iloc[1]['feature']} (coefficient: {coef_df.iloc[1]['coefficient']:+.3f})
     3. {coef_df.iloc[2]['feature']} (coefficient: {coef_df.iloc[2]['coefficient']:+.3f})

📁 Saved Files:
   - model/failure_prediction_model.pkl  (trained model)
   - model/model_metadata.pkl            (performance data)
   - plots/05_confusion_matrix.png
   - plots/06_roc_curve.png
   - plots/07_precision_recall_curve.png
   - plots/08_feature_importance.png

🚀 Ready for Production!
   The model can now be used by your FastAPI service.
""")

print("=" * 60)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 60)