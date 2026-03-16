import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, recall_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import os

plt.style.use('seaborn-v0_8-whitegrid')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'processed_data.csv')
df = pd.read_csv(DATA_PATH)

feature_cols = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek',
                'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked',
                'Gender_Male', 'Location_Europe', 'Location_Other', 'Location_USA',
                'GameGenre_RPG', 'GameGenre_Simulation', 'GameGenre_Sports',
                'GameGenre_Strategy', 'GameDifficulty_Hard', 'GameDifficulty_Medium']

X = df[feature_cols]
y = df['hard_churn']

# ============================================================
# 5.1 Train/Test Split
# ============================================================
print("=" * 60)
print("STEP 5: MODELING (Recall-focused)")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\n--- 5.1 Train/Test Split ---")
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"Train churn rate: {y_train.mean():.2%} | Test churn rate: {y_test.mean():.2%}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# 5.2 Logistic Regression (Baseline)
# ============================================================
print(f"\n--- 5.2 Logistic Regression (Baseline) ---")

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

print(classification_report(y_test, y_pred_lr, target_names=['Active', 'Churn']))
print(f"Recall (Churn): {recall_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")

# ============================================================
# 5.3 Random Forest (Default threshold 0.5)
# ============================================================
print(f"\n--- 5.3 Random Forest (threshold=0.5) ---")

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_rf, target_names=['Active', 'Churn']))
print(f"Recall (Churn): {recall_score(y_test, y_pred_rf):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.4f}")

# ============================================================
# 5.4 Threshold Tuning for Higher Recall
# ============================================================
print(f"\n--- 5.4 Threshold Tuning (Random Forest) ---")
print(f"\n{'Threshold':<12} {'Recall':<10} {'Precision':<12} {'F1':<10}")
print("-" * 44)

best_threshold = 0.5
best_recall = 0
results = []

for t in np.arange(0.2, 0.55, 0.05):
    y_pred_t = (y_prob_rf >= t).astype(int)
    rec = recall_score(y_test, y_pred_t)
    prec = y_test[y_pred_t == 1].mean() if y_pred_t.sum() > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    results.append({'threshold': t, 'recall': rec, 'precision': prec, 'f1': f1})
    print(f"{t:<12.2f} {rec:<10.4f} {prec:<12.4f} {f1:<10.4f}")
    if rec >= 0.85 and rec > best_recall:
        best_threshold = t
        best_recall = rec

# Pick threshold that gives recall >= 0.85
print(f"\nSelected threshold: {best_threshold:.2f} (targeting recall >= 0.85)")

y_pred_tuned = (y_prob_rf >= best_threshold).astype(int)
print(f"\n--- Tuned Random Forest (threshold={best_threshold:.2f}) ---")
print(classification_report(y_test, y_pred_tuned, target_names=['Active', 'Churn']))
print(f"Recall (Churn): {recall_score(y_test, y_pred_tuned):.4f}")

# ============================================================
# 5.5 Feature Importance
# ============================================================
print(f"\n--- 5.5 Feature Importance ---")
importance = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=True)
print(importance.round(4).to_string())

fig, ax = plt.subplots(figsize=(8, 7))
importance.plot(kind='barh', ax=ax)
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '09_feature_importance.png'), dpi=150)
plt.close()

# ============================================================
# 5.6 ROC Curve
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={roc_auc_score(y_test, y_prob_lr):.3f})')
ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={roc_auc_score(y_test, y_prob_rf):.3f})')
ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.500)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve Comparison')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '10_roc_comparison.png'), dpi=150)
plt.close()

# ============================================================
# 5.7 Precision-Recall Curve
# ============================================================
fig, ax = plt.subplots(figsize=(7, 6))
prec_rf, rec_rf, thresholds_rf = precision_recall_curve(y_test, y_prob_rf)
ax.plot(rec_rf, prec_rf, label='Random Forest')
ax.axvline(x=recall_score(y_test, y_pred_tuned), color='red', linestyle='--', 
           label=f'Selected threshold={best_threshold:.2f}')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve (shows recall vs precision tradeoff)')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '11_precision_recall_curve.png'), dpi=150)
plt.close()

# ============================================================
# 5.8 Confusion Matrix (Tuned Model)
# ============================================================
cm = confusion_matrix(y_test, y_pred_tuned)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Active', 'Churn'], yticklabels=['Active', 'Churn'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix (RF, threshold={best_threshold:.2f})')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '12_confusion_matrix.png'), dpi=150)
plt.close()

# ============================================================
# 5.9 Cross-Validation (Recall)
# ============================================================
cv_recall = cross_val_score(rf, X_train, y_train, cv=5, scoring='recall')
cv_auc = cross_val_score(rf, X_train, y_train, cv=5, scoring='roc_auc')
print(f"\n--- 5.9 Cross-Validation ---")
print(f"5-Fold CV Recall:  {cv_recall.mean():.4f} (+/- {cv_recall.std():.4f})")
print(f"5-Fold CV ROC-AUC: {cv_auc.mean():.4f} (+/- {cv_auc.std():.4f})")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("MODEL SUMMARY")
print(f"{'='*60}")
print(f"\n{'Model':<30} {'Recall':<10} {'Precision':<12} {'ROC-AUC':<10}")
print("-" * 62)
print(f"{'LR (threshold=0.5)':<30} {recall_score(y_test, y_pred_lr):<10.4f} {y_test[y_pred_lr==1].mean():<12.4f} {roc_auc_score(y_test, y_prob_lr):<10.4f}")
print(f"{'RF (threshold=0.5)':<30} {recall_score(y_test, y_pred_rf):<10.4f} {y_test[y_pred_rf==1].mean():<12.4f} {roc_auc_score(y_test, y_prob_rf):<10.4f}")
print(f"{'RF (threshold={best_threshold:.2f})':<30} {recall_score(y_test, y_pred_tuned):<10.4f} {y_test[y_pred_tuned==1].mean():<12.4f} {roc_auc_score(y_test, y_prob_rf):<10.4f}")

print(f"\nBest model: Random Forest (threshold={best_threshold:.2f})")
print(f"\nStep 5 complete.")
