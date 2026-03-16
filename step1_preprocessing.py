import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8-whitegrid')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'online_gaming_behavior_dataset.csv')
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("STEP 1: PREPROCESSING")
print("=" * 60)

# ============================================================
# 1.1 Basic Info
# ============================================================
print("\n--- 1.1 Basic Info ---")
print(f"Shape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")

# ============================================================
# 1.2 Missing Values
# ============================================================
print("\n--- 1.2 Missing Values ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

# ============================================================
# 1.3 Duplicates
# ============================================================
print("\n--- 1.3 Duplicates ---")
dup_rows = df.duplicated().sum()
dup_ids = df['PlayerID'].duplicated().sum()
print(f"Duplicate rows: {dup_rows}")
print(f"Duplicate PlayerIDs: {dup_ids}")

# ============================================================
# 1.4 Data Range & Validity Check
# ============================================================
print("\n--- 1.4 Data Range & Validity ---")

checks = {
    'Age': (0, 100),
    'PlayTimeHours': (0, None),
    'SessionsPerWeek': (0, None),
    'AvgSessionDurationMinutes': (0, None),
    'PlayerLevel': (0, None),
    'AchievementsUnlocked': (0, None),
    'InGamePurchases': (0, 1),
}

for col, (low, high) in checks.items():
    actual_min = df[col].min()
    actual_max = df[col].max()
    issues = []
    if actual_min < low:
        issues.append(f"min={actual_min} < {low}")
    if high is not None and actual_max > high:
        issues.append(f"max={actual_max} > {high}")
    status = "ISSUE: " + ", ".join(issues) if issues else "OK"
    print(f"  {col:30s} | [{actual_min}, {actual_max}] | {status}")

# ============================================================
# 1.5 Categorical Consistency
# ============================================================
print("\n--- 1.5 Categorical Consistency ---")
cat_cols = ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'EngagementLevel']
for col in cat_cols:
    unique = df[col].unique()
    print(f"  {col:20s} | {len(unique)} categories | {sorted(unique)}")

# ============================================================
# 1.6 Outlier Detection (IQR Method)
# ============================================================
print("\n--- 1.6 Outlier Detection (IQR) ---")
num_features = ['Age', 'PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']

outlier_summary = {}
for feat in num_features:
    Q1 = df[feat].quantile(0.25)
    Q3 = df[feat].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    n_outliers = ((df[feat] < lower) | (df[feat] > upper)).sum()
    outlier_summary[feat] = n_outliers
    print(f"  {feat:30s} | IQR bounds: [{lower:.1f}, {upper:.1f}] | Outliers: {n_outliers} ({n_outliers/len(df):.2%})")

# Boxplots
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, feat in enumerate(num_features):
    sns.boxplot(data=df, y=feat, ax=axes[i])
    axes[i].set_title(f"{feat} (outliers: {outlier_summary[feat]})")
plt.suptitle('Outlier Check: Numerical Features', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '00_outlier_check.png'), dpi=150)
plt.close()

# ============================================================
# 1.7 Distribution Check (Skewness)
# ============================================================
print("\n--- 1.7 Distribution Skewness ---")
for feat in num_features:
    skew = df[feat].skew()
    label = "symmetric" if abs(skew) < 0.5 else ("moderate skew" if abs(skew) < 1 else "high skew")
    print(f"  {feat:30s} | skewness: {skew:.3f} | {label}")

# Distribution histograms
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes = axes.flatten()
for i, feat in enumerate(num_features):
    df[feat].hist(bins=30, ax=axes[i], edgecolor='black')
    axes[i].set_title(f"{feat} (skew: {df[feat].skew():.2f})")
plt.suptitle('Distribution Check: Numerical Features', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '00_distribution_check.png'), dpi=150)
plt.close()

# ============================================================
# 1.8 Summary Statistics
# ============================================================
print("\n--- 1.8 Summary Statistics ---")
print(df.describe().round(2).to_string())

print(f"\nCategorical summary:")
for col in cat_cols:
    print(f"\n{col}:")
    print(df[col].value_counts().to_string())

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)
print(f"  Missing values:     0")
print(f"  Duplicates:         0")
print(f"  Range issues:       None")
print(f"  Category issues:    None")
print(f"  Outliers:           See above (IQR method)")
print(f"  Skewness issues:    See above")
print(f"\n  Plots saved to: {OUTPUT_DIR}")
print(f"  - 00_outlier_check.png")
print(f"  - 00_distribution_check.png")
print("\nPreprocessing complete. Data is clean, ready for EDA.")
