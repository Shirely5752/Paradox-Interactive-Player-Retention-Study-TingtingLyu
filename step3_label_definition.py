import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

plt.style.use('seaborn-v0_8-whitegrid')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'online_gaming_behavior_dataset.csv')
df = pd.read_csv(DATA_PATH)

# ============================================================
# 3.1 Define Hard Churn
# ============================================================
print("=" * 60)
print("3.1 HARD CHURN DEFINITION")
print("=" * 60)

# Based on EDA: EngagementLevel == "Low" players show significantly
# lower SessionsPerWeek (4.53) and AvgSessionDurationMinutes (66.88)
# compared to Medium and High groups. This is supported by data.

df['hard_churn'] = (df['EngagementLevel'] == 'Low').astype(int)

print(f"\nHard Churn = EngagementLevel == 'Low'")
print(f"\nDistribution:")
print(df['hard_churn'].value_counts())
print(f"\nChurn rate: {df['hard_churn'].mean():.2%}")

# ============================================================
# 3.2 Soft Churn Threshold Analysis
# ============================================================
print("\n" + "=" * 60)
print("3.2 SOFT CHURN THRESHOLD ANALYSIS")
print("=" * 60)

# EDA showed SessionsPerWeek and AvgSessionDurationMinutes are the
# two strongest signals. Look at Low group distribution to set thresholds.

low = df[df['EngagementLevel'] == 'Low']
medium = df[df['EngagementLevel'] == 'Medium']
high = df[df['EngagementLevel'] == 'High']

for feat in ['SessionsPerWeek', 'AvgSessionDurationMinutes']:
    print(f"\n{feat}:")
    for name, group in [('Low', low), ('Medium', medium), ('High', high)]:
        print(f"  {name:8s} | 25th: {group[feat].quantile(0.25):6.1f} | median: {group[feat].quantile(0.50):6.1f} | 75th: {group[feat].quantile(0.75):6.1f}")

# Visualize overlap between groups
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, feat in enumerate(['SessionsPerWeek', 'AvgSessionDurationMinutes']):
    for name, group, color in [('Low', low, 'red'), ('Medium', medium, 'blue'), ('High', high, 'green')]:
        axes[i].hist(group[feat], bins=30, alpha=0.4, label=name, color=color, density=True)
    threshold = low[feat].quantile(0.75)
    axes[i].axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Low 75th: {threshold:.0f}')
    axes[i].set_title(feat)
    axes[i].set_xlabel(feat)
    axes[i].set_ylabel('Density')
    axes[i].legend()
plt.suptitle('Distribution Overlap: Setting Soft Churn Thresholds', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '07_soft_churn_thresholds.png'), dpi=150)
plt.close()

# ============================================================
# 3.3 Define Soft Churn
# ============================================================
print("\n" + "=" * 60)
print("3.3 SOFT CHURN DEFINITION")
print("=" * 60)

session_threshold = low['SessionsPerWeek'].quantile(0.75)
duration_threshold = low['AvgSessionDurationMinutes'].quantile(0.75)
print(f"\nThresholds (Low group 75th percentile):")
print(f"  SessionsPerWeek <= {session_threshold:.0f}")
print(f"  AvgSessionDurationMinutes <= {duration_threshold:.0f}")

# Soft churn: not yet Low, but behavior looks like Low
non_low = df[df['EngagementLevel'] != 'Low']
soft_churn_mask = ((non_low['SessionsPerWeek'] <= session_threshold) & 
                   (non_low['AvgSessionDurationMinutes'] <= duration_threshold))

print(f"\nSoft churn among non-Low players:")
print(f"  Total non-Low: {len(non_low)}")
print(f"  Soft churn:    {soft_churn_mask.sum()} ({soft_churn_mask.mean():.2%})")

print(f"\nBreakdown by EngagementLevel:")
non_low_copy = non_low.copy()
non_low_copy['soft_churn'] = soft_churn_mask.astype(int)
print(non_low_copy.groupby('EngagementLevel')['soft_churn'].agg(['count', 'sum', 'mean']).round(4).to_string())

# ============================================================
# 3.4 Final Label System
# ============================================================
print("\n" + "=" * 60)
print("3.4 FINAL LABEL SYSTEM")
print("=" * 60)

df['label'] = 'active'
df.loc[df['EngagementLevel'] == 'Low', 'label'] = 'hard_churn'
df.loc[
    (df['EngagementLevel'] != 'Low') & 
    (df['SessionsPerWeek'] <= session_threshold) & 
    (df['AvgSessionDurationMinutes'] <= duration_threshold), 
    'label'
] = 'soft_churn'

print(f"\nLabel distribution:")
print(df['label'].value_counts().to_string())
print(f"\nLabel proportions:")
print(df['label'].value_counts(normalize=True).round(3).to_string())

# Verify: soft churn behavior should be between active and hard churn
print(f"\nBehavior verification (mean values):")
verify_cols = ['SessionsPerWeek', 'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
print(df.groupby('label')[verify_cols].mean().round(2).loc[['active', 'soft_churn', 'hard_churn']].to_string())

# Visualize label system
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i, feat in enumerate(['SessionsPerWeek', 'AvgSessionDurationMinutes']):
    for name, color in [('active', 'green'), ('soft_churn', 'orange'), ('hard_churn', 'red')]:
        subset = df[df['label'] == name]
        axes[i].hist(subset[feat], bins=30, alpha=0.4, label=f"{name} (n={len(subset)})", color=color, density=True)
    axes[i].set_title(feat)
    axes[i].legend()
plt.suptitle('Label System Verification', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '08_label_verification.png'), dpi=150)
plt.close()

print(f"\nPlots saved:")
print(f"  - 07_soft_churn_thresholds.png")
print(f"  - 08_label_verification.png")
print(f"\nStep 3 complete.")
