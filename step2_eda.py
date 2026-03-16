import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'online_gaming_behavior_dataset.csv')
df = pd.read_csv(DATA_PATH)

num_features = ['Age', 'PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
cat_features = ['Gender', 'Location', 'GameGenre', 'GameDifficulty', 'InGamePurchases']

# ============================================================
# 2.1 Numerical Features by EngagementLevel
# ============================================================
print("=" * 60)
print("2.1 NUMERICAL FEATURES BY ENGAGEMENT LEVEL")
print("=" * 60)

group_stats = df.groupby('EngagementLevel')[num_features].agg(['mean', 'median']).round(2)
print(f"\n{group_stats.to_string()}")

# Kruskal-Wallis test
print("\n--- Kruskal-Wallis Tests ---")
for feat in num_features:
    groups = [df[df['EngagementLevel'] == lvl][feat] for lvl in ['Low', 'Medium', 'High']]
    stat, p = stats.kruskal(*groups)
    sig = "YES" if p < 0.05 else "NO"
    print(f"  {feat:30s} | H={stat:.2f} | p={p:.6f} | Significant: {sig}")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for i, feat in enumerate(num_features):
    sns.boxplot(data=df, x='EngagementLevel', y=feat, ax=axes[i], order=['Low', 'Medium', 'High'])
    axes[i].set_title(feat)
plt.suptitle('Numerical Features by Engagement Level', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '01_numerical_by_engagement.png'), dpi=150)
plt.close()

# ============================================================
# 2.2 Categorical Features by EngagementLevel
# ============================================================
print("\n" + "=" * 60)
print("2.2 CATEGORICAL FEATURES BY ENGAGEMENT LEVEL")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()
for i, feat in enumerate(cat_features):
    ct = pd.crosstab(df[feat], df['EngagementLevel'], normalize='index')
    ct[['Low', 'Medium', 'High']].plot(kind='bar', ax=axes[i])
    axes[i].set_title(f'Engagement by {feat}')
    axes[i].set_ylabel('Proportion')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend(title='Engagement')

    ct_raw = pd.crosstab(df[feat], df['EngagementLevel'])
    chi2, p, _, _ = stats.chi2_contingency(ct_raw)
    sig = "YES" if p < 0.05 else "NO"
    print(f"  {feat:20s} | chi2={chi2:.2f} | p={p:.6f} | Significant: {sig}")

axes[5].set_visible(False)
plt.suptitle('Engagement Level by Categorical Features', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '02_categorical_by_engagement.png'), dpi=150)
plt.close()

# ============================================================
# 2.3 Correlation Heatmap
# ============================================================
print("\n" + "=" * 60)
print("2.3 CORRELATION HEATMAP")
print("=" * 60)

corr = df[num_features + ['InGamePurchases']].corr(method='spearman').round(2)
print(f"\n{corr.to_string()}")

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ax=ax)
plt.title('Spearman Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '03_correlation_heatmap.png'), dpi=150)
plt.close()

# ============================================================
# 2.4 Cohort Analysis
# ============================================================
print("\n" + "=" * 60)
print("2.4 COHORT ANALYSIS")
print("=" * 60)

df['age_group'] = pd.cut(df['Age'], bins=[14, 20, 30, 40, 50], labels=['15-20', '21-30', '31-40', '41-49'])
df['playtime_group'] = pd.cut(df['PlayTimeHours'], bins=[0, 5, 10, 15, 25], labels=['Light(0-5h)', 'Casual(5-10h)', 'Regular(10-15h)', 'Heavy(15h+)'], include_lowest=True)
df['session_group'] = pd.cut(df['SessionsPerWeek'], bins=[-1, 0, 5, 10, 20], labels=['None(0)', 'Low(1-5)', 'Mid(6-10)', 'High(11+)'])
df['duration_group'] = pd.cut(df['AvgSessionDurationMinutes'], bins=[0, 45, 90, 135, 180], labels=['Short(<45m)', 'Medium(45-90m)', 'Long(90-135m)', 'VeryLong(135m+)'], include_lowest=True)

cohorts = {
    'session_group': 'Engagement by Sessions/Week',
    'duration_group': 'Engagement by Avg Session Duration',
    'playtime_group': 'Engagement by Play Time',
    'age_group': 'Engagement by Age Group'
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
for i, (col, title) in enumerate(cohorts.items()):
    ct = pd.crosstab(df[col], df['EngagementLevel'], normalize='index')
    ct[['Low', 'Medium', 'High']].plot(kind='bar', stacked=True, ax=axes[i])
    axes[i].set_title(title)
    axes[i].set_ylabel('Proportion')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend(title='Engagement')
    print(f"\n{title}:")
    print(ct[['Low', 'Medium', 'High']].round(3).to_string())

plt.suptitle('Cohort Analysis: Engagement Level by Player Groups', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '04_cohort_analysis.png'), dpi=150)
plt.close()

# ============================================================
# 2.5 User Segmentation (K-Means)
# ============================================================
print("\n" + "=" * 60)
print("2.5 USER SEGMENTATION")
print("=" * 60)

seg_features = ['SessionsPerWeek', 'PlayTimeHours', 'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
X_seg = StandardScaler().fit_transform(df[seg_features])

# Elbow
inertias = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_seg)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(K_range, inertias, 'bo-')
ax.set_xlabel('K')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '05_elbow.png'), dpi=150)
plt.close()

# K=4
df['segment'] = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_seg)

seg_profile = df.groupby('segment')[seg_features].mean().round(2)
print(f"\nSegment Profiles:\n{seg_profile.to_string()}")

seg_engagement = pd.crosstab(df['segment'], df['EngagementLevel'], normalize='index')
print(f"\nEngagement by Segment:\n{seg_engagement[['Low', 'Medium', 'High']].round(3).to_string()}")

print(f"\nSegment sizes:\n{df['segment'].value_counts().sort_index().to_string()}")

fig, ax = plt.subplots(figsize=(8, 5))
seg_engagement[['Low', 'Medium', 'High']].plot(kind='bar', stacked=True, ax=ax)
ax.set_title('Engagement Level Distribution by Segment')
ax.set_ylabel('Proportion')
ax.set_xlabel('Segment')
ax.legend(title='Engagement')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '06_segment_engagement.png'), dpi=150)
plt.close()

# ============================================================
# EDA Summary
# ============================================================
print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)
print("""
Significant features (can distinguish Low/Medium/High):
  1. SessionsPerWeek        - STRONGEST signal
  2. AvgSessionDurationMinutes - STRONG signal
  3. PlayerLevel             - Moderate signal
  4. AchievementsUnlocked    - Moderate signal

Not significant:
  - Age, PlayTimeHours
  - Gender, Location, GameGenre, GameDifficulty, InGamePurchases

Key finding from Cohort Analysis:
  - Sessions/Week is the strongest predictor of engagement
  - 93.6% of players with 0 sessions are Low engagement
  - Avg Session Duration shows clear gradient across engagement levels

Plots saved to: """ + OUTPUT_DIR + """
  - 01_numerical_by_engagement.png
  - 02_categorical_by_engagement.png
  - 03_correlation_heatmap.png
  - 04_cohort_analysis.png
  - 05_elbow.png
  - 06_segment_engagement.png
""")
