import pandas as pd
import numpy as np
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'online_gaming_behavior_dataset.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'processed_data.csv')
df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 60)

# ============================================================
# 4.1 Create Labels (from Step 3)
# ============================================================
print("\n--- 4.1 Create Labels ---")

df['hard_churn'] = (df['EngagementLevel'] == 'Low').astype(int)

low = df[df['EngagementLevel'] == 'Low']
session_threshold = low['SessionsPerWeek'].quantile(0.75)
duration_threshold = low['AvgSessionDurationMinutes'].quantile(0.75)

df['label'] = 'active'
df.loc[df['EngagementLevel'] == 'Low', 'label'] = 'hard_churn'
df.loc[
    (df['EngagementLevel'] != 'Low') &
    (df['SessionsPerWeek'] <= session_threshold) &
    (df['AvgSessionDurationMinutes'] <= duration_threshold),
    'label'
] = 'soft_churn'

print(f"Labels created: {df['label'].value_counts().to_dict()}")

# ============================================================
# 4.2 Feature Selection
# ============================================================
print("\n--- 4.2 Feature Selection ---")

# Based on EDA:
# - Significant: SessionsPerWeek, AvgSessionDurationMinutes, PlayerLevel, AchievementsUnlocked
# - Not significant but keep for model: Age, PlayTimeHours, InGamePurchases
# - Categorical: Gender, Location, GameGenre, GameDifficulty
# - Drop: PlayerID (identifier), EngagementLevel (used to create label)

drop_cols = ['PlayerID', 'EngagementLevel']
print(f"Dropping: {drop_cols}")

# ============================================================
# 4.3 Encode Categorical Variables
# ============================================================
print("\n--- 4.3 Encode Categorical Variables ---")

# One-hot encoding for nominal categories (no ordinal relationship)
cat_cols = ['Gender', 'Location', 'GameGenre', 'GameDifficulty']
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

print(f"One-hot encoded: {cat_cols}")
print(f"New columns added:")
for col in df_encoded.columns:
    if any(cat in col for cat in cat_cols):
        print(f"  - {col}")

# ============================================================
# 4.4 Final Feature Set
# ============================================================
print("\n--- 4.4 Final Feature Set ---")

# Features for hard churn prediction model
feature_cols = [col for col in df_encoded.columns 
                if col not in ['PlayerID', 'EngagementLevel', 'hard_churn', 'label']]

print(f"\nFeatures ({len(feature_cols)}):")
for col in feature_cols:
    print(f"  - {col}")

# ============================================================
# 4.5 Check Final Data
# ============================================================
print("\n--- 4.5 Final Data Check ---")
print(f"Shape: {df_encoded.shape}")
print(f"Missing values: {df_encoded[feature_cols].isnull().sum().sum()}")
print(f"\nTarget (hard_churn) distribution:")
print(f"  0 (active): {(df_encoded['hard_churn']==0).sum()} ({(df_encoded['hard_churn']==0).mean():.2%})")
print(f"  1 (churn):  {(df_encoded['hard_churn']==1).sum()} ({(df_encoded['hard_churn']==1).mean():.2%})")

# ============================================================
# 4.6 Save Processed Data
# ============================================================
df_encoded.to_csv(OUTPUT_PATH, index=False)
print(f"\nProcessed data saved to: {OUTPUT_PATH}")
print(f"\nStep 4 complete.")
