import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'plots')

fig, ax = plt.subplots(figsize=(18, 10))
ax.set_xlim(0, 18)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(9, 9.5, 'Data Roles: From Raw Data to Production', 
        fontsize=16, fontweight='bold', ha='center', va='center')

# ============================================================
# Main pipeline boxes (top row)
# ============================================================
roles = [
    {'x': 1.5, 'label': 'Data\nEngineer', 'color': '#4ECDC4', 'short': 'DE'},
    {'x': 5.5, 'label': 'Analytics\nEngineer', 'color': '#FFE66D', 'short': 'AE'},
    {'x': 9.5, 'label': 'Data\nAnalyst', 'color': '#FF6B6B', 'short': 'DA'},
    {'x': 13.0, 'label': 'Data\nScientist', 'color': '#A8E6CF', 'short': 'DS'},
    {'x': 16.0, 'label': 'ML\nEngineer', 'color': '#DDA0DD', 'short': 'MLE'},
]

box_w, box_h = 2.4, 1.6
y_top = 7.2

for r in roles:
    rect = mpatches.FancyBboxPatch((r['x'] - box_w/2, y_top - box_h/2), box_w, box_h,
                                     boxstyle="round,pad=0.15", facecolor=r['color'], 
                                     edgecolor='#333333', linewidth=2)
    ax.add_patch(rect)
    ax.text(r['x'], y_top, r['label'], fontsize=12, fontweight='bold',
            ha='center', va='center', color='#333333')

# Arrows between boxes
arrow_y = y_top
for i in range(len(roles) - 1):
    x_start = roles[i]['x'] + box_w/2 + 0.05
    x_end = roles[i+1]['x'] - box_w/2 - 0.05
    # Skip arrow between DA and DS (they branch from AE)
    if i == 2:
        continue
    ax.annotate('', xy=(x_end, arrow_y), xytext=(x_start, arrow_y),
                arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

# Branch arrows from AE to DA and DS
ax.annotate('', xy=(roles[2]['x'] - box_w/2 - 0.05, y_top), 
            xytext=(roles[1]['x'] + box_w/2 + 0.05, y_top),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

ax.annotate('', xy=(roles[3]['x'] - box_w/2 - 0.05, y_top), 
            xytext=(roles[1]['x'] + box_w/2 + 0.05, y_top + 0.0),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=2, 
                          connectionstyle='arc3,rad=-0.3'))

# Arrow from DS to MLE
ax.annotate('', xy=(roles[4]['x'] - box_w/2 - 0.05, y_top), 
            xytext=(roles[3]['x'] + box_w/2 + 0.05, y_top),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=2))

# ============================================================
# What they do (middle row)
# ============================================================
tasks = [
    {'x': 1.5, 'text': 'Build data pipelines\nExtract raw data\nLoad into warehouse'},
    {'x': 5.5, 'text': 'Data modeling (dbt)\nClean & standardize\nSingle source of truth'},
    {'x': 9.5, 'text': 'BI dashboards\nVisualize trends\nReport to stakeholders'},
    {'x': 13.0, 'text': 'ML models\nPredict future\nExperiment in Jupyter'},
    {'x': 16.0, 'text': 'Deploy models\nDocker, CI/CD, K8s\n24/7 production'},
]

y_mid = 4.8
for t in tasks:
    ax.text(t['x'], y_mid, t['text'], fontsize=9, ha='center', va='center',
            color='#555555', style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#F8F8F8', edgecolor='#CCCCCC'))

# ============================================================
# Data flow labels (bottom row)
# ============================================================
flow = [
    {'x': 0.3, 'text': 'Raw Data\n(logs, XML,\nAPIs)'},
    {'x': 3.5, 'text': 'Data\nWarehouse\n(Snowflake)'},
    {'x': 7.5, 'text': 'Clean\nModeled\nTables'},
    {'x': 11.2, 'text': 'Dashboards\n& Reports'},
    {'x': 14.5, 'text': 'Trained\nModel\n(prototype)'},
    {'x': 17.2, 'text': 'Production\nSystem\n(live)'},
]

y_bot = 2.5
for f in flow:
    ax.text(f['x'], y_bot, f['text'], fontsize=8, ha='center', va='center',
            color='#777777',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#AAAAAA', linestyle='--'))

# Flow arrows at bottom
for i in range(len(flow) - 1):
    ax.annotate('', xy=(flow[i+1]['x'] - 0.8, y_bot), 
                xytext=(flow[i]['x'] + 0.8, y_bot),
                arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=1.5, linestyle='--'))

# ============================================================
# Key distinction callout
# ============================================================
ax.text(14.5, 3.8, 'DS cares: "Is it accurate?"\nMLE cares: "Is it stable?"', 
        fontsize=8, ha='center', va='center', color='#8B0000', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF0F0', edgecolor='#8B0000'))

# Row labels
ax.text(0.15, y_top, 'ROLES', fontsize=9, fontweight='bold', color='#999999', rotation=90, va='center')
ax.text(0.15, y_mid, 'TASKS', fontsize=9, fontweight='bold', color='#999999', rotation=90, va='center')
ax.text(0.15, y_bot, 'DATA', fontsize=9, fontweight='bold', color='#999999', rotation=90, va='center')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'data_roles_pipeline.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: data_roles_pipeline.png")
