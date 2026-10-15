#!/usr/bin/env python3
"""
Generate diagnostic plots for Phase 6.1 Clean Evaluation
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
summary_path = os.path.join(workspace_root, "evaluation_assets", "phase6_clean", "phase6_double_jump_multi_seed_summary.json")

with open(summary_path) as f:
    data = json.load(f)

trajectories = data["trajectories"]

# Plot 1: Jump 1 vs Jump 2 Clearance Distribution and Consistency
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

labels = [f"S{t['seed']}_E{t['env_id']}" for t in trajectories]
j1_clrs = [t["jump1"]["clearance_cm"] for t in trajectories]
j2_clrs = [t["jump2"]["clearance_cm"] for t in trajectories]
deltas = [abs(t["jump2"]["clearance_cm"] - t["jump1"]["clearance_cm"]) for t in trajectories]

x = np.arange(len(labels))
width = 0.35

rects1 = ax1.bar(x - width/2, j1_clrs, width, label='Jump 1 Clearance', color='#2b5c8f', alpha=0.9)
rects2 = ax1.bar(x + width/2, j2_clrs, width, label='Jump 2 Clearance', color='#e26d5c', alpha=0.9)
ax1.axhline(20.0, color='red', linestyle='--', linewidth=1.5, label='20 cm Redline Threshold')
ax1.set_ylabel('Exact 3D Clearance [cm]', fontsize=11)
ax1.set_title('Phase 6.1 Clean Evaluation: Jump 1 vs Jump 2 Clearance Across 10 Trajectories', fontsize=11, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax1.set_ylim(0, 28)
ax1.legend(loc='lower left', fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Add values above bars
for r in rects1:
    h = r.get_height()
    ax1.annotate(f"{h:.1f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2),
                textcoords="offset points", ha='center', va='bottom', fontsize=7.5)
for r in rects2:
    h = r.get_height()
    ax1.annotate(f"{h:.1f}", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2),
                textcoords="offset points", ha='center', va='bottom', fontsize=7.5)

# Plot 2: Absolute Consistency Delta (|Jump 2 - Jump 1|)
colors = ['#2ca02c' if d < 0.5 else '#1f77b4' for d in deltas]
rects3 = ax2.bar(x, [d * 10.0 for d in deltas], width=0.5, color=colors, alpha=0.85) # in mm
ax2.set_ylabel('Clearance Difference |Jump2 - Jump1| [mm]', fontsize=11)
ax2.set_title('Double Jump Consistency Metric (Mean Delta: 5.1 mm)', fontsize=11, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
ax2.set_ylim(0, 15)
ax2.grid(axis='y', alpha=0.3)

for r in rects3:
    h = r.get_height()
    ax2.annotate(f"{h:.1f} mm", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 2),
                textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
out_plot1 = os.path.join(workspace_root, "evaluation_assets", "phase6_clean", "plots", "phase6_clean_double_jump_consistency.png")
os.makedirs(os.path.dirname(out_plot1), exist_ok=True)
plt.savefig(out_plot1, dpi=200, bbox_inches='tight')
print(f"Saved plot to {out_plot1}")

brain_plot_dir = "/home/xy/.gemini/antigravity/brain/a711c2de-3c50-48db-b87a-241c57720d52/plots"
os.makedirs(brain_plot_dir, exist_ok=True)
import shutil
shutil.copyfile(out_plot1, os.path.join(brain_plot_dir, "phase6_clean_double_jump_consistency.png"))
print("Copied to brain artifact directory")
