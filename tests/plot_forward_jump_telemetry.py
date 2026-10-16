#!/usr/bin/env python3
# ==============================================================================
# MyRobot Forward Jump: Telemetry Plotting Utility
# ==============================================================================

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
csv_path = os.path.join(workspace_root, "evaluation_assets", "phase1_forward_jump", "forward_jump_telemetry.csv")
plot_path = os.path.join(workspace_root, "evaluation_assets", "phase1_forward_jump", "forward_jump_telemetry_plot.png")

if not os.path.exists(csv_path):
    print(f"Error: CSV file not found: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

t = df['time_s']

# 1. Height & Clearance
ax1 = axes[0]
ax1.plot(t, df['root_z'], label='Root Z (m)', color='#1f77b4', lw=2)
ax1.plot(t, df['clr_min_cm'] / 100.0, label='Min Clearance (m)', color='#2ca02c', lw=2, linestyle='--')
ax1.axhline(0.20, color='r', linestyle=':', label='20cm Target Baseline')
airborne_mask = df['is_airborne'] == 1
if airborne_mask.any():
    t_start = df.loc[airborne_mask, 'time_s'].iloc[0]
    t_end = df.loc[airborne_mask, 'time_s'].iloc[-1]
    ax1.axvspan(t_start, t_end, color='orange', alpha=0.15, label='Airborne Window')
ax1.set_ylabel('Height / Clearance (m)', fontsize=11, fontweight='bold')
ax1.set_title('MyRobot Phase 1 Forward Jump: Kinematics & Physical Telemetry', fontsize=13, fontweight='bold')
ax1.legend(loc='upper right', frameon=True)
ax1.grid(True, linestyle='--', alpha=0.6)

# 2. Forward Position Y & Velocity vy
ax2 = axes[1]
color_pos = '#9467bd'
color_vel = '#d62728'
ax2.plot(t, df['root_y'] - df['root_y'].iloc[0], label='Forward Disp ΔY (m)', color=color_pos, lw=2)
ax2_twin = ax2.twinx()
ax2_twin.plot(t, df['vel_y'], label='Forward Vel vy (m/s)', color=color_vel, lw=2, linestyle='-.')
ax2.set_ylabel('Forward Position ΔY (m)', color=color_pos, fontsize=11, fontweight='bold')
ax2_twin.set_ylabel('Forward Vel vy (m/s)', color=color_vel, fontsize=11, fontweight='bold')
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)
ax2.grid(True, linestyle='--', alpha=0.6)

# 3. Attitude: Pitch & Roll
ax3 = axes[2]
ax3.plot(t, df['pitch_deg'], label='Pitch (deg)', color='#ff7f0e', lw=2)
ax3.plot(t, df['roll_deg'], label='Roll (deg)', color='#17becf', lw=2)
ax3.plot(t, df['yaw_deg'], label='Yaw (deg)', color='#8c564b', lw=1.5, linestyle=':')
ax3.axhline(0, color='gray', linestyle='--', alpha=0.7)
ax3.set_ylabel('Attitude (deg)', fontsize=11, fontweight='bold')
ax3.legend(loc='upper right', frameon=True)
ax3.grid(True, linestyle='--', alpha=0.6)

# 4. Joint Torques
ax4 = axes[3]
ax4.plot(t, df['tau_knee_L'], label='Knee L (Nm)', color='#e377c2', lw=1.8)
ax4.plot(t, df['tau_hip_L'], label='Hip L (Nm)', color='#bcbd22', lw=1.8)
ax4.plot(t, df['tau_wheel_L'], label='Wheel L (Nm)', color='#333333', lw=1.8, linestyle='--')
ax4.axhline(120.0, color='#e377c2', linestyle=':', alpha=0.7, label='Knee Limit (120Nm)')
ax4.axhline(-120.0, color='#e377c2', linestyle=':', alpha=0.7)
ax4.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Joint Torque (Nm)', fontsize=11, fontweight='bold')
ax4.legend(loc='upper right', ncol=2, frameon=True)
ax4.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(plot_path, dpi=200)
print(f"✅ 遥测分析曲线已导出至: {plot_path}")
