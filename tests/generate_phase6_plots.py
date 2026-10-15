#!/usr/bin/env python3
"""
MyRobot Phase 6: Publication-Quality Plotting Script
Generates:
  1. evaluation_assets/phase6/plots/phase6_mission_timeline.png
  2. evaluation_assets/phase6/plots/phase6_torques_comparison.png
  3. evaluation_assets/phase6/plots/phase6_clearance_and_attitude.png
"""

import os
import sys
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = "evaluation_assets/phase6/double_jump_seed42_telemetry.csv"
SUMMARY_PATH = "evaluation_assets/phase6/phase6_double_jump_multi_seed_summary.json"
PLOTS_DIR = "evaluation_assets/phase6/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 1.8,
    'grid.alpha': 0.4
})

def load_env1_data(csv_path):
    with open(csv_path) as f:
        rows = [r for r in csv.DictReader(f) if r['env_id'] == '1']
    
    t = np.array([float(r['timestamp_s']) for r in rows])
    fsm = np.array([int(r['fsm_state']) for r in rows])
    cmd_vy = np.array([float(r['cmd_fwd_vel_m_s']) for r in rows])
    act_vy = np.array([float(r['actual_fwd_vel_m_s']) for r in rows])
    h_tgt = np.array([float(r['target_height_m']) for r in rows])
    h_act = np.array([float(r['actual_height_m']) for r in rows])
    roll = np.array([float(r['roll_deg']) for r in rows])
    # unwrap roll around 360 if any
    roll = np.where(roll > 180, roll - 360, roll)
    pitch = np.array([float(r['pitch_deg']) for r in rows])
    pitch = np.where(pitch > 180, pitch - 360, pitch)
    yaw_rate = np.array([float(r['yaw_rate_deg_s']) for r in rows])
    
    clr_ex = np.array([float(r['bilateral_clr_cm']) for r in rows])
    clr_L = np.array([float(r['wheel_clr_L_cm']) for r in rows])
    clr_R = np.array([float(r['wheel_clr_R_cm']) for r in rows])
    
    tau_hip_L = np.array([float(r['tau_cmd_hip_L_Nm']) for r in rows])
    tau_knee_L = np.array([float(r['tau_cmd_knee_L_Nm']) for r in rows])
    tau_hip_R = np.array([float(r['tau_cmd_hip_R_Nm']) for r in rows])
    tau_knee_R = np.array([float(r['tau_cmd_knee_R_Nm']) for r in rows])
    tau_wheel_L = np.array([float(r['tau_cmd_wheel_L_Nm']) for r in rows])
    tau_wheel_R = np.array([float(r['tau_cmd_wheel_R_Nm']) for r in rows])
    tau_diff = np.array([float(r['tau_cmd_wheel_diff_Nm']) for r in rows])
    
    fn_L = np.array([float(r['fn_contact_L_N']) for r in rows])
    fn_R = np.array([float(r['fn_contact_R_N']) for r in rows])
    
    return {
        't': t, 'fsm': fsm, 'cmd_vy': cmd_vy, 'act_vy': act_vy,
        'h_tgt': h_tgt, 'h_act': h_act, 'roll': roll, 'pitch': pitch, 'yaw_rate': yaw_rate,
        'clr_ex': clr_ex, 'clr_L': clr_L, 'clr_R': clr_R,
        'tau_hip_L': tau_hip_L, 'tau_knee_L': tau_knee_L,
        'tau_hip_R': tau_hip_R, 'tau_knee_R': tau_knee_R,
        'tau_wheel_L': tau_wheel_L, 'tau_wheel_R': tau_wheel_R, 'tau_diff': tau_diff,
        'fn_L': fn_L, 'fn_R': fn_R
    }

def plot_mission_timeline(d):
    t = d['t']
    fig, axs = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    fig.suptitle("Phase 6: Full Mission Timeline (Stand -> Move -> Jump 1 -> Recover -> Re-Move -> Jump 2 -> Stand >= 3s)", fontsize=13, fontweight='bold')

    # Shaded jump regions
    # Jump 1: ~4.0s to ~6.24s; Jump 2: ~9.0s to ~11.44s
    for ax in axs:
        ax.axvspan(4.0, 6.24, color='#e3f2fd', alpha=0.6, label='Jump 1 Sequence' if ax == axs[0] else None)
        ax.axvspan(9.0, 11.44, color='#f3e5f5', alpha=0.6, label='Jump 2 Sequence' if ax == axs[0] else None)
        ax.axvspan(1.5, 4.0, color='#e8f5e9', alpha=0.3, label='Move 1' if ax == axs[0] else None)
        ax.axvspan(6.5, 9.0, color='#fff9c4', alpha=0.3, label='Move 2 (Re-Move)' if ax == axs[0] else None)
        ax.grid(True, linestyle='--', alpha=0.5)

    # 1. Height & Clearance
    axs[0].plot(t, d['h_act'] * 100.0, 'b-', label='Base Height (cm)')
    axs[0].plot(t, d['clr_ex'], 'm-', linewidth=2.2, label='Bilateral Wheel Clearance (cm)')
    axs[0].axhline(20.0, color='r', linestyle='--', linewidth=1.5, label='20 cm Target Redline')
    axs[0].set_ylabel('Height / Clearance (cm)')
    axs[0].set_title('1. Base Height & 3D Dual-Wheel Clearance')
    axs[0].legend(loc='upper right', ncol=4)
    axs[0].set_ylim(-2, 70)

    # Annotate jump peaks
    axs[0].annotate('Jump 1: 22.1 cm', xy=(4.98, 22.1), xytext=(4.3, 35),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))
    axs[0].annotate('Jump 2: 23.4 cm', xy=(9.98, 23.4), xytext=(9.3, 35),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6))

    # 2. Forward Velocity
    axs[1].plot(t, d['cmd_vy'], 'k--', label='Cmd Fwd Vel (m/s)')
    axs[1].plot(t, d['act_vy'], 'g-', label='Actual Fwd Vel (m/s)')
    axs[1].axhline(0.30, color='gray', linestyle=':', alpha=0.6)
    axs[1].axhline(0.0, color='black', linestyle='-', alpha=0.3)
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].set_title('2. Forward Velocity Tracking & Smooth Stopping Transitions')
    axs[1].legend(loc='upper right')
    axs[1].set_ylim(-0.15, 0.45)

    # 3. FSM State Transitions
    fsm_names = ['BALANCE', 'STOPPING', 'PREP', 'JUMP', 'RECOVER']
    axs[2].step(t, d['fsm'], 'darkorange', where='post', linewidth=2.0, label='FSM State')
    axs[2].set_yticks(range(5))
    axs[2].set_yticklabels(fsm_names)
    axs[2].set_ylabel('FSM State')
    axs[2].set_title('3. Finite State Machine Mode & Safety Gate Progress')
    axs[2].legend(loc='upper right')

    # 4. Attitude & Yaw Rate
    ax4_twin = axs[3].twinx()
    axs[3].plot(t, d['roll'], 'c-', label='Roll (deg)')
    axs[3].plot(t, d['pitch'], 'm-', label='Pitch (deg)')
    ax4_twin.plot(t, d['yaw_rate'], 'k:', label='Yaw Rate (deg/s)', alpha=0.7)
    axs[3].set_ylabel('Attitude (deg)')
    ax4_twin.set_ylabel('Yaw Rate (deg/s)')
    axs[3].set_xlabel('Mission Timestamp (s)')
    axs[3].set_title('4. Robot Attitude Stability & Yaw Dynamics')
    lines1, labels1 = axs[3].get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    axs[3].legend(lines1 + lines2, labels1 + labels2, loc='upper right', ncol=3)
    axs[3].set_ylim(-8, 8)
    ax4_twin.set_ylim(-25, 25)

    plt.tight_layout()
    out_file = os.path.join(PLOTS_DIR, "phase6_mission_timeline.png")
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"--> Saved: {out_file}")

def plot_torques_comparison(d):
    t = d['t']
    # Window for Jump 1: 3.8s to 6.6s (2.8s)
    # Window for Jump 2: 8.8s to 11.6s (2.8s)
    mask1 = (t >= 3.8) & (t <= 6.6)
    mask2 = (t >= 8.8) & (t <= 11.6)

    t1_rel = t[mask1] - 4.0
    t2_rel = t[mask2] - 9.0

    fig, axs = plt.subplots(3, 2, figsize=(14, 9), sharey='row')
    fig.suptitle("Phase 6: Actuator Dynamics Comparison (Jump 1 vs Jump 2 Aligned by Trigger)", fontsize=13, fontweight='bold')

    # Row 1: Knee & Hip Torques
    axs[0, 0].plot(t1_rel, d['tau_knee_L'][mask1], 'b-', label='Knee L (Nm)')
    axs[0, 0].plot(t1_rel, d['tau_knee_R'][mask1], 'r--', label='Knee R (Nm)')
    axs[0, 0].plot(t1_rel, d['tau_hip_L'][mask1], 'c:', label='Hip L (Nm)')
    axs[0, 0].plot(t1_rel, d['tau_hip_R'][mask1], 'm:', label='Hip R (Nm)')
    axs[0, 0].axhline(120, color='r', linestyle=':', alpha=0.6, label='Knee Saturation (120 Nm)')
    axs[0, 0].axhline(-120, color='r', linestyle=':', alpha=0.6)
    axs[0, 0].set_title('Jump 1: Leg Joint Torques')
    axs[0, 0].set_ylabel('Torque (Nm)')
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)
    axs[0, 0].legend(loc='lower right', ncol=2)

    axs[0, 1].plot(t2_rel, d['tau_knee_L'][mask2], 'b-', label='Knee L (Nm)')
    axs[0, 1].plot(t2_rel, d['tau_knee_R'][mask2], 'r--', label='Knee R (Nm)')
    axs[0, 1].plot(t2_rel, d['tau_hip_L'][mask2], 'c:', label='Hip L (Nm)')
    axs[0, 1].plot(t2_rel, d['tau_hip_R'][mask2], 'm:', label='Hip R (Nm)')
    axs[0, 1].axhline(120, color='r', linestyle=':', alpha=0.6)
    axs[0, 1].axhline(-120, color='r', linestyle=':', alpha=0.6)
    axs[0, 1].set_title('Jump 2: Leg Joint Torques')
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)
    axs[0, 1].legend(loc='lower right', ncol=2)

    # Row 2: Wheel Torques & Differential
    axs[1, 0].plot(t1_rel, d['tau_wheel_L'][mask1], 'b-', label='Wheel L (Nm)')
    axs[1, 0].plot(t1_rel, d['tau_wheel_R'][mask1], 'r--', label='Wheel R (Nm)')
    axs[1, 0].plot(t1_rel, d['tau_diff'][mask1], 'k:', label='Wheel Diff (Nm)')
    axs[1, 0].set_title('Jump 1: Wheel Torques & Differential')
    axs[1, 0].set_ylabel('Torque (Nm)')
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)
    axs[1, 0].legend(loc='upper right')

    axs[1, 1].plot(t2_rel, d['tau_wheel_L'][mask2], 'b-', label='Wheel L (Nm)')
    axs[1, 1].plot(t2_rel, d['tau_wheel_R'][mask2], 'r--', label='Wheel R (Nm)')
    axs[1, 1].plot(t2_rel, d['tau_diff'][mask2], 'k:', label='Wheel Diff (Nm)')
    axs[1, 1].set_title('Jump 2: Wheel Torques & Differential')
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)
    axs[1, 1].legend(loc='upper right')

    # Row 3: Ground Normal Contact Forces
    axs[2, 0].plot(t1_rel, d['fn_L'][mask1], 'b-', label='Contact Fn L (N)')
    axs[2, 0].plot(t1_rel, d['fn_R'][mask1], 'r--', label='Contact Fn R (N)')
    axs[2, 0].set_title('Jump 1: Ground Contact Normal Force')
    axs[2, 0].set_ylabel('Force (N)')
    axs[2, 0].set_xlabel('Time relative to Trigger (s)')
    axs[2, 0].grid(True, linestyle='--', alpha=0.5)
    axs[2, 0].legend(loc='upper right')

    axs[2, 1].plot(t2_rel, d['fn_L'][mask2], 'b-', label='Contact Fn L (N)')
    axs[2, 1].plot(t2_rel, d['fn_R'][mask2], 'r--', label='Contact Fn R (N)')
    axs[2, 1].set_title('Jump 2: Ground Contact Normal Force')
    axs[2, 1].set_xlabel('Time relative to Trigger (s)')
    axs[2, 1].grid(True, linestyle='--', alpha=0.5)
    axs[2, 1].legend(loc='upper right')

    plt.tight_layout()
    out_file = os.path.join(PLOTS_DIR, "phase6_torques_comparison.png")
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"--> Saved: {out_file}")

def plot_clearance_and_attitude(summary_path):
    with open(summary_path) as f:
        data = json.load(f)
    trajs = data['trajectories']

    n = len(trajs)
    seeds_envs = [f"s{t['seed']}e{t['env_id']}" for t in trajs]
    j1_clrs = [t['jump1']['clearance_cm'] for t in trajs]
    j2_clrs = [t['jump2']['clearance_cm'] for t in trajs]
    remove_d = [t['re_move']['distance_m'] for t in trajs]

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Phase 6: Multi-Seed Continuous Jump Metrics & Clearance Distribution", fontsize=13, fontweight='bold')

    x = np.arange(n)
    width = 0.35

    # 1. Dual Jump Clearance Comparison
    axs[0, 0].bar(x - width/2, j1_clrs, width, label='Jump 1 Clearance (cm)', color='#42a5f5', edgecolor='black')
    axs[0, 0].bar(x + width/2, j2_clrs, width, label='Jump 2 Clearance (cm)', color='#ab47bc', edgecolor='black')
    axs[0, 0].axhline(20.0, color='r', linestyle='--', linewidth=1.5, label='20 cm Requirement')
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(seeds_envs, rotation=45)
    axs[0, 0].set_ylabel('3D Wheel Clearance (cm)')
    axs[0, 0].set_title('1. Jump 1 vs Jump 2 Clearance Across 10 Trajectories')
    axs[0, 0].legend(loc='lower right')
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)
    axs[0, 0].set_ylim(0, 30)

    # 2. In-Between Re-Move Distance
    bars = axs[0, 1].bar(x, remove_d, color='#66bb6a', edgecolor='black', width=0.5)
    axs[0, 1].axhline(0.20, color='r', linestyle='--', label='Min Move Distance (0.20 m)')
    axs[0, 1].set_xticks(x)
    axs[0, 1].set_xticklabels(seeds_envs, rotation=45)
    axs[0, 1].set_ylabel('Re-Move Distance (m)')
    axs[0, 1].set_title('2. Re-Move Forward Translation Between Jumps')
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)
    axs[0, 1].set_ylim(0, 0.85)
    axs[0, 1].legend(loc='lower right')
    for bar in bars:
        h = bar.get_height()
        axs[0, 1].annotate(f'{h:.2f}m', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3),
                           textcoords="offset points", ha='center', va='bottom', fontsize=8)

    # 3. Liftoff Roll Angle
    j1_roll = [t['jump1']['roll_at_release_deg'] for t in trajs]
    j2_roll = [t['jump2']['roll_at_release_deg'] for t in trajs]
    axs[1, 0].plot(x, j1_roll, 'bo-', label='Jump 1 Release Roll')
    axs[1, 0].plot(x, j2_roll, 'mo--', label='Jump 2 Release Roll')
    axs[1, 0].axhline(3.0, color='r', linestyle=':', label='Roll Gate (+-3.0 deg)')
    axs[1, 0].axhline(-3.0, color='r', linestyle=':')
    axs[1, 0].set_xticks(x)
    axs[1, 0].set_xticklabels(seeds_envs, rotation=45)
    axs[1, 0].set_ylabel('Roll Angle at Release (deg)')
    axs[1, 0].set_title('3. Takeoff Symmetry: Roll Angle at Release Gate')
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)
    axs[1, 0].legend(loc='upper right')
    axs[1, 0].set_ylim(-3.5, 3.5)

    # 4. Touchdown Impact Peak Force Fn
    j1_fn = [t['jump1']['peak_touchdown_fn_N'] for t in trajs]
    j2_fn = [t['jump2']['peak_touchdown_fn_N'] for t in trajs]
    axs[1, 1].plot(x, j1_fn, 's-', color='#ff7043', label='Jump 1 Touchdown Fn')
    axs[1, 1].plot(x, j2_fn, 'd--', color='#8d6e63', label='Jump 2 Touchdown Fn')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(seeds_envs, rotation=45)
    axs[1, 1].set_ylabel('Peak Contact Force Fn (N)')
    axs[1, 1].set_title('4. Touchdown Cushion Impact Force Fn')
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)
    axs[1, 1].legend(loc='upper right')
    axs[1, 1].set_ylim(200, 350)

    plt.tight_layout()
    out_file = os.path.join(PLOTS_DIR, "phase6_clearance_and_attitude.png")
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"--> Saved: {out_file}")

if __name__ == "__main__":
    d = load_env1_data(CSV_PATH)
    plot_mission_timeline(d)
    plot_torques_comparison(d)
    plot_clearance_and_attitude(SUMMARY_PATH)
