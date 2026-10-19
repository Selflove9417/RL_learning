#!/usr/bin/env python3
import json

with open("/tmp/audit_trajectories/traj_seed_100.json") as f:
    d100 = json.load(f)
with open("/tmp/audit_trajectories/traj_seed_17.json") as f:
    d17 = json.load(f)
with open("/tmp/audit_trajectories/traj_seed_400.json") as f:
    d400 = json.load(f)
with open("/tmp/audit_trajectories/traj_seed_2.json") as f:
    d2 = json.load(f)

print(f"Takeoff steps: 100={d100['takeoff_step']}, 17={d17['takeoff_step']}, 400={d400['takeoff_step']}, 2={d2['takeoff_step']}")

for s in [42, 100, 200, 300, 400, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 50, 150, 250, 350, 450]:
    with open(f"/tmp/audit_trajectories/traj_seed_{s}.json") as f:
        d = json.load(f)
    print(f"Seed {s:3d}: takeoff_step={d['takeoff_step']}, touchdown_step={d['touchdown_step']}")
