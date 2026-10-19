#!/usr/bin/env python3
import os
import struct

mesh_dir = "/home/xy/IsaacTu_jump/legged_gym/resources/robots/myrobot/myrobot_urdf/meshes"

def read_stl_bounds(filepath):
    with open(filepath, 'rb') as f:
        header = f.read(80)
        num_triangles = struct.unpack('<I', f.read(4))[0]
        min_coords = [float('inf'), float('inf'), float('inf')]
        max_coords = [float('-inf'), float('-inf'), float('-inf')]
        for _ in range(num_triangles):
            # normal (12 bytes), 3 vertices (36 bytes), attr count (2 bytes)
            data = f.read(50)
            if len(data) < 50:
                break
            v1 = struct.unpack('<fff', data[12:24])
            v2 = struct.unpack('<fff', data[24:36])
            v3 = struct.unpack('<fff', data[36:48])
            for v in (v1, v2, v3):
                for i in range(3):
                    if v[i] < min_coords[i]: min_coords[i] = v[i]
                    if v[i] > max_coords[i]: max_coords[i] = v[i]
        return min_coords, max_coords, num_triangles

overall_min = [float('inf'), float('inf'), float('inf')]
overall_max = [float('-inf'), float('-inf'), float('-inf')]

files = sorted(os.listdir(mesh_dir))
print(f"Total STL files in {mesh_dir}: {len(files)}")

extreme_files = []

for fn in files:
    if fn.endswith(".stl"):
        fp = os.path.join(mesh_dir, fn)
        cmin, cmax, ntri = read_stl_bounds(fp)
        # scale is 0.001
        cmin = [c * 0.001 for c in cmin]
        cmax = [c * 0.001 for c in cmax]
        for i in range(3):
            if cmin[i] < overall_min[i]: overall_min[i] = cmin[i]
            if cmax[i] > overall_max[i]: overall_max[i] = cmax[i]
        # check if unusually large (> 2 meters)
        if any(abs(c) > 1.0 for c in cmin + cmax):
            extreme_files.append((fn, cmin, cmax))

print(f"Overall mesh bounds (scaled by 0.001, local mesh frame):")
print(f"  Min: {overall_min}")
print(f"  Max: {overall_max}")
if extreme_files:
    print(f"Found {len(extreme_files)} files with coords > 1m:")
    for ef in extreme_files[:10]:
        print(f"  {ef[0]}: min={ef[1]}, max={ef[2]}")
else:
    print("No meshes have coordinates > 1.0 meter. Max span is within 1 meter.")
