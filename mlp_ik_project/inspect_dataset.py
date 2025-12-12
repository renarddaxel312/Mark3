#!/usr/bin/env python3
"""
Quick script to inspect the contents of the generated dataset.
"""
import numpy as np
import json
import sys
import os

# Ensure local package imports work when running from repo root
sys.path.insert(0, os.path.dirname(__file__))

npz_path = "mlp/ik_dataset.npz"
json_path = "mlp/ik_dataset_meta.json"

if not os.path.exists(npz_path):
    print(f"Error: {npz_path} not found!")
    sys.exit(1)

print("=" * 60)
print("DATASET INSPECTION")
print("=" * 60)

# Load and display summary
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        summary = json.load(f)
    print("\n📊 Summary (from meta.json):")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()

# Load npz file
data = np.load(npz_path, allow_pickle=True)

print("📦 NPZ File Contents:")
print(f"  Keys: {list(data.keys())}")
print()

# Inspect each array
for key in data.keys():
    arr = data[key]
    print(f"🔹 {key}:")
    print(f"   Shape: {arr.shape}")
    print(f"   Dtype: {arr.dtype}")
    
    if key == 'X':
        print(f"   First sample (target + config encoding + mask):")
        print(f"   {arr[0]}")
        print(f"   Target (x,y,z): {arr[0][:3]}")
        print(f"   Joint type encoding: {arr[0][3:10]}")
        print(f"   Mask: {arr[0][10:17]}")
    elif key == 'y':
        print(f"   First sample (normalized joint angles):")
        print(f"   {arr[0]}")
        print(f"   Non-zero values (active joints): {arr[0][arr[0] != 0]}")
    elif key == 'meta':
        print(f"   First sample metadata:")
        meta0 = arr[0]
        for k, v in meta0.items():
            if isinstance(v, (list, np.ndarray)) and len(v) > 5:
                print(f"     {k}: {type(v).__name__} of length {len(v)}")
            else:
                print(f"     {k}: {v}")
    
    print()

print("=" * 60)
print(f"✅ Dataset loaded successfully!")
print(f"   Total samples: {data['X'].shape[0]}")
print("=" * 60)

