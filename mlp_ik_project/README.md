# MLP Initializer for Modular Robot IK Solver

This project trains a **Multi-Layer Perceptron (MLP)** to predict a good initial joint configuration for a **numerical IK solver** on **modular robots** (variable DOF, joint-type sequences). The final approach is:

1. MLP predicts an initial guess \(q_{init}\) (in degrees).
2. A Damped Least Squares IK refines it to reach the target position.

The solver behavior is aligned with the IK solver used in the ROS2 node (`IKsolverNode`): base-yaw heuristic, init policy, damping, and LR adaptation.

## Setup

Create and activate a venv, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install numpy torch matplotlib
```

## 1) Generate dataset

This generates `mlp/ik_dataset.npz` and `mlp/ik_dataset_meta.json'.

```bash
python mlp/dataset_gen.py
```



## 2) Train

This writes `mlp/mlp_initializer.pt` and `mlp/mlp_initializer.pt.meta.json`.

```bash
python mlp/train_mlp.py --epochs 50 --device cpu
```

If you have CUDA:

```bash
python mlp/train_mlp.py --epochs 50 --device cuda
```

## 3) Evaluate (baseline vs MLP init)

```bash
python mlp/eval_compare.py --trials 200 --device cpu --save_path eval_results.json
python plot_eval_results.py
```

Outputs:
- `eval_results.json`: full trial records
- `eval_results_plots.png`: plots comparing baseline vs MLP-init

## Notes
- Angles are handled in **degrees** in the solver APIs; URDF joint limits are in **radians** (converted internally).
- This repo intentionally does **not** commit datasets (`*.npz`) or trained weights (`*.pt`) to keep the project lightweight.


