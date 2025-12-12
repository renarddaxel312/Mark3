import json
import os
import sys
import time
from typing import List, Dict

import numpy as np


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import urdf, parse_urdf
from urdf_position_ik.solver import inverse_kinematics_pos, forward_kinematics_pos, inverse_kinematics_batch


def filtered_random_config(rng, min_joints=3, max_joints=7):
    """Random joint-type config."""
    joint_types = ["rot180", "rot360"]
    n = rng.integers(min_joints, max_joints + 1)
    config = [rng.choice(joint_types) for _ in range(n)]
    config.append("rot360")  # terminal element for urdf()
    return config


def canonical_configs():
    """Seed configs per DOF."""
    configs = []
    for n in range(3, 8):
        alt = ["rot360" if i % 2 == 0 else "rot180" for i in range(n)]
        configs.append(alt + ["rot360"])
        alt2 = ["rot180" if i % 2 == 0 else "rot360" for i in range(n)]
        configs.append(alt2 + ["rot360"])
    return configs


def success_configs_from_log(path="ik_run_data.json", err_thresh=0.01):
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        cfgs = []
        for r in data.get("records", []):
            if r.get("error_m", 1.0) < err_thresh:
                cfgs.append(r.get("config"))
        return cfgs
    except Exception:
        return []


def mutate_config(config, rng, max_mutations=1):
    """Small config mutation."""
    base = config[:-1]
    for _ in range(max_mutations):
        if rng.random() < 0.5:
            idx = rng.integers(0, len(base))
            base[idx] = "rot360" if base[idx] == "rot180" else "rot180"
        else:
            if len(base) >= 2:
                idx = rng.integers(0, len(base) - 1)
                base[idx], base[idx + 1] = base[idx + 1], base[idx]
    return list(base) + ["rot360"]


def build_config_pool_all():
    """All joint-type configs for DOF 3..7 (248 total)."""
    pool = []
    joint_types = ["rot180", "rot360"]
    for n in range(3, 8):
        for mask in range(2**n):
            cfg = []
            for i in range(n):
                bit = (mask >> i) & 1
                cfg.append(joint_types[bit])
            cfg.append("rot360")  
            pool.append(cfg)
    return pool


def sample_target_via_fk(urdf_info, rng, noise_std=0.01):
    n = urdf_info["n_joints"]
    limits = urdf_info["joint_limits"]
    lows = np.rad2deg([lo for lo, _ in limits])
    highs = np.rad2deg([hi for _, hi in limits])
    q = rng.uniform(lows, highs)
    pos = forward_kinematics_pos(urdf_info, q)
    pos_noisy = pos + rng.normal(scale=noise_std, size=3)
    pos_noisy[2] = max(pos_noisy[2], 0.03)
    return pos_noisy


def encode_input(target, config, max_joints=7):
    joint_types = config[:-1]
    jt_enc = np.zeros(max_joints, dtype=np.float32)
    mask = np.zeros(max_joints, dtype=np.float32)
    for i, jt in enumerate(joint_types):
        if i >= max_joints:
            break
        jt_enc[i] = 1.0 if jt == "rot360" else 0.0
        mask[i] = 1.0
    target = np.asarray(target, dtype=np.float32)
    x = np.concatenate([target, jt_enc, mask]).astype(np.float32)
    return x


def normalize_angles_deg(q_deg, limits):
    q_deg = np.asarray(q_deg, dtype=np.float32)
    norm = np.zeros_like(q_deg)
    for i, (lo, hi) in enumerate(limits):
        lo_d = np.rad2deg(lo)
        hi_d = np.rad2deg(hi)
        mid = 0.5 * (lo_d + hi_d)
        span = (hi_d - lo_d) * 0.5
        norm[i] = (q_deg[i] - mid) / (span + 1e-6)
    return norm


def generate_dataset(
    out_path="mlp/ik_dataset.npz",
    summary_path="mlp/ik_dataset_meta.json",
    trials_per_config=2500,
    error_keep_thresh=0.01,  # keep samples with IK error <= 1 cm 
    seed=0,
    collect_solver_stats=True,
):
    rng = np.random.default_rng(seed)
    configs = build_config_pool_all()

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    meta: List[Dict] = []

    t_start = time.time()
    total = 0
    kept = 0
    filtered_out = 0
    successes = 0

    for cfg_idx, cfg in enumerate(configs):
        urdf_info = parse_urdf(urdf(cfg))
        limits = urdf_info["joint_limits"]
        print(f"Processing config {cfg_idx + 1}/{len(configs)}: {cfg} (DOF: {urdf_info['n_joints']})")
        
        # Batch generation
        targets = []
        for _ in range(trials_per_config):
            targets.append(sample_target_via_fk(urdf_info, rng, noise_std=0.01))
        
        targets = np.array(targets)
        print(f"  Prepared {len(targets)} targets, launching IK batch...")
        
        t0 = time.time()
        # Using batch solver with GPU support
        if collect_solver_stats:
            q_sols, errors, solver_stats = inverse_kinematics_batch(
                urdf_info,
                targets=targets,
                q_init=None,
                max_iter=2000,
                lr=0.25,
                lam=5e-4,
                tol=5e-3,
                use_base_yaw=True,
                verbose=True,
                log_interval=200,
                return_stats=True,
            )
        else:
            q_sols, errors = inverse_kinematics_batch(
                urdf_info,
                targets=targets,
                q_init=None,
                max_iter=2000,
                lr=0.25,
                lam=5e-4,
                tol=5e-3,
                use_base_yaw=True,
                verbose=True,
                log_interval=200,
            )
        t1 = time.time()
        batch_duration = t1 - t0
        print(f"  Batch done in {batch_duration:.2f}s | mean err={errors.mean():.6f} | max err={errors.max():.6f}")
        
        # Process results
        for i in range(trials_per_config):
            total += 1
            q_sol = q_sols[i]
            target = targets[i]
            err = float(errors[i])
            success = err < 0.01
            
            if success:
                successes += 1
            
            # Filter IK error
            if err > error_keep_thresh:
                filtered_out += 1
                continue

            kept += 1
            x = encode_input(target, cfg)
            y_norm = normalize_angles_deg(q_sol[: urdf_info["n_joints"]], limits)
            y_padded = np.zeros(7, dtype=np.float32)
            y_padded[: urdf_info["n_joints"]] = y_norm
            X_list.append(x)
            y_list.append(y_padded)
            
           
            meta.append(
                {
                    "config": cfg,
                    "dof": urdf_info["n_joints"],
                    "target": target.tolist(),
                    # re-compute EE for meta consistency or trust solver? 
                    # Trust solver's final pos if needed, but we have error.
                    # Let's just store the target and error.
                    "ee": forward_kinematics_pos(urdf_info, q_sol).tolist(),
                    "error_m": err,
                    "time_s": batch_duration / trials_per_config,
                    "success": success,
                    "solver_first_converged_iter": int(solver_stats["first_converged_iter"][i]) if collect_solver_stats else None,
                    "solver_final_lr": float(solver_stats["final_lr"][i]) if collect_solver_stats else None,
                }
            )

    if not X_list:
        raise RuntimeError("No samples kept after filtering; consider relaxing error_keep_thresh.")

    X = np.stack(X_list)
    y = np.stack(y_list)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, X=X, y=y, meta=np.array(meta, dtype=object))

    summary = {
        "total": total,
        "kept": kept,
        "filtered_out": filtered_out,
        "kept_rate": kept / max(total, 1),
        "successes": successes,
        "success_rate": successes / max(total, 1),
        "configs": len(configs),
        "trials_per_config": trials_per_config,
        "duration_s": time.time() - t_start,
        "out_path": os.path.abspath(out_path),
    }
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved dataset to {out_path}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    generate_dataset()
