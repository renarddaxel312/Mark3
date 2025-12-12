"""Quick position-only IK tests (URDF modular robots, 3–7 DOF)."""

import numpy as np
import time
import json
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from functions import urdf, parse_urdf
from urdf_position_ik.solver import inverse_kinematics_pos, forward_kinematics_pos


def random_config(rng, min_joints=3, max_joints=7):
    """Random joint-type config with basic constraints."""
    joint_types = ["rot180", "rot360"]
    for _ in range(50):
        n = rng.integers(min_joints, max_joints + 1)
        seq = []
        while len(seq) < n:
            choice = rng.choice(joint_types)
            if (
                choice == "rot180"
                and len(seq) >= 2
                and seq[-1] == "rot180"
                and seq[-2] == "rot180"
            ):
                continue
            seq.append(choice)

        if seq.count("rot360") < 1:
            continue
        if n >= 5 and seq.count("rot360") < 2:
            continue

        config = list(seq)
        config.append("rot360")  # terminal element for urdf()
        return config

    n = rng.integers(min_joints, max_joints + 1)
    config = [rng.choice(joint_types) for _ in range(n)]
    config.append("rot360")
    return config


def estimate_reach_radius(urdf_info):
    """Rough reach upper bound."""
    reach = 0.0
    for entry in urdf_info["chain"]:
        reach += np.linalg.norm(entry["transform"]["xyz"])
    return max(reach, 0.05)


def is_target_reachable_heuristic(urdf_info, target, margin=0.1):
    """Simple reachability check."""
    reach = estimate_reach_radius(urdf_info)
    return np.linalg.norm(target) <= reach * (1.0 + margin)


def sample_reachable_target_via_fk(urdf_info, rng, noise_std=0.01):
    """Sample a target from FK + noise."""
    n = urdf_info["n_joints"]
    limits = urdf_info["joint_limits"]
    lows = np.rad2deg([lo for lo, _ in limits])
    highs = np.rad2deg([hi for _, hi in limits])

    q = rng.uniform(lows, highs)
    pos = forward_kinematics_pos(urdf_info, q)
    pos_noisy = pos + rng.normal(scale=noise_std, size=3)
    pos_noisy[2] = max(pos_noisy[2], 0.03)
    return pos_noisy


def run_batch(trials=50, seed=0, top_k=5, save_path="ik_run_data.json"):
    rng = np.random.default_rng(seed)
    errors = []
    successes = 0
    records = []

    for _ in range(trials):
        config = random_config(rng)
        urdf_info = parse_urdf(urdf(config))
        n = urdf_info["n_joints"]

        target = sample_reachable_target_via_fk(urdf_info, rng, noise_std=0.01)

        t0 = time.time()
        q_sol = inverse_kinematics_pos(
            urdf_info,
            target_pos=target,
            q_init=None,
            max_iter=2000,
            lr=0.25,
            lam=5e-4,
            tol=5e-3,  # 5 mm tolerance
            eps_fd=0.08,
            n_restarts=5,
            use_base_yaw=True,
        )
        t1 = time.time()
        duration_s = t1 - t0
        ee = forward_kinematics_pos(urdf_info, q_sol)
        err = float(np.linalg.norm(target - ee))
        errors.append(err)
        if err < 0.01:  # 1 cm 
            successes += 1
        records.append(
            {
                "dof": n,
                "config": config,
                "target": target.tolist(),
                "solution": q_sol.tolist(),
                "ee": ee.tolist(),
                "error_m": err,
                "time_s": duration_s,
            }
        )

    errors = np.array(errors)
    summary = {
        "trials": trials,
        "successes(<1cm)": successes,
        "success_rate": successes / trials,
        "mean_error_m": float(errors.mean()),
        "max_error_m": float(errors.max()),
        "median_error_m": float(np.median(errors)),
        "mean_time_s": float(np.mean([r["time_s"] for r in records] or [0.0])),
        "max_time_s": float(np.max([r["time_s"] for r in records] or [0.0])),
        "median_time_s": float(np.median([r["time_s"] for r in records] or [0.0])),
    }
    worst = sorted(records, key=lambda r: r["error_m"], reverse=True)[:top_k]

    if save_path:
        payload = {"summary": summary, "records": records}
        with open(save_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved run data to {os.path.abspath(save_path)}")

    return summary, worst


if __name__ == "__main__":
    summary, worst = run_batch(trials=50, seed=0, top_k=5, save_path="ik_run_data.json")
    print("Position-only URDF IK (3–7 DOF)")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("\nTop worst cases:")
    for i, rec in enumerate(worst, 1):
        print(f"- #{i} | dof={rec['dof']} | err={rec['error_m']:.4f} m")
        print(f"  config: {rec['config']}")
        print(f"  target: {np.round(rec['target'], 4).tolist()}")
        print(f"  ee:     {np.round(rec['ee'], 4).tolist()}")

