#!/usr/bin/env python3
import argparse
import time

import numpy as np

from IKsolverNode.dh_utils import urdf as build_urdf
from IKsolverNode.kinematics import forward_kinematics_urdf, inverse_kinematics_urdf, parse_urdf
from IKsolverNode.mlp_initializer import predict_q_init_deg


def sample_target_from_fk(urdf_info, rng: np.random.Generator):
    limits = urdf_info["joint_limits"]
    lows = np.rad2deg([lo for lo, _ in limits])
    highs = np.rad2deg([hi for _, hi in limits])
    q_rand = rng.uniform(lows, highs)
    pos = forward_kinematics_urdf(urdf_info, q_rand)
    return pos, q_rand


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model_path", default=None, help="Path to mlp_initializer.pt (optional)")
    parser.add_argument("--model_meta_path", default=None, help="Path to mlp_initializer.pt.meta.json (optional)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    joint_types_choices = ["rot180", "rot360"]

    rows = []
    for t in range(args.trials):
        dof = int(rng.integers(3, 8))
        joint_types = [rng.choice(joint_types_choices) for _ in range(dof)]
        # IKsolverNode URDF generator expects an effector entry at the end (gripper)
        urdf_str = build_urdf(joint_types + ["gripper"], name="modular_robot")
        urdf_info = parse_urdf(urdf_str)

        target_pos, _ = sample_target_from_fk(urdf_info, rng)

        # Baseline
        t0 = time.time()
        q_base = inverse_kinematics_urdf(urdf_str, target_pos=target_pos, target_rpy=None, q_init=None, max_iter=2000, lr=0.3)
        t1 = time.time()
        ee_base = forward_kinematics_urdf(urdf_info, q_base)
        err_base = float(np.linalg.norm(np.asarray(target_pos) - np.asarray(ee_base)))

        # MLP init + refine
        model_path = args.model_path
        meta_path = args.model_meta_path
        if model_path is None:
            # Default assumes running from a workspace where src/IKsolverNode/mlp exists
            model_path = "src/IKsolverNode/mlp/mlp_initializer.pt"
        if meta_path is None:
            meta_path = "src/IKsolverNode/mlp/mlp_initializer.pt.meta.json"

        t2 = time.time()
        q_init = predict_q_init_deg(
            target_xyz_m=np.asarray(target_pos).tolist(),
            joint_types=joint_types,
            joint_limits_rad=urdf_info["joint_limits"],
            model_path=model_path,
            device=args.device,
            meta_path=meta_path,
        )
        q_mlp = inverse_kinematics_urdf(urdf_str, target_pos=target_pos, target_rpy=None, q_init=q_init, max_iter=2000, lr=0.3)
        t3 = time.time()
        ee_mlp = forward_kinematics_urdf(urdf_info, q_mlp)
        err_mlp = float(np.linalg.norm(np.asarray(target_pos) - np.asarray(ee_mlp)))

        rows.append(
            {
                "trial": t,
                "dof": dof,
                "baseline_time_s": t1 - t0,
                "baseline_err_m": err_base,
                "mlp_time_s": t3 - t2,
                "mlp_err_m": err_mlp,
            }
        )

        print(
            f"[{t+1:02d}/{args.trials}] dof={dof} "
            f"baseline: {t1-t0:.3f}s err={err_base:.4f}m | "
            f"mlp+refine: {t3-t2:.3f}s err={err_mlp:.4f}m"
        )

    # Summary
    b_times = np.array([r["baseline_time_s"] for r in rows])
    m_times = np.array([r["mlp_time_s"] for r in rows])
    b_errs = np.array([r["baseline_err_m"] for r in rows])
    m_errs = np.array([r["mlp_err_m"] for r in rows])

    print("\nSummary:")
    print(f"  Baseline mean_time={b_times.mean():.3f}s median_time={np.median(b_times):.3f}s mean_err={b_errs.mean():.4f}m median_err={np.median(b_errs):.4f}m")
    print(f"  MLP+refine mean_time={m_times.mean():.3f}s median_time={np.median(m_times):.3f}s mean_err={m_errs.mean():.4f}m median_err={np.median(m_errs):.4f}m")


if __name__ == "__main__":
    main()


