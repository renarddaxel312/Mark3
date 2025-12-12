import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import urdf, parse_urdf
from urdf_position_ik.solver import inverse_kinematics_pos, forward_kinematics_pos
from mlp.model import MLPInitializer
from mlp.dataset_gen import encode_input, filtered_random_config, sample_target_via_fk


def load_model(path, input_dim, output_dim, device="cpu"):
    model = MLPInitializer(input_dim, output_dim)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def denorm_angles(norm_q, limits):
    norm_q = np.asarray(norm_q, dtype=np.float32)
    q_deg = np.zeros_like(norm_q)
    for i, (lo, hi) in enumerate(limits):
        lo_d = np.rad2deg(lo)
        hi_d = np.rad2deg(hi)
        mid = 0.5 * (lo_d + hi_d)
        span = (hi_d - lo_d) * 0.5
        q_deg[i] = norm_q[i] * span + mid
    return q_deg


def compare(model, device, trials=100, seed=0, log_every=1, max_err=0.01, max_resamples=30):
    rng = np.random.default_rng(seed)
    stats = {"baseline": [], "mlp": []}

    for t in range(1, trials + 1):
        # Resample until baseline error meets threshold (if provided)
        resamples = 0
        while True:
            cfg = filtered_random_config(rng)
            urdf_info = parse_urdf(urdf(cfg))
            target = sample_target_via_fk(urdf_info, rng, noise_std=0.01)

            # Baseline
            t0 = time.time()
            q_base = inverse_kinematics_pos(
                urdf_info,
                target_pos=target,
                q_init=None,
                max_iter=2000,
                lr=0.25,
                lam=5e-4,
                tol=5e-3,
                eps_fd=0.08,
                n_restarts=3,
                use_base_yaw=True,
            )
            t1 = time.time()
            ee_base = forward_kinematics_pos(urdf_info, q_base)
            err_base = float(np.linalg.norm(target - ee_base))

            if max_err is None or err_base <= max_err or resamples >= max_resamples:
                break
            resamples += 1

        # Baseline (computed already if loop broke on threshold)
        stats["baseline"].append({"err": err_base, "time": t1 - t0})

        # MLP init
        x = encode_input(target, cfg)
        with torch.no_grad():
            pred = model(torch.tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
        q_init = denorm_angles(pred[: urdf_info["n_joints"]], urdf_info["joint_limits"])

        t2 = time.time()
        q_mlp = inverse_kinematics_pos(
            urdf_info,
            target_pos=target,
            q_init=q_init,
            max_iter=2000,
            lr=0.25,
            lam=5e-4,
            tol=5e-3,
            eps_fd=0.08,
            n_restarts=3,
            use_base_yaw=True,
        )
        t3 = time.time()
        ee_mlp = forward_kinematics_pos(urdf_info, q_mlp)
        err_mlp = float(np.linalg.norm(target - ee_mlp))
        stats["mlp"].append({"err": err_mlp, "time": t3 - t2})

        if log_every and t % log_every == 0:
            mean_b_err = np.mean([r["err"] for r in stats["baseline"]])
            mean_m_err = np.mean([r["err"] for r in stats["mlp"]])
            mean_b_t = np.mean([r["time"] for r in stats["baseline"]])
            mean_m_t = np.mean([r["time"] for r in stats["mlp"]])
            print(
                f"[{t}/{trials}] "
                f"baseline mean_err={mean_b_err:.4f} mean_time={mean_b_t:.3f}s | "
                f"mlp mean_err={mean_m_err:.4f} mean_time={mean_m_t:.3f}s"
            )

    def summarize(records):
        errs = [r["err"] for r in records]
        times = [r["time"] for r in records]
        return {
            "mean_err": float(np.mean(errs)),
            "median_err": float(np.median(errs)),
            "max_err": float(np.max(errs)),
            "mean_time": float(np.mean(times)),
            "median_time": float(np.median(times)),
            "max_time": float(np.max(times)),
        }

    return {
        "baseline": summarize(stats["baseline"]),
        "mlp": summarize(stats["mlp"]),
        "baseline_records": stats["baseline"],
        "mlp_records": stats["mlp"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlp/mlp_initializer.pt")
    parser.add_argument("--model_meta", default="mlp/mlp_initializer.pt.meta.json")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_every", type=int, default=5, help="Progress logging interval in trials (0 to disable)")
    parser.add_argument("--max_err", type=float, default=None, help="If set, resample until baseline err <= max_err")
    parser.add_argument("--max_resamples", type=int, default=20, help="Limit on resamples when filtering by max_err")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save full results JSON")
    args = parser.parse_args()

    with open(args.model_meta, "r") as f:
        meta = json.load(f)
    model = load_model(args.model, meta["input_dim"], meta["output_dim"], device=args.device)

    results = compare(
        model,
        args.device,
        trials=args.trials,
        seed=args.seed,
        log_every=args.log_every,
        max_err=args.max_err,
        max_resamples=args.max_resamples,
    )
    print(json.dumps({k: v for k, v in results.items() if not k.endswith('_records')}, indent=2))

    if args.save_path:
        with open(args.save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {os.path.abspath(args.save_path)}")

