#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv


res_path = Path("eval_results.json")
with open(res_path, "r") as f:
    data = json.load(f)

def only_converged(records):
    return [r for r in records if r.get("converged")]

baseline_conv = only_converged(data["baseline_records"])
mlp_conv = only_converged(data["mlp_records"])

# Converged-only arrays (plots and tables)
b_err = np.array([r["err"] for r in baseline_conv], dtype=float)
b_time = np.array([r["time"] for r in baseline_conv], dtype=float)
m_err = np.array([r["err"] for r in mlp_conv], dtype=float)
m_time = np.array([r["time"] for r in mlp_conv], dtype=float)

# Optional: iterations per solve (requires updated eval_compare.py output)
b_iters = np.array(
    [r.get("iters") for r in baseline_conv if r.get("iters") is not None],
    dtype=float,
)
m_iters = np.array(
    [r.get("iters") for r in mlp_conv if r.get("iters") is not None],
    dtype=float,
)
has_iters = len(b_iters) > 0 and len(m_iters) > 0

fig, axs = plt.subplots(2, 3, figsize=(16, 9))

# Errors: overlaid hist with transparency
axs[0, 0].hist(b_err, bins=40, alpha=0.6, label="baseline", color="#1f77b4")
axs[0, 0].hist(m_err, bins=40, alpha=0.6, label="mlp", color="#ff7f0e")
axs[0, 0].set_title("Error distribution (m)")
axs[0, 0].set_xlabel("Position error (m)")
axs[0, 0].set_ylabel("Count")
axs[0, 0].legend()

# Times: side-by-side hist for clearer separation (no overlap)
width = 0.35
bins = 30
b_hist, edges = np.histogram(b_time, bins=bins)
m_hist, _ = np.histogram(m_time, bins=edges)
centers = 0.5 * (edges[:-1] + edges[1:])
axs[0, 1].bar(centers - width / 2, b_hist, width=width, label="baseline", color="#1f77b4")
axs[0, 1].bar(centers + width / 2, m_hist, width=width, label="mlp", color="#ff7f0e")
axs[0, 1].set_title("Time distribution (s)")
axs[0, 1].set_xlabel("Solve time (s)")
axs[0, 1].set_ylabel("Count")
axs[0, 1].legend()

# Iterations histogram (if present)
if has_iters:
    bins_i = 30
    b_hist_i, edges_i = np.histogram(b_iters, bins=bins_i)
    m_hist_i, _ = np.histogram(m_iters, bins=edges_i)
    centers_i = 0.5 * (edges_i[:-1] + edges_i[1:])
    axs[0, 2].bar(centers_i - width / 2, b_hist_i, width=width, label="baseline", color="#1f77b4")
    axs[0, 2].bar(centers_i + width / 2, m_hist_i, width=width, label="mlp", color="#ff7f0e")
    axs[0, 2].set_title("Iterations distribution (converged only)")
    axs[0, 2].set_xlabel("Iterations")
    axs[0, 2].set_ylabel("Count")
    axs[0, 2].legend()
else:
    axs[0, 2].axis("off")
    axs[0, 2].set_title("Iterations distribution (missing)")

# Error vs Time scatter
axs[1, 0].scatter(b_err, b_time, alpha=0.5, label="baseline", color="#1f77b4", s=18)
axs[1, 0].scatter(m_err, m_time, alpha=0.5, label="mlp", color="#ff7f0e", s=18)
axs[1, 0].set_xlabel("Position error (m)")
axs[1, 0].set_ylabel("Solve time (s)")
axs[1, 0].set_title("Error vs Time")
axs[1, 0].legend()

# CDF of times (clear separation)
for arr, label, c in [(b_time, "baseline", "#1f77b4"), (m_time, "mlp", "#ff7f0e")]:
    sorted_t = np.sort(arr)
    cdf = np.linspace(0, 1, len(sorted_t))
    axs[1, 1].plot(sorted_t, cdf, label=label, color=c)
axs[1, 1].set_xlabel("Solve time (s)")
axs[1, 1].set_ylabel("CDF")
axs[1, 1].set_title("Time CDF")
axs[1, 1].legend()

# CDF of iterations
if has_iters:
    for arr, label, c in [(b_iters, "baseline", "#1f77b4"), (m_iters, "mlp", "#ff7f0e")]:
        sorted_i = np.sort(arr)
        cdf = np.linspace(0, 1, len(sorted_i))
        axs[1, 2].plot(sorted_i, cdf, label=label, color=c)
    axs[1, 2].set_xlabel("Iterations")
    axs[1, 2].set_ylabel("CDF")
    axs[1, 2].set_title("Iterations CDF (converged only)")
    axs[1, 2].legend()
else:
    axs[1, 2].axis("off")
    axs[1, 2].set_title("Iterations CDF (missing)")

plt.tight_layout()
plt.savefig("eval_results_plots.png", dpi=150)
print("Saved eval_results_plots.png")


# -------------------------------
# Tables (Markdown + CSV)
# -------------------------------
def summarize_arr(arr):
    if arr is None or len(arr) == 0:
        return {"mean": None, "median": None, "p90": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


baseline_summary = {
    "converged_rate": data.get("baseline", {}).get("converged_rate"),
    "n_converged": data.get("baseline", {}).get("n_converged"),
    "n_total": data.get("baseline", {}).get("n_total"),
    "err": summarize_arr(b_err),
    "time": summarize_arr(b_time),
    "iters": summarize_arr(b_iters) if has_iters else summarize_arr([]),
}
mlp_summary = {
    "converged_rate": data.get("mlp", {}).get("converged_rate"),
    "n_converged": data.get("mlp", {}).get("n_converged"),
    "n_total": data.get("mlp", {}).get("n_total"),
    "err": summarize_arr(m_err),
    "time": summarize_arr(m_time),
    "iters": summarize_arr(m_iters) if has_iters else summarize_arr([]),
}


def fmt(x, nd=4):
    if x is None:
        return "-"
    return f"{x:.{nd}f}"


md_lines = []
md_lines.append("# Evaluation summary (converged only)")
md_lines.append("")
md_lines.append("All metrics below are computed **only on converged trials**. The convergence rate is reported separately.")
md_lines.append("")
md_lines.append("| Method | Converged rate | N converged | N total | Median iters | P90 iters | Median time (s) | P90 time (s) | Median err (m) | P90 err (m) |")
md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

for name, s in [("baseline", baseline_summary), ("mlp", mlp_summary)]:
    md_lines.append(
        "| {name} | {cr} | {nc} | {nt} | {mi} | {p90i} | {mt} | {p90t} | {me} | {p90e} |".format(
            name=name,
            cr=fmt(s["converged_rate"], nd=3),
            nc=s["n_converged"] if s["n_converged"] is not None else "-",
            nt=s["n_total"] if s["n_total"] is not None else "-",
            mi=fmt(s["iters"]["median"], nd=1),
            p90i=fmt(s["iters"]["p90"], nd=1),
            mt=fmt(s["time"]["median"], nd=3),
            p90t=fmt(s["time"]["p90"], nd=3),
            me=fmt(s["err"]["median"], nd=4),
            p90e=fmt(s["err"]["p90"], nd=4),
        )
    )

Path("eval_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
print("Saved eval_summary.md")

# CSV version (same columns)
with open("eval_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["method", "converged_rate", "n_converged", "n_total", "median_iters", "p90_iters", "median_time_s", "p90_time_s", "median_err_m", "p90_err_m"])
    for name, s in [("baseline", baseline_summary), ("mlp", mlp_summary)]:
        w.writerow([
            name,
            s["converged_rate"],
            s["n_converged"],
            s["n_total"],
            s["iters"]["median"],
            s["iters"]["p90"],
            s["time"]["median"],
            s["time"]["p90"],
            s["err"]["median"],
            s["err"]["p90"],
        ])
print("Saved eval_summary.csv")
