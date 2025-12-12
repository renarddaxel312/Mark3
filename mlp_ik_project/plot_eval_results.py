#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


res_path = Path("eval_results.json")
with open(res_path, "r") as f:
    data = json.load(f)

b_err = np.array([r["err"] for r in data["baseline_records"]])
b_time = np.array([r["time"] for r in data["baseline_records"]])
m_err = np.array([r["err"] for r in data["mlp_records"]])
m_time = np.array([r["time"] for r in data["mlp_records"]])

fig, axs = plt.subplots(2, 2, figsize=(12, 9))

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

plt.tight_layout()
plt.savefig("eval_results_plots.png", dpi=150)
print("Saved eval_results_plots.png")
