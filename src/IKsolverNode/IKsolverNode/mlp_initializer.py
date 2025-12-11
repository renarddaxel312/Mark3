from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int = 17
    output_dim: int = 7
    max_joints: int = 7


class _MLPState:
    """
    Lazy-loaded torch model holder.
    We keep torch imports inside the loader to avoid paying import cost unless enabled.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._loaded = False
        self._load_error: Optional[Exception] = None
        self._model = None
        self._device = "cpu"
        self._cfg = MLPConfig()

    def load(self, model_path: str, meta_path: Optional[str], device: str = "cpu") -> None:
        with self._lock:
            if self._loaded:
                return

            try:
                import torch
                import torch.nn as nn

                class MLPInitializer(nn.Module):
                    def __init__(self, input_dim: int, output_dim: int, hidden=(512, 512)):
                        super().__init__()
                        layers = []
                        dims = [input_dim] + list(hidden) + [output_dim]
                        for i in range(len(dims) - 2):
                            layers.append(nn.Linear(dims[i], dims[i + 1]))
                            layers.append(nn.ReLU())
                        layers.append(nn.Linear(dims[-2], dims[-1]))
                        self.net = nn.Sequential(*layers)

                    def forward(self, x):
                        return self.net(x)

                # Default meta values (match training defaults)
                input_dim = self._cfg.input_dim
                output_dim = self._cfg.output_dim
                hidden = (512, 512)

                if meta_path and os.path.isfile(meta_path):
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    input_dim = int(meta.get("input_dim", input_dim))
                    output_dim = int(meta.get("output_dim", output_dim))
                    h = meta.get("hidden", list(hidden))
                    if isinstance(h, list) and len(h) >= 1:
                        hidden = tuple(int(x) for x in h)

                self._cfg = MLPConfig(input_dim=input_dim, output_dim=output_dim, max_joints=7)

                self._device = device or "cpu"
                model = MLPInitializer(input_dim, output_dim, hidden=hidden)
                state = torch.load(model_path, map_location=self._device)
                model.load_state_dict(state)
                model.to(self._device)
                model.eval()
                self._model = model
                self._loaded = True
            except Exception as e:
                self._load_error = e
                self._loaded = True

    @property
    def load_error(self) -> Optional[Exception]:
        return self._load_error

    @property
    def cfg(self) -> MLPConfig:
        return self._cfg

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self._loaded:
            raise RuntimeError("MLP model not loaded")
        if self._load_error is not None:
            raise RuntimeError(f"MLP model failed to load: {self._load_error}") from self._load_error
        if self._model is None:
            raise RuntimeError("MLP model is missing")

        import torch

        xt = torch.tensor(x, dtype=torch.float32, device=self._device)
        with torch.no_grad():
            y = self._model(xt).detach().cpu().numpy()
        return y


_STATE = _MLPState()


def encode_input(target_xyz_m: Sequence[float], joint_types: Sequence[str], max_joints: int = 7) -> np.ndarray:
    """
    Encode problem as [target_xyz(3), joint_type_enc(7), mask(7)] = 17 floats.
    - joint_type_enc[i] = 1.0 for rot360 else 0.0 (rot180)
    - mask[i] = 1.0 for active joints else 0.0
    """
    jt_enc = np.zeros(max_joints, dtype=np.float32)
    mask = np.zeros(max_joints, dtype=np.float32)

    for i, jt in enumerate(joint_types):
        if i >= max_joints:
            break
        jt_enc[i] = 1.0 if jt == "rot360" else 0.0
        mask[i] = 1.0

    target = np.asarray(target_xyz_m, dtype=np.float32).reshape(3)
    x = np.concatenate([target, jt_enc, mask]).astype(np.float32)
    return x


def denorm_angles_deg(norm_q: Sequence[float], joint_limits_rad: Sequence[Tuple[float, float]]) -> np.ndarray:
    """
    Convert normalized angle predictions to degrees, using the same normalization
    as in ML_project:
      norm = (q_deg - mid) / (span + eps) , where span=(hi-lo)/2 in degrees
    """
    norm_q = np.asarray(norm_q, dtype=np.float32)
    q_deg = np.zeros(len(joint_limits_rad), dtype=np.float32)

    for i, (lo, hi) in enumerate(joint_limits_rad):
        lo_d = np.rad2deg(lo)
        hi_d = np.rad2deg(hi)
        mid = 0.5 * (lo_d + hi_d)
        span = (hi_d - lo_d) * 0.5
        q_deg[i] = float(norm_q[i]) * float(span) + float(mid)

    # Clamp to limits for safety
    for i, (lo, hi) in enumerate(joint_limits_rad):
        q_deg[i] = np.clip(q_deg[i], np.rad2deg(lo), np.rad2deg(hi))

    return q_deg


def ensure_model_loaded(model_path: str, device: str = "cpu", meta_path: Optional[str] = None) -> None:
    _STATE.load(model_path=model_path, meta_path=meta_path, device=device)


def predict_q_init_deg(
    target_xyz_m: Sequence[float],
    joint_types: Sequence[str],
    joint_limits_rad: Sequence[Tuple[float, float]],
    model_path: str,
    device: str = "cpu",
    meta_path: Optional[str] = None,
) -> np.ndarray:
    """
    Returns q_init in degrees (length = dof), suitable for passing into IK.
    """
    ensure_model_loaded(model_path=model_path, device=device, meta_path=meta_path)

    if _STATE.load_error is not None:
        raise RuntimeError(f"MLP load failed: {_STATE.load_error}") from _STATE.load_error

    dof = len(joint_limits_rad)
    x = encode_input(target_xyz_m, joint_types, max_joints=_STATE.cfg.max_joints)

    y = _STATE.predict(x)
    # y is shape (7,) for single sample
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if y.shape[0] < dof:
        raise ValueError(f"MLP output dim {y.shape[0]} < dof {dof}")

    q_init_deg = denorm_angles_deg(y[:dof], joint_limits_rad)
    return q_init_deg


