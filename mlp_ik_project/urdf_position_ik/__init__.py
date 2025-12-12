"""
Position-only IK utilities for URDF-described modular robots (3–7 DOF).

This package implements a damped-least-squares numeric solver with a
finite-difference Jacobian. It depends only on NumPy and the existing
URDF helpers in `functions.py`.
"""

from .solver import (
    forward_kinematics_pos,
    numeric_jacobian_pos,
    inverse_kinematics_pos,
)

__all__ = [
    "forward_kinematics_pos",
    "numeric_jacobian_pos",
    "inverse_kinematics_pos",
]

