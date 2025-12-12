# Project Analysis: Modular Robot Inverse Kinematics with MLP Initialization

## 1. Overview
This project implements a Machine Learning approach to solve the Inverse Kinematics (IK) problem for modular robots. Modular robots can change their physical structure (link lengths, joint types, sequence), making traditional analytical IK difficult. 

The project uses a **Damped Least Squares (DLS)** numerical solver as a baseline and ground truth generator. It then trains a **Multi-Layer Perceptron (MLP)** to predict an initial joint configuration close to the solution. This predicted initialization is intended to accelerate the numerical solver or help it avoid local minima compared to random initialization.

## 2. Directory Structure

The project is organized into three main components: core utilities, the IK solver, and the Machine Learning pipeline.

### Core Utilities (`ML_project/`)
- **`functions.py`**: The backbone of the robot definition.
    - **`urdf(config)`**: Dynamically generates a URDF (Unified Robot Description Format) string based on a list of joint types (e.g., `["rot180", "rot360"]`). It assembles standard meshes (`Base.stl`, `ElbowBottom.stl`, etc.) and transforms.
    - **`parse_urdf(...)`**: Parses URDFs to extract kinematic chains, joint limits, and transforms for use in Python.
    - **`forward_kinematics_urdf(...)`**: Computes the pose of the robot given joint angles using the parsed chain.

- **`run_position_ik_tests.py`**: A script to evaluate the standalone numerical IK solver. It generates random modular robots, picks reachable targets, and measures the solver's success rate and speed.

### IK Solver (`ML_project/urdf_position_ik/`)
- **`solver.py`**: Implements the numerical Inverse Kinematics logic.
    - **`inverse_kinematics_pos(...)`**: Solves for joint angles given a target $(x, y, z)$ position.
        - **Algorithm**: Damped Least Squares (Levenberg-Marquardt style).
        - **Features**: Joint limit clamping, random restarts to avoid local minima, and a base yaw heuristic (aligns the first joint towards the target).
    - **`numeric_jacobian_pos(...)`**: Computes the gradients of the end-effector position with respect to joint angles using finite differences.

### Machine Learning Pipeline (`ML_project/mlp/`)
- **`dataset_gen.py`**: Generates training data.
    - Creates random valid robot configurations.
    - Samples reachable target positions using Forward Kinematics (FK) + noise.
    - Solves IK using the numerical solver to get ground truth joint angles.
    - **Data**: Inputs are (Target Position + Robot Config Encoding); Outputs are (Normalized Joint Angles).
    
- **`model.py`**: Defines the Neural Network.
    - **`MLPInitializer`**: A simple feed-forward network (MLP) with ReLU activations. It maps the problem definition (target + structure) to a solution guess.

- **`train_mlp.py`**: Handles model training.
    - Splits data by **configuration**, ensuring the model is tested on robot structures it has never seen before (generalization).
    - Optimizes Mean Squared Error (MSE) between predicted and ground-truth angles.

- **`eval_compare.py`**: Benchmarks the hybrid approach.
    - Compares **Baseline** (Numeric Solver with Random Init) vs. **MLP-Hybrid** (Numeric Solver with MLP Init).
    - Metrics: Success rate, convergence time, and final error.

## 3. Key Technical Details

### Robot Configuration
The robots are serial manipulators constructed from modular joints:
- **`rot180`**: A hinge-like joint (pitch).
- **`rot360`**: A cylindrical rotation joint (yaw/roll depending on orientation).
The system supports variable Degrees of Freedom (DOF), typically 3 to 7.

### Input Encoding
To allow the neural network to handle different robot structures, the input is encoded as:
1. **Target Position**: $(x, y, z)$ coordinates.
2. **Joint Type Encoding**: A fixed-size vector indicating which joint types are present (e.g., `1.0` for rot360, `0.0` for rot180).
3. **Mask**: Indicates active joints vs. padding (for robots with fewer than max joints).

### Normalization
Joint angles are normalized to the range $[-0.5, 0.5]$ (approx) based on their physical limits before being fed into the network. This stabilizes training.

## 4. Workflow

1.  **Generate Data**:
    ```bash
    python mlp/dataset_gen.py
    ```
    Produces `mlp/ik_dataset.npz`.

2.  **Train Model**:
    ```bash
    python mlp/train_mlp.py --epochs 50
    ```
    Saves the best model to `mlp/mlp_initializer.pt`.

3.  **Evaluate**:
    ```bash
    python mlp/eval_compare.py
    ```
    Outputs a JSON comparing the baseline numerical solver against the MLP-initialized solver.

## 5. Summary
The project demonstrates a **hybrid neuro-symbolic approach**. Instead of replacing the precise numerical solver with an approximate neural network, it uses the network to provide a high-quality starting point. This typically reduces the number of iterations required for the numerical solver to converge and improves reliability on complex, redundant modular manipulators.

