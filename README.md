# Spot Locomotion Sandbox

This folder contains a small set of scripts and assets built around Boston Dynamics Spot in Omniverse Isaac Sim / Isaac Lab.
The long‑term goal is to use these pieces as building blocks for a **multi‑agent VLA (Vision‑Language‑Action) policy
controller**:

- Isaac Sim provides high‑fidelity simulation and robot articulation.
- Isaac Lab provides task definitions and low‑level locomotion policies.
- This `spot` project provides lightweight standalone entry points and assets so that higher‑level VLA controllers can
  issue commands and evaluate policies in a reproducible environment.

At the moment, the focus is on single‑robot locomotion with Spot and Spot‑with‑arm, both with simple scripted commands
and with policies exported from Isaac Lab. The new delivery‑style Gym environment and OmniVLA example are the first steps
toward language‑conditioned control.

---

## Contents

All paths below are relative to the `multi_agent_spot` repository root:

- `spot/spot_base_env.py`  
  Minimal example that runs the built‑in `SpotFlatTerrainPolicy` from Isaac Sim on a single Spot robot on flat terrain.

- `spot/getting_started_spot.py`  
  A “getting started” example for **Spot‑with‑arm** that:
  - loads a local USD asset (`spot/spot_with_arm.usd`),
  - wraps it as an articulation, and
  - (optionally) loads a TorchScript navigation policy exported from Isaac Lab.

- `spot/spot_with_policy.py`  
  A more structured controller that:
  - loads `spot_with_arm.usd` and `spot_policy.pt`,
  - reconstructs the flat‑velocity Spot observation used in Isaac Lab,
  - and drives Spot‑with‑arm with the RSL‑RL policy.

- `spot/spot_with_arm.usd`  
  Local USD asset for the Spot‑with‑arm robot used by the examples.

- `spot/spot_policy.pt`  
  TorchScript navigation policy exported from an Isaac Lab Spot flat‑velocity task (used by `spot_with_policy.py` and
  optionally by `getting_started_spot.py`).

- `spot/spot_policy_base.pt`  
  TorchScript low‑level locomotion policy used by the delivery Gym environment for leg control.

- `spot/spot_delivery_gym_env.py`  
  A Gymnasium‑style Spot‑with‑arm delivery environment that mirrors
  `spot/examples/src/tasks/test_delivery_spot.py` but exposes a standard Gym API:
  - action: high‑level base command `[vx, vy, yaw_rate]`,
  - observation: dict with `"state"` (48‑dim vector) and `"image"` / `"front_image"` RGB frames.

- `spot/run_omnivla_delivery.py`  
  Example script that loads `SpotDeliveryGymEnv` and runs an OmniVLA checkpoint from Hugging Face as a **high‑level
  controller**. It wires the OmniVLA model and processor, prints output shapes, and currently returns a placeholder
  zero action until you map the model outputs to `[vx, vy, yaw_rate]`.

- `spot/examples/`  
  Additional Spot‑with‑arm example tasks and assets. See `spot/examples/README.md` for details.

---

## Requirements

These examples assume:

- **Omniverse Isaac Sim Standalone 5.1.0 (Windows, x86_64)**  
  Installed at (adjust as needed):
  - `D:\isaacsim\isaac-sim-standalone-5.1.0-windows-x86_64\`

- **Isaac Sim standalone Python**  
  All scripts and package installs should be run through the Isaac Sim Python entry point, for example:
  - `D:\isaacsim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat`

- **This repository** (`multi_agent_spot`) cloned locally, with your shell / terminal working directory set to the repo
  root so that relative paths like `spot/spot_base_env.py` resolve correctly.

### Python dependencies

Most core dependencies (Isaac Sim, Omniverse APIs, NumPy, etc.) are provided by the bundled Isaac Sim Python
environment. If you need additional Python packages (for example, `hydra-core` or extra utilities), install them into
the Isaac Sim environment via:

```bash
D:\isaacsim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat -m pip install hydra-core
```

You can replace `hydra-core` with any other Python package you need. Always use the standalone Isaac Sim `python.bat`
when installing, so that packages end up in the correct environment.

For the **OmniVLA delivery example**, you will additionally need:

- a cloned OmniVLA repo at `OmniVLA/` in the repo root (see `OmniVLA/README.md` for setup), and
- its Python dependencies (at minimum `transformers`, `prismatic`, and their transitive dependencies) installed into the
  same Isaac Sim Python environment.

---

## Running the examples

All commands below assume:

- Your current working directory is the `multi_agent_spot` repo root.
- You are using the standalone Isaac Sim Python:  
  `D:\isaacsim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat`

Adjust the base path if Isaac Sim is installed elsewhere.

### 1. Flat‑terrain Spot base environment

This script runs the built‑in Isaac Sim `SpotFlatTerrainPolicy` in a simple warehouse / grid environment:

```bash
D:\isaacsim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat spot/spot_base_env.py
```

To run in a lightweight “test” mode (e.g., for CI or sanity checks), use:

```bash
D:\isaacsim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat spot/spot_base_env.py --test
```

The script will open an Isaac Sim window, spawn Spot on flat terrain, and execute a simple scripted command sequence
(forward, rotate, sideways).

### 2. Getting started with Spot‑with‑arm

To verify that the local Spot‑with‑arm USD and optional policy load correctly:

```bash
D:\isaacsim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat spot/getting_started_spot.py
```

This script:

- creates a simple world with a ground plane,
- loads `spot/spot_with_arm.usd` at `/World/Spot`,
- wraps it as an articulation, and
- optionally loads `spot/spot_policy.pt` as a navigation policy (if present).

By default it uses a simple “dummy” observation and mainly serves as a quick check that the asset and policy can be
loaded and stepped.

### 3. Running Spot‑with‑arm using the exported policy

To run the full controller that reconstructs the Isaac Lab flat‑velocity observation and drives Spot‑with‑arm using the
exported policy:

```bash
D:\isaacsim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat spot/spot_with_policy.py
```

This script:

- loads `spot_with_arm.usd` and `spot_policy.pt`,
- builds a 48‑dim observation vector compatible with the Isaac Lab policy (base velocities, gravity, commands, joint
  states, and previous action), and
- applies the policy output as joint position offsets for the leg joints while leaving arm joints free.

### 4. OmniVLA high‑level control on the delivery environment

The delivery environment `SpotDeliveryGymEnv` lives in `spot/spot_delivery_gym_env.py` and creates a three‑cone route
for Spot‑with‑arm. Low‑level leg control is handled by a fixed TorchScript policy (`spot_policy_base.pt`), while a
higher‑level controller chooses base commands `[vx, vy, yaw_rate]`.

`spot/run_omnivla_delivery.py` wires this environment to an OmniVLA checkpoint from Hugging Face (for example,
`NHirose/omnivla-original`) via the `OmniVLAController` class:

```bash
D:\isaacsim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat spot/run_omnivla_delivery.py ^
  --model-id NHirose/omnivla-original ^
  --instruction "Navigate through the three cones in order."
```

On the first call to `OmniVLAController.act()` the script:

- loads the OmniVLA processor and model with `trust_remote_code=True`,
- runs a forward pass on the front‑camera image + instruction,
- prints the keys and shapes of the model outputs for inspection, and
- returns a placeholder zero action.

To actually control the robot, update `OmniVLAController.act()` in `spot/run_omnivla_delivery.py` to map the model’s
outputs to a continuous action `[vx, vy, yaw_rate]`. Depending on how the checkpoint was trained, this might involve:

- decoding action tokens to continuous actions, or
- taking a regression head that is already continuous.

---

## Relation to multi‑agent VLA controllers

The `spot` folder currently focuses on **single‑agent** locomotion for Spot and Spot‑with‑arm, but it is designed to fit
into a broader research workflow:

- Isaac Lab tasks (defined elsewhere in this repo) train locomotion policies for Spot.
- Those policies are exported to TorchScript (`spot_policy.pt`, `spot_policy_base.pt`) and exercised here in standalone
  Isaac Sim.
- The delivery Gym environment + OmniVLA example show how a language‑conditioned policy can issue high‑level commands on
  top of a fixed low‑level locomotion controller.

Future work will extend these scripts to:

- support **multiple Spot instances** in a shared environment,
- expose higher‑level command interfaces suitable for VLA models (for example, language‑conditioned navigation goals),
- and integrate with multi‑agent coordination layers.

As this evolves, this README will be updated with additional entry points and configuration instructions.

