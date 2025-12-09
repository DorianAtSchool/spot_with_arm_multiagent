# SPDX-FileCopyrightText: Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})  # start the simulation app, with GUI open

import os
import sys

import carb
import numpy as np
import torch
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.viewports import set_camera_view

# Path to your Spot-with-arm USD in this repo.
SPOT_USD_PATH = r"spot/spot_with_arm.usd"
# Path to your TorchScript navigation policy (exported from IsaacLab).
SPOT_POLICY_PATH = r"spot/spot_policy.pt"

if not os.path.isfile(SPOT_USD_PATH):
    carb.log_error(f"Could not find Spot USD at '{SPOT_USD_PATH}'")
    simulation_app.close()
    sys.exit(1)


# ---------------------------------------------------------------------------
# World and scene setup
# ---------------------------------------------------------------------------

my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()

set_camera_view(
    eye=[5.0, 0.0, 2.0],
    target=[0.0, 0.0, 0.5],
    camera_prim_path="/OmniverseKit_Persp",
)

# Add Spot-with-arm to the stage
spot_prim_path = "/World/Spot"
add_reference_to_stage(usd_path=SPOT_USD_PATH, prim_path=spot_prim_path)

# Wrap Spot as an articulation
spot = Articulation(prim_paths_expr=spot_prim_path, name="spot_with_arm")

# Optionally set initial pose a bit above the ground
spot.set_world_poses(
    positions=np.array([[0.0, 0.0, 0.0]]) / get_stage_units()
)


# ---------------------------------------------------------------------------
# Navigation policy wrapper (TorchScript from IsaacLab / RSL-RL)
# ---------------------------------------------------------------------------

class SpotNavigationPolicy:
    """Thin wrapper around a TorchScript Spot navigation policy.

    The policy is expected to be a TorchScript module that takes a
    tensor of shape (1, N_obs) and returns joint efforts of shape
    (1, num_dofs). Here we feed a simple observation consistent with
    the exported Spot flat-terrain policy (obs_dim=48) and apply
    the resulting torques to Spot's joints each physics step.
    """

    def __init__(self, robot: Articulation, policy_path: str) -> None:
        self._robot = robot
        self._policy: torch.jit.ScriptModule | None = None
        self._device = torch.device("cpu")
        self._num_dofs: int | None = None
        self._obs_dim: int | None = None
        self._logged_action_mismatch: bool = False

        if os.path.isfile(policy_path):
            try:
                self._policy = torch.jit.load(policy_path).eval()
                carb.log_info(f"[getting_started_spot] Loaded navigation policy from '{policy_path}'.")
            except Exception as exc:
                carb.log_error(
                    f"[getting_started_spot] Failed to load navigation policy from '{policy_path}': {exc}"
                )
                self._policy = None
        else:
            carb.log_warn(
                f"[getting_started_spot] Navigation policy file '{policy_path}' not found; "
                "using zero joint efforts."
            )

    def _infer_obs_dim(self) -> None:
        """Set observation dimension for the navigation policy.

        For the provided Spot policy exported from IsaacLab / RSL-RL, we have
        empirically verified that it expects an observation vector of size 48
        (calling the TorchScript module with a (1, 48) tensor succeeds).
        """
        if self._obs_dim is not None:
            return

        self._obs_dim = 48
        carb.log_info("[getting_started_spot] Using fixed obs_dim=48 for Spot navigation policy.")

    def _ensure_robot_metadata(self) -> None:
        if self._num_dofs is not None:
            return

        joint_pos = self._robot.get_joint_positions()
        if joint_pos is None or joint_pos.size == 0:
            return

        # joint_pos has shape (num_envs, num_dofs); here num_envs == 1.
        self._num_dofs = int(joint_pos.shape[1])

    def compute_action(self) -> np.ndarray | None:
        """Compute joint efforts for Spot from the navigation policy.

        Falls back to zeros if the policy is missing or incompatible.
        """
        self._ensure_robot_metadata()
        if self._num_dofs is None:
            return None

        # Infer observation dimension on first use.
        self._infer_obs_dim()
        obs_dim = self._obs_dim if self._obs_dim is not None else 1

        # Default: zero efforts
        zero_action = np.zeros((1, self._num_dofs), dtype=np.float32)

        if self._policy is None:
            return zero_action

        # For this getting-started example, we feed a dummy observation.
        # If you want meaningful locomotion, build a full observation vector
        # matching the training env (base state, commands, joint states, etc.).
        obs = torch.zeros((1, obs_dim), device=self._device, dtype=torch.float32)
        # generate random observation for testing
        obs = torch.randn((1, obs_dim), device=self._device, dtype=torch.float32)
        try:
            with torch.no_grad():
                action = self._policy(obs)
                print("Action: ", action)
        except Exception as exc:
            carb.log_error(
                f"[getting_started_spot] Navigation policy forward failed: {exc}. "
                "Disabling policy and reverting to zero efforts."
            )
            self._policy = None
            return zero_action

        if action.ndim != 2:
            carb.log_warn(
                "[getting_started_spot] Navigation policy action tensor must be 2D; "
                f"got shape {tuple(action.shape)}. Using zero joint efforts instead."
            )
            return zero_action

        # If the policy controls fewer DOFs (e.g., 12 leg joints) than the full
        # Spot articulation (which may include arm joints), pad the remaining
        # joints with zeros so we can still use the learned policy.
        if action.shape[1] < self._num_dofs:
            if not self._logged_action_mismatch:
                carb.log_warn(
                    "[getting_started_spot] Navigation policy action shape "
                    f"{tuple(action.shape)} has fewer DOFs than Spot ({self._num_dofs}); "
                    "padding remaining joints with zeros."
                )
                self._logged_action_mismatch = True

            padded = np.zeros((1, self._num_dofs), dtype=np.float32)
            padded[:, : action.shape[1]] = action.detach().cpu().numpy()
            return padded

        # If the policy outputs more DOFs than Spot has, ignore it.
        if action.shape[1] > self._num_dofs:
            carb.log_warn(
                "[getting_started_spot] Navigation policy action shape "
                f"{tuple(action.shape)} exceeds Spot DOFs ({self._num_dofs}); "
                "using zero joint efforts instead."
            )
            return zero_action

        return action.detach().cpu().numpy().astype(np.float32)


def main() -> None:
    # Initialize the world
    my_world.reset()

    nav_policy = SpotNavigationPolicy(robot=spot, policy_path=SPOT_POLICY_PATH)

    # Main control loop: drive Spot using the navigation policy.
    while simulation_app.is_running():
        action = nav_policy.compute_action()
        if action is not None:
            spot.set_joint_efforts(action)

        my_world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()

