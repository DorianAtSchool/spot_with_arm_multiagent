from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import sys
from typing import Optional

import carb
import numpy as np
import torch
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction


# ---------------------------------------------------------------------------
# Paths: local Spot-with-arm asset and exported Isaac Lab policy
# ---------------------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SPOT_USD_PATH = os.path.join(THIS_DIR, "spot_with_arm.usd")
SPOT_POLICY_PATH = os.path.join(THIS_DIR, "spot_policy.pt")

if not os.path.isfile(SPOT_USD_PATH):
    carb.log_error(f"[spot_with_policy] Could not find Spot-with-arm USD at '{SPOT_USD_PATH}'")
    simulation_app.close()
    sys.exit(1)

if not os.path.isfile(SPOT_POLICY_PATH):
    carb.log_error(f"[spot_with_policy] Could not find Spot navigation policy at '{SPOT_POLICY_PATH}'")
    simulation_app.close()
    sys.exit(1)


class SpotWithPolicyController:
    """Controller that runs an Isaac Lab RSL-RL policy on Spot-with-arm.

    The observation layout and policy call follow the Isaac Lab flat-velocity
    Spot task, mirroring the logic of ``SpotFlatTerrainPolicy``:

        obs[0:3]   - base linear velocity in body frame
        obs[3:6]   - base angular velocity in body frame
        obs[6:9]   - gravity vector in body frame
        obs[9:12]  - commanded base velocity [vx, vy, yaw_rate]
        obs[12:24] - joint position error (q - default_q)
        obs[24:36] - joint velocity
        obs[36:48] - previous action
    """

    def __init__(
        self,
        prim_path: str,
        usd_path: str,
        name: str = "Spot",
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        policy_path: Optional[str] = None,
    ) -> None:
        # Ensure the prim exists and references the Spot-with-arm USD
        prim = define_prim(prim_path, "Xform")
        if usd_path:
            prim.GetReferences().ClearReferences()
            prim.GetReferences().AddReference(usd_path)

        self.robot = SingleArticulation(
            prim_path=prim_path,
            name=name,
            position=position,
            orientation=orientation,
        )

        # Load TorchScript policy exported from Isaac Lab
        if policy_path is None:
            policy_path = SPOT_POLICY_PATH
        try:
            self._policy = torch.jit.load(policy_path).eval()
            carb.log_info(f"[spot_with_policy] Loaded navigation policy from '{policy_path}'.")
        except Exception as exc:
            carb.log_error(f"[spot_with_policy] Failed to load navigation policy: {exc}")
            self._policy = None

        # Policy / controller hyper-parameters (matching Isaac Lab style)
        self._obs_dim = 48
        self._action_scale = 0.2
        self._decimation = 4  # call policy every 4 physics steps

        self._default_pos: Optional[np.ndarray] = None
        self._previous_action = np.zeros(12, dtype=np.float32)
        self._policy_counter = 0

    def initialize(self) -> None:
        """Initialize articulation and cache default joint configuration."""
        self.robot.initialize()
        self._default_pos = self.robot.get_joint_positions().copy()
        self._previous_action = np.zeros_like(self._previous_action, dtype=np.float32)
        self._policy_counter = 0
        carb.log_info("[spot_with_policy] Controller initialized for Spot-with-arm.")

    # ------------------------------------------------------------------ obs

    def _compute_observation(self, command: np.ndarray) -> np.ndarray:
        """Build observation vector as in Isaac Lab's Spot velocity task."""
        if self._default_pos is None:
            self._default_pos = self.robot.get_joint_positions().copy()

        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        pos_IB, q_IB = self.robot.get_world_pose()

        # Transform velocities and gravity into the body frame.
        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = R_BI @ lin_vel_I
        ang_vel_b = R_BI @ ang_vel_I
        gravity_b = R_BI @ np.array([0.0, 0.0, -1.0])

        obs = np.zeros(self._obs_dim, dtype=np.float32)

        # Base lin vel
        obs[0:3] = lin_vel_b
        # Base ang vel
        obs[3:6] = ang_vel_b
        # Gravity
        obs[6:9] = gravity_b
        # Command (vx, vy, yaw_rate)
        obs[9:12] = command

        # Joint states (first 12 DOFs: legs). Extra arm joints are ignored by policy.
        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()

        obs[12:24] = current_joint_pos[:12] - self._default_pos[:12]
        obs[24:36] = current_joint_vel[:12]

        # Previous Action
        obs[36:48] = self._previous_action

        return obs

    # ------------------------------------------------------------------ policy step

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        if self._policy is None:
            return np.zeros_like(self._previous_action, dtype=np.float32)

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).view(1, -1).float()
            action = self._policy(obs_tensor).detach().view(-1).cpu().numpy().astype(np.float32)
        return action

    def forward(self, dt: float, command: np.ndarray) -> None:
        """Compute and apply policy action for this physics step."""
        if self._default_pos is None:
            self.initialize()

        # Only query the policy every `_decimation` steps.
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self._previous_action = self._compute_action(obs)

        # Interpret policy output as joint position offsets for the legs;
        # apply as position targets around the default pose.
        target_pos = self._default_pos.copy()
        target_pos[:12] = self._default_pos[:12] + self._action_scale * self._previous_action

        action_struct = ArticulationAction(joint_positions=target_pos)
        self.robot.apply_action(action_struct)

        self._policy_counter += 1


def main() -> None:
    # Match Isaac Lab training dt (0.005s) and a reasonable render rate (~50Hz).
    world = World(physics_dt=0.005, rendering_dt=0.02, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    # Create Spot-with-arm with policy controller
    spot_prim_path = "/World/Spot"
    controller = SpotWithPolicyController(
        prim_path=spot_prim_path,
        usd_path=SPOT_USD_PATH,
        name="SpotWithArm",
        position=np.array([0.0, 0.0, 0.8]),
    )

    world.reset()

    # Constant forward command [vx, vy, yaw_rate]
    base_command = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    while simulation_app.is_running():
        controller.forward(dt=world.get_physics_dt(), command=base_command)
        world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()

