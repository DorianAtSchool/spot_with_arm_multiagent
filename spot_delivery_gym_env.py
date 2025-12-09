#!/usr/bin/env python
"""
Gymnasium-style Spot-with-arm delivery environment for OmniVLA high-level control.

This environment mirrors the layout and waypoint logic of
``spot/examples/src/tasks/test_delivery_spot.py``, but exposes a standard
Gymnasium API so you can train an OmniVLA controller as a high-level policy.

- Action: high-level base command [vx, vy, yaw_rate] for the Spot body frame.
- Low-level leg control: handled by a fixed TorchScript Spot locomotion policy
  (same observation/action layout as ``SpotWithPolicyController``).
- Observation: a dict with
    - "state": 48-dim vector (base state, commands, joint states, prev action)
    - "image": RGB frame from an overhead camera (uint8, HxWx3)

You can plug OmniVLA in as:

    base_cmd = omnivla_policy(image, instruction_text, state)

and feed that into this environment's `step` as the action.
"""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(description="Spot-with-arm delivery Gym env for OmniVLA.")
parser.add_argument(
    "--headless",
    dest="headless",
    action="store_true",
    help="Run without GUI rendering (default).",
)
parser.add_argument(
    "--gui",
    dest="headless",
    action="store_false",
    help="Run with GUI visualization.",
)
parser.set_defaults(headless=True)
_cli_args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": _cli_args.headless})

import carb
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.rotations import quat_to_euler_angles, quat_to_rot_matrix
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SPOT_USD_PATH = REPO_ROOT / "spot" / "spot_with_arm.usd"
LOW_LEVEL_POLICY_PATH = REPO_ROOT / "spot" / "spot_policy_base.pt"


if not SPOT_USD_PATH.is_file():
    carb.log_error(f"[spot_delivery_gym_env] Could not find Spot-with-arm USD at '{SPOT_USD_PATH}'")
    simulation_app.close()
    sys.exit(1)

if not LOW_LEVEL_POLICY_PATH.is_file():
    carb.log_error(f"[spot_delivery_gym_env] Could not find low-level Spot policy at '{LOW_LEVEL_POLICY_PATH}'")
    simulation_app.close()
    sys.exit(1)


@dataclass
class EnvConfig:
    """Configuration for the Gym environment."""

    # Camera resolution
    cam_width: int = 128
    cam_height: int = 128

    # Physics / control
    physics_dt: float = 0.005
    # Use a render_dt that matches common camera frequencies (e.g., 30 Hz).
    render_dt: float = 1.0 / 30.0
    substeps: int = 4  # low-level controller calls per env step

    # Episode length
    max_steps: int = 500

    # Base command bounds (vx, vy, yaw_rate)
    vx_range: Tuple[float, float] = (-1.5, 1.5)
    vy_range: Tuple[float, float] = (-1.0, 1.0)
    yaw_range: Tuple[float, float] = (-2.0, 2.0)

    # Navigation
    base_speed: float = 0.6
    visit_radius: float = 0.25
    stuck_threshold: float = 0.05


class SpotLowLevelController:
    """Fixed low-level controller using a TorchScript policy on Spot-with-arm legs."""

    def __init__(self, robot: SingleArticulation, policy_path: Path) -> None:
        self._robot = robot
        self._policy = torch.jit.load(str(policy_path)).eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._obs_dim = 48
        self._default_pos: np.ndarray | None = None
        self._previous_action = np.zeros(12, dtype=np.float32)
        self._decimation = 4
        self._counter = 0

    def initialize(self) -> None:
        self._robot.initialize()
        joint_pos = self._robot.get_joint_positions()
        if joint_pos is None or joint_pos.size == 0:
            raise RuntimeError("[SpotLowLevelController] Failed to query joint positions.")
        self._default_pos = joint_pos.copy()
        self._previous_action = np.zeros_like(self._previous_action)
        self._counter = 0

    def _compute_observation(self, command: np.ndarray) -> np.ndarray:
        assert self._default_pos is not None

        lin_vel_I = self._robot.get_linear_velocity()
        ang_vel_I = self._robot.get_angular_velocity()
        _, q_IB = self._robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = R_BI @ lin_vel_I
        ang_vel_b = R_BI @ ang_vel_I
        gravity_b = R_BI @ np.array([0.0, 0.0, -1.0])

        obs = np.zeros(self._obs_dim, dtype=np.float32)
        obs[0:3] = lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b
        obs[9:12] = command

        joint_pos = self._robot.get_joint_positions()
        joint_vel = self._robot.get_joint_velocities()

        obs[12:24] = joint_pos[:12] - self._default_pos[:12]
        obs[24:36] = joint_vel[:12]
        obs[36:48] = self._previous_action
        return obs

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).view(1, -1).float().to(self._device)
            action = self._policy(obs_t).detach().view(-1).cpu().numpy().astype(np.float32)
        return action

    def step(self, command: np.ndarray) -> None:
        """Advance the low-level controller one physics step for the given base command."""
        if self._default_pos is None:
            self.initialize()

        if self._counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self._previous_action = self._compute_action(obs)

        target_pos = self._default_pos.copy()
        target_pos[:12] = self._default_pos[:12] + 0.2 * self._previous_action

        act_struct = ArticulationAction(joint_positions=target_pos)
        self._robot.apply_action(act_struct)

        self._counter += 1


class SpotDeliveryGymEnv(gym.Env):
    """Gymnasium Env for Spot-with-arm delivery with OmniVLA high-level control."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, cfg: EnvConfig | None = None, render_mode: str | None = None) -> None:
        super().__init__()
        self._cfg = cfg or EnvConfig()
        self.render_mode = render_mode

        # Build world and robot
        self._world = World(
            physics_dt=self._cfg.physics_dt,
            rendering_dt=self._cfg.render_dt,
            stage_units_in_meters=1.0,
        )
        self._world.scene.add_default_ground_plane()

        spot_prim_path = "/World/Spot"
        add_reference_to_stage(usd_path=str(SPOT_USD_PATH), prim_path=spot_prim_path)

        self._robot = SingleArticulation(
            prim_path=spot_prim_path,
            name="SpotWithArmDeliveryGym",
            position=np.array([0.0, 0.0, 0.8]),
        )

        # Camera setup (overhead). Camera frequency must be divisible by the
        # rendering frequency, so we match it to the render rate (e.g., 30 Hz).
        self._camera = Camera(
            prim_path="/World/camera",
            frequency=int(1.0 / self._cfg.render_dt),
            resolution=(self._cfg.cam_width, self._cfg.cam_height),
        )
        self._camera.initialize()
        set_camera_view(
            eye=[4.9, 0.0, 5.0],
            target=[0.0, 0.0, 0.5],
            camera_prim_path="/World/camera",
        )
        self._camera.set_focal_length(1.8)
        self._camera.add_rgb_to_frame()

        # Front camera (first-person-ish) mounted at the front of the robot.
        # Parent under the base link so it moves with the robot.
        self._front_camera = Camera(
            prim_path="/World/Spot/base/front_camera",
            frequency=int(1.0 / self._cfg.render_dt),
            resolution=(self._cfg.cam_width, self._cfg.cam_height),
        )
        self._front_camera.initialize()
        self._front_camera.set_focal_length(1.8)
        self._front_camera.add_rgb_to_frame()

        # Low-level leg controller
        self._low_level = SpotLowLevelController(self._robot, LOW_LEVEL_POLICY_PATH)

        # Navigation state
        self._start_xy = np.array([0.0, 0.0], dtype=float)
        self._waypoints: list[np.ndarray] = []
        self._visited: list[bool] = []
        self._current_wp_index: int = 0
        self._finished_route: bool = False
        self._stuck_threshold: float = self._cfg.stuck_threshold
        self._last_pos: np.ndarray = np.zeros(3)
        self._recovery_timer: int = 0
        self._base_speed: float = self._cfg.base_speed

        self._step_count: int = 0
        self._command = np.zeros(3, dtype=np.float32)

        self._add_cones_and_waypoints()
        self._world.reset()
        self._low_level.initialize()

        # Define Gym spaces
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32),
                "image": spaces.Box(
                    low=0, high=255, shape=(self._cfg.cam_height, self._cfg.cam_width, 3), dtype=np.uint8
                ),
                "front_image": spaces.Box(
                    low=0, high=255, shape=(self._cfg.cam_height, self._cfg.cam_width, 3), dtype=np.uint8
                ),
            }
        )
        self.action_space = spaces.Box(
            low=np.array([self._cfg.vx_range[0], self._cfg.vy_range[0], self._cfg.yaw_range[0]], dtype=np.float32),
            high=np.array([self._cfg.vx_range[1], self._cfg.vy_range[1], self._cfg.yaw_range[1]], dtype=np.float32),
        )

    # ------------------------------------------------------------------ scene & navigation

    def _add_cones_and_waypoints(self) -> None:
        from pxr import Gf, UsdGeom, UsdPhysics

        stage = get_current_stage()
        cone_positions = [
            np.array([1.8, -1.2, 0.0]),   # A
            np.array([0.4, 3.0, 0.0]),    # B
            np.array([-3.5, -0.16, 0.0]), # C
        ]

        margin = 0.8
        waypoints: list[np.ndarray] = []
        prev_xy = self._start_xy.copy()

        for cone in cone_positions:
            cone_xy = cone[:2]
            direction = cone_xy - prev_xy
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                direction = np.array([1.0, 0.0])
                norm = 1.0
            direction /= norm
            wp_xy = cone_xy - margin * direction
            waypoints.append(np.array([wp_xy[0], wp_xy[1], cone[2]]))
            prev_xy = cone_xy

        self._waypoints = waypoints
        self._visited = [False for _ in self._waypoints]

        radii = [0.15, 0.15, 0.15]
        heights = [0.5, 0.5, 0.5]
        colors = [Gf.Vec3f(1.0, 0.5, 0.0), Gf.Vec3f(0.0, 0.6, 1.0), Gf.Vec3f(0.2, 0.8, 0.2)]

        for idx, (pos, radius, height, color) in enumerate(zip(cone_positions, radii, heights, colors)):
            name = chr(ord("A") + idx)
            prim_path = f"/World/Cone_{name}"
            cone = UsdGeom.Cone.Define(stage, prim_path)
            cone.CreateRadiusAttr(radius)
            cone.CreateHeightAttr(height)
            cone_prim = cone.GetPrim()
            xform = UsdGeom.Xformable(cone_prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(height / 2.0)))
            cone.CreateDisplayColorAttr().Set([color])
            try:
                UsdPhysics.CollisionAPI.Apply(cone_prim)
            except Exception:
                pass

    # ------------------------------------------------------------------ Gym API

    def _get_state_obs(self) -> np.ndarray:
        """Return the 48-dim low-level observation given the current command."""
        # Reuse low-level obs computation but avoid stepping.
        assert self._low_level._default_pos is not None
        lin_vel_I = self._robot.get_linear_velocity()
        ang_vel_I = self._robot.get_angular_velocity()
        _, q_IB = self._robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = R_BI @ lin_vel_I
        ang_vel_b = R_BI @ ang_vel_I
        gravity_b = R_BI @ np.array([0.0, 0.0, -1.0])

        obs = np.zeros(48, dtype=np.float32)
        obs[0:3] = lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b
        obs[9:12] = self._command

        joint_pos = self._robot.get_joint_positions()
        joint_vel = self._robot.get_joint_velocities()

        obs[12:24] = joint_pos[:12] - self._low_level._default_pos[:12]
        obs[24:36] = joint_vel[:12]
        obs[36:48] = self._low_level._previous_action
        return obs

    def _get_obs(self) -> Dict[str, np.ndarray]:
        state = self._get_state_obs()

        # Overhead camera image
        self._camera.get_current_frame()
        rgb_overhead = self._camera.get_rgb()
        if rgb_overhead is None:
            rgb_overhead = np.zeros((self._cfg.cam_height, self._cfg.cam_width, 3), dtype=np.uint8)

        # Front (first-person) camera image
        self._front_camera.get_current_frame()
        rgb_front = self._front_camera.get_rgb()
        if rgb_front is None:
            rgb_front = np.zeros((self._cfg.cam_height, self._cfg.cam_width, 3), dtype=np.uint8)

        return {
            "state": state.astype(np.float32),
            "image": rgb_overhead.astype(np.uint8),
            "front_image": rgb_front.astype(np.uint8),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._world.reset()
        self._low_level.initialize()
        self._step_count = 0
        self._command[:] = 0.0

        # Reset navigation state
        self._current_wp_index = 0
        self._visited = [False for _ in self._waypoints]
        self._finished_route = False
        self._recovery_timer = 0
        self._last_pos[:] = 0.0

        # Position the front camera slightly ahead of and above the base,
        # tilted down toward the ground in front of the robot.
        base_pos, _ = self._robot.get_world_pose()
        eye = [
            float(base_pos[0] + 0.6),
            float(base_pos[1]),
            float(base_pos[2] + 0.3),
        ]
        target = [
            float(base_pos[0] + 2.0),
            float(base_pos[1]),
            float(base_pos[2]),
        ]
        set_camera_view(eye=eye, target=target, camera_prim_path="/World/Spot/base/front_camera")

        # Let the world settle
        for _ in range(10):
            self._world.step(render=not _cli_args.headless)

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        # Clip action to allowed range
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(
            action,
            [self._cfg.vx_range[0], self._cfg.vy_range[0], self._cfg.yaw_range[0]],
            [self._cfg.vx_range[1], self._cfg.vy_range[1], self._cfg.yaw_range[1]],
        )
        self._command = action

        mission_done = False
        visit_bonus = 0.0

        # Navigation: progress toward current waypoint, similar to DeliverySpotTask but
        # letting the agent choose the base command.
        current_position, current_orientation = self._robot.get_world_pose()
        _, _, yaw = quat_to_euler_angles(current_orientation)

        while self._current_wp_index < len(self._waypoints) and self._visited[self._current_wp_index]:
            self._current_wp_index += 1

        if self._current_wp_index >= len(self._waypoints):
            mission_done = True
        else:
            target = self._waypoints[self._current_wp_index]
            target_pos_2d = np.array([target[0], target[1]])
            current_pos_2d = np.array([current_position[0], current_position[1]])
            diff = target_pos_2d - current_pos_2d
            dist = np.linalg.norm(diff)

            if dist < self._cfg.visit_radius:
                self._visited[self._current_wp_index] = True
                visit_bonus = 5.0

        # Low-level control and physics substeps
        for _ in range(self._cfg.substeps):
            self._low_level.step(self._command)
            self._world.step(render=not _cli_args.headless)

        obs = self._get_obs()
        state = obs["state"]
        lin_vel_b = state[0:3]

        # Distance to current waypoint after applying action
        current_position, _ = self._robot.get_world_pose()
        dist_to_goal = 0.0
        if self._current_wp_index < len(self._waypoints):
            target = self._waypoints[self._current_wp_index]
            target_pos_2d = np.array([target[0], target[1]])
            current_pos_2d = np.array([current_position[0], current_position[1]])
            dist_to_goal = np.linalg.norm(target_pos_2d - current_pos_2d)

        forward_vel = lin_vel_b[0]
        reward = -dist_to_goal + 0.1 * forward_vel + visit_bonus
        if mission_done:
            reward += 20.0
        reward = float(reward)

        pos, _ = self._robot.get_world_pose()
        done = bool(pos[2] < 0.3 or mission_done)
        self._step_count += 1
        if self._step_count >= self._cfg.max_steps:
            done = True

        info = {"mission_done": mission_done, "dist_to_goal": dist_to_goal}
        terminated = done
        truncated = self._step_count >= self._cfg.max_steps and not mission_done
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            self._camera.get_current_frame()
            rgb = self._camera.get_rgb()
            if rgb is None:
                rgb = np.zeros((self._cfg.cam_height, self._cfg.cam_width, 3), dtype=np.uint8)
            return rgb
        return None

    def close(self):
        pass


class OmniVLAHighLevelController:
    """Thin adapter to use an OmniVLA policy as a high-level controller.

    Expected usage (pseudo-code):

        import omnivla
        model = omnivla.load_your_model(...)
        ctrl = OmniVLAHighLevelController(model, instruction="Go through cones A, B, C.")

        obs, _ = env.reset()
        action = ctrl.act(obs)   # -> np.array([vx, vy, yaw_rate])

    The exact call into OmniVLA depends on your repo; this class assumes a
    callable that accepts image + text (+ optional state) and returns a
    3-dim continuous action.
    """

    def __init__(self, model, instruction: str = "Navigate through the cones in order.") -> None:
        self._model = model
        self._instruction = instruction
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        front_img = obs["front_image"]  # H x W x 3, uint8
        state = obs["state"]            # 48-dim float32

        # Convert to tensors; adjust preprocessing to match OmniVLA expectations.
        img_t = torch.from_numpy(front_img).permute(2, 0, 1).unsqueeze(0).float().to(self._device)
        state_t = torch.from_numpy(state).unsqueeze(0).float().to(self._device)

        with torch.no_grad():
            # Placeholder call: update to the actual OmniVLA API.
            # For example:
            #   action_t = self._model(image=img_t, text=self._instruction, state=state_t)
            # Here we just return zeros as a safe fallback.
            action_t = torch.zeros(1, 3, device=self._device)

        return action_t.squeeze(0).cpu().numpy().astype(np.float32)


def main():
    env = SpotDeliveryGymEnv(render_mode="rgb_array")
    obs, _ = env.reset()
    print("Initial obs shapes:", {k: v.shape for k, v in obs.items()})

    # Example loop using random actions. Replace `env.action_space.sample()`
    # with `omnivla_controller.act(obs)` once you have wired in your OmniVLA
    # policy instance.
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
