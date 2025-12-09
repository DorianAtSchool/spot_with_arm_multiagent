"""Spot-with-arm delivery task with three cones as waypoints.

This mirrors the logic of ``isaacExamples/src/tasks/test_delivery_robot.py``,
but uses the local Spot-with-arm robot and navigation policy instead of the
Nova Carter wheeled robot.

It:
- loads ``spot/spot_with_arm.usd`` at ``/World/Spot``,
- uses a TorchScript navigation policy exported from Isaac Lab
  (``spot/spot_policy.pt``) to control locomotion,
- places three cones in the scene as waypoints,
- and drives Spot through them with simple waypoint + "unstuck" logic.

Run from the repository root using Isaac Sim standalone Python:

    D:\isaacsim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat ^
        spot/examples/src/tasks/test_delivery_spot.py
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import sys
from typing import List, Optional

import numpy as np
from PIL import Image

import carb
import torch
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.utils.rotations import quat_to_euler_angles, quat_to_rot_matrix
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera


THIS_FILE = os.path.realpath(__file__)
TASKS_DIR = os.path.dirname(THIS_FILE)
EXAMPLES_SRC_DIR = os.path.dirname(TASKS_DIR)
EXAMPLES_DIR = os.path.dirname(EXAMPLES_SRC_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(EXAMPLES_DIR))

SPOT_USD_PATH = os.path.join(REPO_ROOT, "spot", "spot_with_arm.usd")
SPOT_POLICY_PATH = os.path.join(REPO_ROOT, "spot", "spot_policy_base.pt")
SAVE_DIR = os.path.join(EXAMPLES_DIR, "data", "test_delivery_spot")


if not os.path.isfile(SPOT_USD_PATH):
    carb.log_error(f"[test_delivery_spot] Could not find Spot-with-arm USD at '{SPOT_USD_PATH}'")
    simulation_app.close()
    sys.exit(1)

if not os.path.isfile(SPOT_POLICY_PATH):
    carb.log_error(f"[test_delivery_spot] Could not find Spot navigation policy at '{SPOT_POLICY_PATH}'")
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
        name: str = "SpotWithArm",
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        policy_path: Optional[str] = None,
    ) -> None:
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

        if policy_path is None:
            policy_path = SPOT_POLICY_PATH
        try:
            self._policy = torch.jit.load(policy_path).eval()
            carb.log_info(f"[test_delivery_spot] Loaded navigation policy from '{policy_path}'.")
        except Exception as exc:
            carb.log_error(f"[test_delivery_spot] Failed to load navigation policy: {exc}")
            self._policy = None

        self._obs_dim = 48
        self._action_scale = 0.2
        self._decimation = 4

        self._default_pos: Optional[np.ndarray] = None
        self._previous_action = np.zeros(12, dtype=np.float32)
        self._policy_counter = 0

    def initialize(self) -> None:
        self.robot.initialize()
        self._default_pos = self.robot.get_joint_positions().copy()
        self._previous_action = np.zeros_like(self._previous_action, dtype=np.float32)
        self._policy_counter = 0
        carb.log_info("[test_delivery_spot] Controller initialized for Spot-with-arm.")

    def _compute_observation(self, command: np.ndarray) -> np.ndarray:
        if self._default_pos is None:
            self._default_pos = self.robot.get_joint_positions().copy()

        lin_vel_I = self.robot.get_linear_velocity()
        ang_vel_I = self.robot.get_angular_velocity()
        _, q_IB = self.robot.get_world_pose()

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

        current_joint_pos = self.robot.get_joint_positions()
        current_joint_vel = self.robot.get_joint_velocities()

        obs[12:24] = current_joint_pos[:12] - self._default_pos[:12]
        obs[24:36] = current_joint_vel[:12]
        obs[36:48] = self._previous_action
        return obs

    def _compute_action(self, obs: np.ndarray) -> np.ndarray:
        if self._policy is None:
            return np.zeros_like(self._previous_action, dtype=np.float32)

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).view(1, -1).float()
            action = self._policy(obs_tensor).detach().view(-1).cpu().numpy().astype(np.float32)
        return action

    def forward(self, dt: float, command: np.ndarray) -> None:
        if self._default_pos is None:
            self.initialize()

        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self._previous_action = self._compute_action(obs)

        target_pos = self._default_pos.copy()
        target_pos[:12] = self._default_pos[:12] + self._action_scale * self._previous_action

        action_struct = ArticulationAction(joint_positions=target_pos)
        self.robot.apply_action(action_struct)

        self._policy_counter += 1


class DeliverySpotTask:
    """Synchronous Spot-with-arm delivery task with simple unstuck logic."""

    def __init__(
        self,
        world: World,
        controller: SpotWithPolicyController,
        save_dir: str,
        max_frames: int = 5000,
        base_speed: float = 0.6,
    ) -> None:
        self.world = world
        self.controller = controller
        self.save_dir = save_dir
        self.max_frames = max_frames
        self.base_speed = base_speed

        self.camera: Camera | None = None
        self._spot_prim: str = "/World/Spot"
        self._start_xy: np.ndarray = np.array([0.0, 0.0], dtype=float)

        self._waypoints: List[np.ndarray] = []
        self._visited: List[bool] = []
        self._current_wp_index: int = 0
        self._finished_route: bool = False

        self._frame_index: int = 0
        self._stuck_threshold: float = 0.05
        self._last_pos: np.ndarray = np.zeros(3)
        self._recovery_timer: int = 0

        self._setup_scene()

    # ---------------------------------------------------------------- scene setup

    def _add_camera(self) -> None:
        self.camera = Camera(prim_path="/World/camera", frequency=30, resolution=(800, 800))
        self.camera.initialize()
        set_camera_view(
            eye=[4.9, 0.0, 5.0],
            target=[0.0, 0.0, 0.5],
            camera_prim_path="/World/camera",
        )
        self.camera.set_focal_length(1.8)
        self.camera.add_rgb_to_frame()

    def _add_cones(self) -> None:
        from pxr import Gf, UsdGeom, UsdPhysics

        stage = get_current_stage()
        cone_positions = [
            np.array([1.8, -1.2, 0.0]),   # A
            np.array([0.4, 3.0, 0.0]),    # B
            np.array([-3.5, -0.16, 0.0]), # C
        ]

        margin = 0.8
        waypoints: List[np.ndarray] = []
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

    def _setup_scene(self) -> None:
        self.world.scene.add_default_ground_plane()
        self._add_cones()
        self._add_camera()
        self.world.reset()

    # ---------------------------------------------------------------- task logic

    def _update_motion(self, dt: float) -> None:
        if self._finished_route:
            return

        current_position, current_orientation = self.controller.robot.get_world_pose()
        _, _, yaw = quat_to_euler_angles(current_orientation)

        while self._current_wp_index < len(self._waypoints) and self._visited[self._current_wp_index]:
            self._current_wp_index += 1

        if self._current_wp_index >= len(self._waypoints):
            command = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            self.controller.forward(dt, command)
            if not self._finished_route:
                print("[delivery_spot] MISSION COMPLETE: All waypoints visited. Closing Simulation.")
                self._finished_route = True
            return

        target = self._waypoints[self._current_wp_index]
        target_pos_2d = np.array([target[0], target[1]])
        current_pos_2d = np.array([current_position[0], current_position[1]])

        diff = target_pos_2d - current_pos_2d
        dist = np.linalg.norm(diff)

        visit_radius = 0.25
        if dist < visit_radius:
            print(f"[delivery_spot] Visited Waypoint {self._current_wp_index}")
            self._visited[self._current_wp_index] = True
            return

        if self._recovery_timer > 0:
            self._recovery_timer -= 1
            command = np.array([-0.3, 0.0, 1.0], dtype=np.float32)
            self.controller.forward(dt, command)
            return

        if self._frame_index % 60 == 0:
            move_dist = np.linalg.norm(current_position - self._last_pos)
            self._last_pos = current_position.copy()
            if move_dist < self._stuck_threshold:
                print("[delivery_spot] STUCK DETECTED! Initiating recovery...")
                self._recovery_timer = 60
                return

        target_angle = np.arctan2(diff[1], diff[0])
        angle_error = target_angle - yaw
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        if abs(angle_error) > 0.5:
            v_cmd = 0.0
            w_cmd = 1.5 * np.sign(angle_error)
        else:
            v_cmd = self.base_speed
            w_cmd = 2.0 * angle_error

        command = np.array([v_cmd, 0.0, w_cmd], dtype=np.float32)
        self.controller.forward(dt, command)

    def step(self, dt: float) -> None:
        if self._frame_index >= self.max_frames:
            return

        self._update_motion(dt)

        if self.camera is not None:
            self.camera.get_current_frame()
            if self._frame_index % 10 == 0:
                os.makedirs(SAVE_DIR, exist_ok=True)
                rgb = self.camera.get_rgb()
                if rgb is not None:
                    img_path = os.path.join(SAVE_DIR, f"img_{self._frame_index:05d}.png")
                    Image.fromarray(rgb).save(img_path)

        self._frame_index += 1


def main() -> None:
    world = World(physics_dt=0.005, rendering_dt=0.02, stage_units_in_meters=1.0)

    set_camera_view(
        eye=[6.0, 0.0, 3.0],
        target=[0.0, 0.0, 0.5],
        camera_prim_path="/OmniverseKit_Persp",
    )

    controller = SpotWithPolicyController(
        prim_path="/World/Spot",
        usd_path=SPOT_USD_PATH,
        name="SpotWithArmDelivery",
        position=np.array([0.0, 0.0, 0.8]),
    )

    task = DeliverySpotTask(world=world, controller=controller, save_dir=SAVE_DIR, max_frames=5000, base_speed=0.6)

    while simulation_app.is_running():
        task.step(dt=world.get_physics_dt())
        world.step(render=True)
        if task._finished_route or task._frame_index >= task.max_frames:
            break

    simulation_app.close()


if __name__ == "__main__":
    main()

