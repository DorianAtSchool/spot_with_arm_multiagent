"""Spot-with-arm delivery task with three cones, using hardcoded motion.

This is a manual variant of ``test_delivery_spot.py``:

    - The same three cones and waypoint logic are used.
    - The Spot-with-arm robot is loaded from ``spot/spot_with_arm.usd``.
    - Instead of a learned policy, the robot base is moved directly toward
      each waypoint in sequence with a simple kinematic update.

Run from the repository root using Isaac Sim standalone Python:

    D:\isaacsim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat ^
        spot/examples/src/tasks/test_delivery_spot_manual.py
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os
import sys
from typing import List

import numpy as np
from PIL import Image

from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.prims import define_prim
from isaacsim.core.utils.rotations import quat_to_euler_angles
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera


THIS_FILE = os.path.realpath(__file__)
TASKS_DIR = os.path.dirname(THIS_FILE)
EXAMPLES_SRC_DIR = os.path.dirname(TASKS_DIR)
EXAMPLES_DIR = os.path.dirname(EXAMPLES_SRC_DIR)
REPO_ROOT = os.path.dirname(os.path.dirname(EXAMPLES_DIR))

SPOT_USD_PATH = os.path.join(REPO_ROOT, "spot", "spot_with_arm.usd")
SAVE_DIR = os.path.join(EXAMPLES_DIR, "data", "test_delivery_spot_manual")


if not os.path.isfile(SPOT_USD_PATH):
    print(f"[test_delivery_spot_manual] Could not find Spot-with-arm USD at '{SPOT_USD_PATH}'", file=sys.stderr)
    simulation_app.close()
    sys.exit(1)


class DeliverySpotManualTask:
    """Synchronous Spot-with-arm delivery task with hardcoded motion."""

    def __init__(
        self,
        world: World,
        spot: SingleArticulation,
        save_dir: str,
        max_frames: int = 5000,
        base_speed: float = 0.6,
    ) -> None:
        self.world = world
        self.spot = spot
        self.save_dir = save_dir
        self.max_frames = max_frames
        self.base_speed = base_speed

        self.camera: Camera | None = None
        self._start_xy: np.ndarray = np.array([0.0, 0.0], dtype=float)

        self._waypoints: List[np.ndarray] = []
        self._visited: List[bool] = []
        self._current_wp_index: int = 0
        self._finished_route: bool = False

        self._frame_index: int = 0

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

        position, orientation = self.spot.get_world_pose()
        _, _, yaw = quat_to_euler_angles(orientation)

        while self._current_wp_index < len(self._waypoints) and self._visited[self._current_wp_index]:
            self._current_wp_index += 1

        if self._current_wp_index >= len(self._waypoints):
            if not self._finished_route:
                print("[delivery_spot_manual] MISSION COMPLETE: All waypoints visited. Closing Simulation.")
                self._finished_route = True
            return

        target = self._waypoints[self._current_wp_index]
        target_pos_2d = np.array([target[0], target[1]])
        current_pos_2d = np.array([position[0], position[1]])

        diff = target_pos_2d - current_pos_2d
        dist = np.linalg.norm(diff)

        visit_radius = 0.25
        if dist < visit_radius:
            print(f"[delivery_spot_manual] Visited Waypoint {self._current_wp_index}")
            self._visited[self._current_wp_index] = True
            return

        # Compute small step toward the waypoint.
        max_step = self.base_speed * dt
        if dist <= max_step:
            new_pos_2d = target_pos_2d
        else:
            direction = diff / dist
            new_pos_2d = current_pos_2d + direction * max_step

        # Keep height fixed; assume Spot's base is around the same z.
        new_position = np.array([new_pos_2d[0], new_pos_2d[1], position[2]], dtype=float)
        # SingleArticulation exposes `set_world_pose` for a single instance.
        self.spot.set_world_pose(position=new_position, orientation=orientation)

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

    prim = define_prim("/World/Spot", "Xform")
    prim.GetReferences().ClearReferences()
    prim.GetReferences().AddReference(SPOT_USD_PATH)

    spot = SingleArticulation(
        prim_path="/World/Spot",
        name="SpotWithArmManual",
        position=np.array([0.0, 0.0, 0.8]),
    )

    task = DeliverySpotManualTask(world=world, spot=spot, save_dir=SAVE_DIR, max_frames=5000, base_speed=0.6)

    while simulation_app.is_running():
        task.step(dt=world.get_physics_dt())
        world.step(render=True)
        if task._finished_route or task._frame_index >= task.max_frames:
            break

    simulation_app.close()


if __name__ == "__main__":
    main()
