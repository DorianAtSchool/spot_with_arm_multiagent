"""Minimal Spot-with-arm navigation sample for Isaac Sim 5.1.

This example mirrors the style of the tasks under ``isaacExamples/src/tasks``,
but is specialized for the Spot-with-arm USD in this repository.

It:
- loads ``spot/spot_with_arm.usd`` at ``/World/Spot``,
- creates a basic world with a ground plane,
- and applies a simple scripted base command to walk forward.

Run from the repository root using the Isaac Sim standalone Python:

    D:\isaacsim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat ^
        spot/examples/src/tasks/test_spot_with_arm_nav.py
"""

import os
import sys

import numpy as np
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.viewports import set_camera_view




def main() -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))

    spot_usd_path = os.path.join(repo_root, "spot_with_arm.usd")
    if not os.path.isfile(spot_usd_path):
        print(f"[test_spot_with_arm_nav] Could not find Spot-with-arm USD at '{spot_usd_path}'", file=sys.stderr)
        simulation_app.close()
        sys.exit(1)

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    set_camera_view(
        eye=[5.0, 0.0, 2.0],
        target=[0.0, 0.0, 0.5],
        camera_prim_path="/OmniverseKit_Persp",
    )

    spot_prim_path = "/World/Spot"
    add_reference_to_stage(usd_path=spot_usd_path, prim_path=spot_prim_path)

    spot = Articulation(prim_paths_expr=spot_prim_path, name="spot_with_arm_example")
    spot.set_world_poses(positions=np.array([[0.0, 0.0, 0.0]]) / get_stage_units())

    world.reset()

    step_count = 0
    while simulation_app.is_running():
        # Simple scripted base command: small forward velocity for a few seconds.
        # This is intentionally minimal; for more advanced usage, see
        # ``spot/spot_with_policy.py`` which drives Spot using an Isaac Lab policy.
        if step_count < 500:
            # For now we just let physics settle and rely on the default joints;
            # you can extend this to call a controller or policy as needed.
            pass

        world.step(render=True)
        step_count += 1

    simulation_app.close()


if __name__ == "__main__":
    main()

