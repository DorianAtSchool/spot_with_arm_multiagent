#!/usr/bin/env python
"""
Run OmniVLA as a high-level controller on the Spot delivery Gym environment.

This script:
  - Loads the SpotDeliveryGymEnv from `spot_delivery_gym_env.py`.
  - Loads an OmniVLA checkpoint from Hugging Face (e.g. NHirose/omnivla-original).
  - Wraps the model + processor in a controller that takes the front camera
    image + state and returns a high-level base command [vx, vy, yaw_rate].

NOTE: The exact mapping from OmniVLA outputs to a 3D continuous action depends
on how the checkpoint was trained (tokenized actions vs. direct regression).
This script wires the model and processor and prints output shapes so you can
inspect and then adapt the `OmniVLAController.act()` method accordingly.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
OMNIVLA_ROOT = REPO_ROOT / "OmniVLA"

if OMNIVLA_ROOT.is_dir():
    sys.path.insert(0, str(OMNIVLA_ROOT))


# ---------------------------------------------------------------------------
# Isaac / env imports
# ---------------------------------------------------------------------------

from spot.spot_delivery_gym_env import SpotDeliveryGymEnv  # type: ignore

# ---------------------------------------------------------------------------
# OmniVLA imports (from the cloned OmniVLA repo)
# ---------------------------------------------------------------------------

from transformers import AutoProcessor  # type: ignore
from prismatic.extern.hf.modeling_prismatic import (  # type: ignore
    OpenVLAForActionPrediction_MMNv1,
)


class OmniVLAController:
    """High-level controller using OmniVLA on SpotDeliveryGymEnv.

    This class expects an OmniVLA checkpoint that can be loaded via
    `OpenVLAForActionPrediction_MMNv1.from_pretrained(model_id)` and a
    compatible `AutoProcessor.from_pretrained(model_id)`.

    The current implementation:
      - Preprocesses the front camera image and instruction text with the
        AutoProcessor.
      - Runs a forward pass through the OmniVLA model.
      - Prints out key shapes of the outputs the first time it's called.
      - Returns a placeholder zero action until you adapt the mapping from
        model outputs to [vx, vy, yaw_rate].
    """

    def __init__(self, model_id: str, instruction: str) -> None:
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._instruction = instruction

        print(f"[run_omnivla_delivery] Loading OmniVLA model from '{model_id}'...")
        # `trust_remote_code=True` is required for NHirose/omnivla-original since it
        # uses custom modeling/processing code on Hugging Face.
        self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self._model = OpenVLAForActionPrediction_MMNv1.from_pretrained(model_id).to(self._device).eval()
        print("[run_omnivla_delivery] OmniVLA model loaded.")

        self._printed_debug = False

    def act(self, obs: dict) -> np.ndarray:
        from PIL import Image

        front_img = obs["front_image"]  # H x W x 3 (uint8)

        # Build inputs for the OmniVLA processor.
        img_pil = Image.fromarray(front_img)
        inputs = self._processor(
            images=img_pil,
            text=self._instruction,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # One-time debug print to help you inspect the model outputs.
        if not self._printed_debug:
            print("[run_omnivla_delivery] OmniVLA forward outputs keys:", outputs.keys())
            for k, v in outputs.items():
                if hasattr(v, "shape"):
                    print(f"  {k}: shape={tuple(v.shape)}")
            self._printed_debug = True

        # TODO: Map OmniVLA outputs to [vx, vy, yaw_rate].
        # Depending on how `NHirose/omnivla-original` was trained, you may have:
        #   - A sequence of action tokens that must be decoded into continuous actions, or
        #   - A regression head whose outputs are already continuous.
        #
        # Once you know which tensor encodes the action,
        # replace the zero action below with that mapping.

        action = np.zeros(3, dtype=np.float32)
        return action


def main():
    parser = argparse.ArgumentParser(description="Run OmniVLA on Spot delivery env.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="NHirose/omnivla-original",
        help="Hugging Face model repo ID for OmniVLA.",
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Navigate through the three cones in order.",
        help="Language instruction passed to OmniVLA.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of rollout episodes to run.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the underlying env in headless mode.",
    )
    args = parser.parse_args()

    # Let the env script control SimulationApp via its own CLI; here we just
    # construct the Gym env.
    env = SpotDeliveryGymEnv(render_mode="rgb_array")
    controller = OmniVLAController(model_id=args.model_id, instruction=args.instruction)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        step = 0
        while True:
            action = controller.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1
            done = terminated or truncated
            if step % 10 == 0 or done:
                print(
                    f"[run_omnivla_delivery] ep={ep} step={step} "
                    f"reward={reward:.3f} cum_reward={ep_reward:.3f} "
                    f"done={done} dist_to_goal={info.get('dist_to_goal', None)}"
                )
            if done:
                break

    env.close()


if __name__ == "__main__":
    main()
