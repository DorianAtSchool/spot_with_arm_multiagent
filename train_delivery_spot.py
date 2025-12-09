#!/usr/bin/env python
"""
Standalone PPO training script for Spot-with-arm in Isaac Sim.

This script does NOT depend on Isaac Lab tasks (no Hydra, no gym.make).
It builds a small flat-terrain velocity-tracking environment directly
using the local USD asset ``spot/spot_with_arm.usd`` and trains a policy
whose observation/action layout matches ``SpotWithPolicyController`` in
``spot_with_policy.py`` and ``test_delivery_spot.py``.

PPO hyperparameters (network sizes, learning rate, etc.) are chosen to
match the original Spot RSL-RL training config:

  - actor/critic hidden dims: [512, 256, 128] with ELU
  - learning rate: 1e-3
  - gamma: 0.99, lambda (GAE): 0.95
  - clip_param: 0.2, value_loss_coef: 0.5, entropy_coef: 0.0025
  - num_learning_epochs: 5, num_mini_batches: 4

Usage (from repo root, with Isaac Sim standalone Python):

    D:\isaacsim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat ^
        spot/train_delivery_spot.py

The script writes the latest trained policy as TorchScript to:

    spot/spot_policy.pt

If a previous ``spot_policy.pt`` exists, it is rotated to a timestamped
backup instead of being overwritten.
"""

import argparse
from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Standalone PPO training for Spot-with-arm.")
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
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to a training checkpoint (.pt) to resume from.",
)
parser.set_defaults(headless=True)
_cli_args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": _cli_args.headless})

import os
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.rotations import quat_to_euler_angles, quat_to_rot_matrix
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage, get_stage_units
from isaacsim.core.utils.types import ArticulationAction


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
SPOT_USD_PATH = REPO_ROOT / "spot" / "spot_with_arm.usd"
TARGET_POLICY_PATH = REPO_ROOT / "spot" / "spot_policy.pt"

if not SPOT_USD_PATH.is_file():
    print(f"[train_delivery_spot] Could not find Spot-with-arm USD at '{SPOT_USD_PATH}'", file=sys.stderr)
    simulation_app.close()
    sys.exit(1)


@dataclass
class PPOConfig:
    obs_dim: int = 48
    act_dim: int = 12

    # Runner-style settings
    num_steps_per_env: int = 24
    max_iterations: int = 50000  # reduce from 20000 for practicality

    # Network sizes (match SpotFlatPPORunnerCfg)
    hidden_sizes: Tuple[int, int, int] = (512, 256, 128)

    # PPO hyperparameters (match SpotFlatPPORunnerCfg)
    learning_rate: float = 1.0e-3
    gamma: float = 0.99
    lam: float = 0.95
    clip_param: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.0025
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    max_grad_norm: float = 1.0

    # Exploration / init
    init_noise_std: float = 1.0


class ActorCritic(nn.Module):
    def __init__(self, cfg: PPOConfig) -> None:
        super().__init__()
        obs_dim = cfg.obs_dim
        act_dim = cfg.act_dim
        hidden_sizes = cfg.hidden_sizes

        def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            layers = []
            last = in_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(last, h))
                layers.append(nn.ELU())
                last = h
            layers.append(nn.Linear(last, out_dim))
            return nn.Sequential(*layers)

        self.actor = mlp(obs_dim, act_dim)
        self.critic = mlp(obs_dim, 1)

        self.log_std = nn.Parameter(torch.ones(act_dim) * np.log(cfg.init_noise_std), requires_grad=True)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Squash actor output to [-1, 1] for stability and clamp std.
        mean = torch.tanh(self.actor(obs))
        std = torch.exp(self.log_std).clamp(1e-3, 1.0)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = torch.tanh(self.actor(obs))
        std = torch.exp(self.log_std).clamp(1e-3, 1.0)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, entropy, value


class SpotVelocityEnv:
    """Delivery-style environment for Spot-with-arm with three cone waypoints.

    - Observation layout matches ``SpotWithPolicyController`` in
      ``examples/src/tasks/test_delivery_spot.py``.
    - World layout (cones, waypoints) and base-command logic mirror
      ``DeliverySpotTask`` from the same file so that the trained policy
      is exercised in an identical scenario at test time.
    """

    def __init__(self, world: World, usd_path: Path, cfg: PPOConfig) -> None:
        self._world = world
        self._cfg = cfg
        self._usd_path = str(usd_path)

        # Spawn Spot-with-arm
        spot_prim_path = "/World/Spot"
        add_reference_to_stage(usd_path=self._usd_path, prim_path=spot_prim_path)

        self._robot = SingleArticulation(
            prim_path=spot_prim_path,
            name="SpotWithArmTrain",
            position=np.array([0.0, 0.0, 0.8]),
        )

        self._default_pos: np.ndarray | None = None
        self._prev_action = np.zeros(cfg.act_dim, dtype=np.float32)
        # Base command [vx, vy, yaw_rate]; updated each step using
        # DeliverySpotTask-style waypoint logic.
        self._command = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._step_count = 0
        self._max_episode_steps = 500

        # Delivery navigation state (mirrors DeliverySpotTask).
        self._start_xy: np.ndarray = np.array([0.0, 0.0], dtype=float)
        self._waypoints: list[np.ndarray] = []
        self._visited: list[bool] = []
        self._current_wp_index: int = 0
        self._finished_route: bool = False
        self._stuck_threshold: float = 0.05
        self._last_pos: np.ndarray = np.zeros(3)
        self._recovery_timer: int = 0
        self._base_speed: float = 0.6

        self._world.reset()
        self._robot.initialize()
        self._init_default_state()
        self._add_cones_and_waypoints()

    def _add_cones_and_waypoints(self) -> None:
        """Create three cones and derived waypoints as in DeliverySpotTask."""
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

    def _init_default_state(self) -> None:
        joint_pos = self._robot.get_joint_positions()
        if joint_pos is None or joint_pos.size == 0:
            raise RuntimeError("Failed to query Spot joint positions.")
        self._default_pos = joint_pos.copy()

    def reset(self) -> np.ndarray:
        self._world.reset()
        self._robot.initialize()
        self._step_count = 0
        self._prev_action[:] = 0.0

        if self._default_pos is None:
            self._init_default_state()

        # Reset navigation state.
        self._current_wp_index = 0
        self._visited = [False for _ in self._waypoints]
        self._finished_route = False
        self._recovery_timer = 0
        self._last_pos[:] = 0.0

        # Small random yaw and xy offset
        pos, quat = self._robot.get_world_pose()
        if pos is None:
            pos = np.array([0.0, 0.0, 0.8])
        pos = pos.copy()
        pos[0] += np.random.uniform(-0.2, 0.2)
        pos[1] += np.random.uniform(-0.2, 0.2)
        # SingleArticulation exposes `set_world_pose` for a single instance.
        self._robot.set_world_pose(position=pos, orientation=quat)

        for _ in range(10):
            # Render only when not running in headless mode so the GUI updates.
            self._world.step(render=not _cli_args.headless)

        return self._compute_obs()

    def _compute_obs(self) -> np.ndarray:
        assert self._default_pos is not None

        lin_vel_I = self._robot.get_linear_velocity()
        ang_vel_I = self._robot.get_angular_velocity()
        _, q_IB = self._robot.get_world_pose()

        R_IB = quat_to_rot_matrix(q_IB)
        R_BI = R_IB.transpose()
        lin_vel_b = R_BI @ lin_vel_I
        ang_vel_b = R_BI @ ang_vel_I
        gravity_b = R_BI @ np.array([0.0, 0.0, -1.0])

        obs = np.zeros(self._cfg.obs_dim, dtype=np.float32)
        obs[0:3] = lin_vel_b
        obs[3:6] = ang_vel_b
        obs[6:9] = gravity_b
        obs[9:12] = self._command

        joint_pos = self._robot.get_joint_positions()
        joint_vel = self._robot.get_joint_velocities()

        obs[12:24] = joint_pos[:12] - self._default_pos[:12]
        obs[24:36] = joint_vel[:12]

        obs[36:48] = self._prev_action
        return obs

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        assert self._default_pos is not None

        # ------------------------------------------------------------------
        # Compute navigation command as in DeliverySpotTask._update_motion.
        # ------------------------------------------------------------------
        current_position, current_orientation = self._robot.get_world_pose()
        _, _, yaw = quat_to_euler_angles(current_orientation)

        # Skip waypoints already marked visited.
        while self._current_wp_index < len(self._waypoints) and self._visited[self._current_wp_index]:
            self._current_wp_index += 1

        mission_done = False
        visit_bonus = 0.0

        if self._current_wp_index >= len(self._waypoints):
            # All waypoints visited: command zero velocity.
            self._command = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            mission_done = True
        else:
            target = self._waypoints[self._current_wp_index]
            target_pos_2d = np.array([target[0], target[1]])
            current_pos_2d = np.array([current_position[0], current_position[1]])
            diff = target_pos_2d - current_pos_2d
            dist = np.linalg.norm(diff)

            visit_radius = 0.25
            if dist < visit_radius:
                self._visited[self._current_wp_index] = True
                visit_bonus = 5.0  # reward for reaching a waypoint

            if self._recovery_timer > 0:
                self._recovery_timer -= 1
                self._command = np.array([-0.3, 0.0, 1.0], dtype=np.float32)
            else:
                # Simple stuck detection every ~60 frames.
                if self._step_count % 60 == 0:
                    move_dist = np.linalg.norm(current_position - self._last_pos)
                    self._last_pos = current_position.copy()
                    if move_dist < self._stuck_threshold:
                        self._recovery_timer = 60
                        self._command = np.array([-0.3, 0.0, 1.0], dtype=np.float32)
                    else:
                        target_angle = np.arctan2(diff[1], diff[0])
                        angle_error = target_angle - yaw
                        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

                        if abs(angle_error) > 0.5:
                            v_cmd = 0.0
                            w_cmd = 1.5 * np.sign(angle_error)
                        else:
                            v_cmd = self._base_speed
                            w_cmd = 2.0 * angle_error
                        self._command = np.array([v_cmd, 0.0, w_cmd], dtype=np.float32)
                else:
                    # When not checking for stuck, keep previous command.
                    pass

        # ------------------------------------------------------------------
        # Apply policy action as joint position offsets (legs) like in the
        # SpotWithPolicyController used by test_delivery_spot.py.
        # ------------------------------------------------------------------
        self._prev_action = action.astype(np.float32).copy()

        target_pos = self._default_pos.copy()
        target_pos[:12] = self._default_pos[:12] + 0.2 * self._prev_action

        act_struct = ArticulationAction(joint_positions=target_pos)
        self._robot.apply_action(act_struct)

        # Step physics several times to simulate actuation between policy calls.
        for _ in range(4):
            # Render only when not running in headless mode so the GUI updates.
            self._world.step(render=not _cli_args.headless)

        obs = self._compute_obs()
        lin_vel_b = obs[0:3]

        # Reward: encourage forward progress toward waypoints, minimal action.
        forward_vel = lin_vel_b[0]
        dist_to_goal = 0.0
        if self._current_wp_index < len(self._waypoints):
            target = self._waypoints[self._current_wp_index]
            target_pos_2d = np.array([target[0], target[1]])
            current_pos_2d = np.array([current_position[0], current_position[1]])
            dist_to_goal = np.linalg.norm(target_pos_2d - current_pos_2d)

        # Base reward: negative distance plus small forward velocity term.
        reward = -dist_to_goal + 0.1 * forward_vel
        reward += visit_bonus
        if mission_done:
            reward += 20.0  # big bonus for completing all waypoints
        # Action penalty.
        reward -= 0.001 * float(np.sum(self._prev_action**2))

        # Terminate if base falls too low, episode too long, or mission complete.
        pos, _ = self._robot.get_world_pose()
        done = bool(pos[2] < 0.3 or mission_done)
        self._step_count += 1
        if self._step_count >= self._max_episode_steps:
            done = True

        return obs, reward, done


def compute_gae(cfg: PPOConfig, rewards, values, dones):
    T = len(rewards)
    returns = np.zeros_like(rewards)
    advantages = np.zeros_like(rewards)
    last_gae = 0.0
    last_value = 0.0
    for t in reversed(range(T)):
        next_non_terminal = 1.0 - float(dones[t])
        next_value = values[t + 1] if t < T - 1 else last_value
        delta = rewards[t] + cfg.gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + cfg.gamma * cfg.lam * next_non_terminal * last_gae
        advantages[t] = last_gae
        returns[t] = advantages[t] + values[t]
    return returns, advantages


def train():
    print("[train_delivery_spot] Starting training()", flush=True)
    cfg = PPOConfig()

    world = World(physics_dt=0.005, rendering_dt=0.02, stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    env = SpotVelocityEnv(world=world, usd_path=SPOT_USD_PATH, cfg=cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ac = ActorCritic(cfg).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=cfg.learning_rate)

    episodes = 0
    next_ckpt_ep = 500

    # Optionally resume from checkpoint.
    if _cli_args.checkpoint is not None:
        ckpt_path = Path(_cli_args.checkpoint)
        if ckpt_path.is_file():
            print(f"[train_delivery_spot] Loading checkpoint from: {ckpt_path}", flush=True)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            ac.load_state_dict(ckpt["model"])
            episodes = int(ckpt.get("episodes", 0))
            # Next checkpoint boundary after the resumed episode count.
            next_ckpt_ep = ((episodes // 500) + 1) * 500
            print(f"[train_delivery_spot] Resumed model; starting from episode {episodes}.", flush=True)
        else:
            print(
                f"[train_delivery_spot] WARNING: checkpoint not found at '{ckpt_path}', starting from scratch.",
                flush=True,
            )

    obs = env.reset()

    for it in range(cfg.max_iterations):
        obs_buf = []
        act_buf = []
        logp_buf = []
        rew_buf = []
        val_buf = []
        done_buf = []

        for step in range(cfg.num_steps_per_env):
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action_t, logp_t, value_t = ac(obs_t)
            action = action_t.squeeze(0).cpu().numpy()
            logp = logp_t.squeeze(0).cpu().numpy()
            value = value_t.squeeze(0).cpu().numpy()

            next_obs, reward, done = env.step(action)

            obs_buf.append(obs.copy())
            act_buf.append(action.copy())
            logp_buf.append(logp)
            rew_buf.append(reward)
            val_buf.append(value)
            done_buf.append(done)

            if done:
                episodes += 1
                # Save a checkpoint every 500 episodes.
                if episodes >= next_ckpt_ep:
                    ckpt_dir = REPO_ROOT / "spot" / "checkpoints"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = ckpt_dir / f"spot_policy_ep{episodes}.pt"
                    torch.save(
                        {
                            "model_state_dict": ac.state_dict(),
                            "cfg": cfg.__dict__,
                            "episodes": episodes,
                        },
                        ckpt_path,
                    )
                    print(f"[train_delivery_spot] Saved checkpoint: {ckpt_path}", flush=True)
                    next_ckpt_ep += 500

                obs = env.reset()
            else:
                obs = next_obs

        # Bootstrap last value
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            _, _, last_value_t = ac(obs_t)
        last_value = last_value_t.squeeze(0).cpu().numpy()
        val_buf.append(last_value)

        rewards = np.asarray(rew_buf, dtype=np.float32)
        values = np.asarray(val_buf, dtype=np.float32)
        dones = np.asarray(done_buf, dtype=bool)

        returns, advantages = compute_gae(cfg, rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_tensor = torch.from_numpy(np.asarray(obs_buf, dtype=np.float32)).to(device)
        act_tensor = torch.from_numpy(np.asarray(act_buf, dtype=np.float32)).to(device)
        old_logp_tensor = torch.from_numpy(np.asarray(logp_buf, dtype=np.float32)).to(device)
        return_tensor = torch.from_numpy(returns.astype(np.float32)).to(device)
        adv_tensor = torch.from_numpy(advantages.astype(np.float32)).to(device)

        batch_size = cfg.num_steps_per_env
        minibatch_size = batch_size // cfg.num_mini_batches

        for epoch in range(cfg.num_learning_epochs):
            idxs = np.random.permutation(batch_size)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_idx = idxs[start:end]

                mb_obs = obs_tensor[mb_idx]
                mb_actions = act_tensor[mb_idx]
                mb_old_logp = old_logp_tensor[mb_idx]
                mb_returns = return_tensor[mb_idx]
                mb_advs = adv_tensor[mb_idx]

                new_logp, entropy, value = ac.evaluate_actions(mb_obs, mb_actions)

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_advs
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * mb_advs
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = cfg.value_loss_coef * (mb_returns - value).pow(2).mean()
                entropy_loss = -cfg.entropy_coef * entropy.mean()

                loss = policy_loss + value_loss + entropy_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), cfg.max_grad_norm)
                optimizer.step()

        if (it + 1) % 10 == 0:
            avg_rew = float(rewards.mean())
            print(f"[train_delivery_spot] Iter {it+1}/{cfg.max_iterations}  avg_reward={avg_rew:.3f}", flush=True)

    # ------------------------------------------------------------------
    # Export trained policy as TorchScript and write to spot/spot_policy.pt
    # ------------------------------------------------------------------
    ac.eval()
    print("[train_delivery_spot] Training complete, exporting policy...", flush=True)
    dummy_obs = torch.zeros(1, cfg.obs_dim, dtype=torch.float32, device=device)
    traced = torch.jit.trace(lambda x: ac.actor(x), dummy_obs)

    export_dir = REPO_ROOT / "spot"
    export_dir.mkdir(parents=True, exist_ok=True)
    exported_policy_path = export_dir / "spot_policy_latest.pt"
    traced.save(str(exported_policy_path))

    target_path = TARGET_POLICY_PATH
    try:
        if target_path.exists():
            backup_path = target_path.with_name(
                f"{target_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{target_path.suffix}"
            )
            target_path.rename(backup_path)
            print(f"[INFO] Previous Spot policy rotated to: {backup_path}")

        exported_policy_path.replace(target_path)
        print(f"[INFO] Saved trained policy to: {target_path}")
    except Exception as exc:
        print(f"[WARN] Failed to move exported policy to '{target_path}': {exc}")


if __name__ == "__main__":
    try:
        train()
        print("[train_delivery_spot] Done without uncaught exceptions.", flush=True)
    except Exception:
        print("[train_delivery_spot] EXCEPTION during training:", file=sys.stderr)
        traceback.print_exc()
    finally:
        simulation_app.close()
