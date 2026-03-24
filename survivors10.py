"""
PPO Survivors — rsl-rl-lib v5 + moderngl + pygame
===================================================
基於 rsl-rl-example.py 的 PPO 架構，引入 survivors8.py 的：
  - 車輛物理模型 (轉向/油門/阻力/慣性)
  - 環境元素 (牆壁、食物、掠食者)
  - 獎勵機制 (食物、撞牆、被殺、餓死、移動品質)
  - moderngl 渲染

執行方式：
    conda activate isaaclab
    pip install pygame moderngl
    python ppo_survivors.py

鍵盤：
    SPACE  — 暫停 / 繼續
    R      — 重置環境
    Q/ESC  — 離開
    U      — 顯示/隱藏單位
    L      — 顯示/隱藏標籤
    V      — 切換詳細程度 (0-3)
"""

import math
import time
import threading

import numpy as np
import torch
import pygame
import moderngl
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

# ═══════════════════════════════════════════════════════════════════════════
#  Constants (from survivors8.py)
# ═══════════════════════════════════════════════════════════════════════════
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)

# 畫面
SCREEN_W, SCREEN_H = 1280, 800
SCALE = 1.0
WINDOW_W, WINDOW_H = int(SCREEN_W * SCALE), int(SCREEN_H * SCALE)

# 代理物理
NUM_ENVS          = 1       # VecEnv 同時跑的環境數 (每個環境內有 POP_SIZE 個 agent)
POP_SIZE          = 50      # 每個環境中的 agent 數
POP_MAX_SPEED     = 3.5
POP_RADIUS        = 4
POP_DAMPING_FACTOR = 0.25
POP_BACKWARD_FACTOR = 0.33
POP_MAX_STEER     = math.radians(15)
POP_PERCEPTION_RADIUS = 200

# 生存難度
STAGE = 2
STAGE_SURVIVAL_MULTIPLIER = 3
EST_STEPS = math.sqrt(SCREEN_W**2 + SCREEN_H**2) * 0.715 / POP_MAX_SPEED * STAGE_SURVIVAL_MULTIPLIER

# 食物
FOOD_SIZE         = int(POP_SIZE * 1.5)
FOOD_RADIUS       = 3
MAX_ENERGY        = 100.0
FOOD_ENERGY       = 25.0
ENERGY_DECAY      = FOOD_ENERGY / EST_STEPS
MOVE_FOOD         = False

# 掠食者
PREDATOR_SIZE     = 5
PREDATOR_RADIUS   = 20.0
PREDATOR_MIN_SPEED = 1.5
PREDATOR_MAX_SPEED = 3.0
POP_ALERT_RADIUS  = max(POP_MAX_SPEED, (PREDATOR_MAX_SPEED / POP_MAX_SPEED) ** 2 * 20.58)
RND_POS_PADDING   = POP_RADIUS + POP_ALERT_RADIUS

# 獎懲
FOOD_REWARD        = 50.0
KILLED_REWARD      = -75.0
COLLIDED_REWARD    = -60.0
STARVED_REWARD     = -70.0
MOVE_REWARD_FACTOR = 0.35
MOVE_REWARD        = FOOD_REWARD * MOVE_REWARD_FACTOR / EST_STEPS
TIME_PENALTY_FACTOR = 0.15
STEP_REWARD        = STARVED_REWARD * TIME_PENALTY_FACTOR / EST_STEPS
WALL_NEARBY_REWARD = COLLIDED_REWARD * 0.25
PREDATOR_NEARBY_REWARD = KILLED_REWARD * 0.25

# 觀測維度
FEAT_IN_DIM  = 8   # 每物件特徵 [cos, sin, dist, energy, is_wall, is_food, is_team, is_pred]
STATE_IN_DIM = 7   # 自身狀態
MAX_OBJ      = 25
OBS_DIM      = FEAT_IN_DIM * MAX_OBJ + STATE_IN_DIM  # 展平後的觀測向量大小
NUM_ACTIONS  = 2   # [轉向, 油門]

# PPO 最大步數 (episode truncation)
MAX_EPISODE_STEPS = int(EST_STEPS * 2)

FPS_RENDER = 60


# ═══════════════════════════════════════════════════════════════════════════
#  VecEnv — rsl-rl 介面，內部維護 survivors8 的物理
# ═══════════════════════════════════════════════════════════════════════════
class SurvivorsVecEnv(VecEnv):
    """
    每個 "env" 維護 POP_SIZE 個 agent。
    rsl-rl 看到的 num_envs = NUM_ENVS * POP_SIZE (展平)。

    Observation TensorDict:
        "policy" → (total_agents, OBS_DIM)
    """

    def __init__(self, num_envs: int = NUM_ENVS, device: str = DEVICE_STR):
        self._n_worlds = num_envs
        self._pop = POP_SIZE
        total = self._n_worlds * self._pop

        # ── VecEnv 必要屬性 ──
        self.num_envs           = total
        self.num_actions        = NUM_ACTIONS
        self.max_episode_length = MAX_EPISODE_STEPS
        self.episode_length_buf = torch.zeros(total, dtype=torch.long, device=device)
        self.device             = device
        self.cfg                = {}

        self.screen_size = torch.tensor([SCREEN_W, SCREEN_H], device=device, dtype=torch.float)
        self.bounds = self.screen_size - 1.0
        self.throttle_factor = POP_MAX_SPEED * POP_DAMPING_FACTOR

        # 統計
        self.total_episodes  = 0
        self.total_successes = 0
        self.reward_buf: list[float] = []
        self._ep_reward = torch.zeros(total, dtype=torch.float32, device=device)

        # 環境統計 (for display)
        self.energy_avg  = 0.0
        self.rewards_avg = 0.0
        self.eaten   = 0
        self.killed  = 0
        self.collided = 0
        self.starved = 0
        self.frames  = 0

        # 牆壁
        self._build_walls()

        # one-hot labels
        self.l_wall = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).view(1, 4, 1).expand(
            total, 4, self.wall_A.shape[0])
        l_team = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device).view(1, 4, 1)
        l_food = torch.tensor([0.0, 0.0, 1.0, 0.0], device=device).view(1, 4, 1)
        l_pred = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device).view(1, 4, 1)
        self.l_all = torch.cat([
            l_team.repeat(1, 1, self._pop),
            l_food.repeat(1, 1, FOOD_SIZE),
            l_pred.repeat(1, 1, PREDATOR_SIZE)
        ], dim=2).expand(total, -1, -1).clone()

        self.all_radius = torch.cat([
            torch.full((self._pop,), POP_RADIUS, device=device),
            torch.full((FOOD_SIZE,), FOOD_RADIUS, device=device),
            torch.full((PREDATOR_SIZE,), PREDATOR_RADIUS, device=device)
        ])

        self._reset_all()

    def _build_walls(self):
        wall_A, wall_B = [], []
        pts = torch.tensor([
            [0, 0], [SCREEN_W-1, 0], [SCREEN_W-1, SCREEN_H-1], [0, SCREEN_H-1]
        ], dtype=torch.float32, device=self.device)
        wall_A.append(pts)
        wall_B.append(torch.roll(pts, -1, 0))
        self.wall_A = torch.cat(wall_A, dim=0)
        self.wall_B = torch.cat(wall_B, dim=0)
        self.wall_v = torch.stack([self.wall_A, self.wall_B], dim=1)

    # ── 重置 ──
    def _reset_all(self):
        N = self.num_envs
        dev = self.device
        self.pos = torch.rand(N, 2, device=dev) * self.screen_size
        self.last_pos = self.pos.clone()
        self.angle = torch.rand(N, device=dev) * (2 * math.pi)
        self.last_angle = self.angle.clone()
        self.vel = torch.zeros(N, 2, device=dev)
        self.forward_speed = torch.zeros(N, device=dev)
        self.energy = torch.full((N,), MAX_ENERGY, device=dev)
        self.alive = torch.ones(N, dtype=torch.bool, device=dev)
        self.respawn_timer = torch.zeros(N, dtype=torch.long, device=dev)
        self.last_actions = torch.zeros(N, NUM_ACTIONS, device=dev)

        self.pred_pos = torch.rand(PREDATOR_SIZE, 2, device=dev) * self.screen_size
        self.pred_vel = (torch.rand(PREDATOR_SIZE, 2, device=dev) - 0.5) * 3.5
        self.food_pos = self._respawn_food(FOOD_SIZE)
        self.food_vel = (torch.rand(FOOD_SIZE, 2, device=dev) - 0.5) * 3.5

        self.episode_length_buf.zero_()
        self._ep_reward.zero_()
        self.wall_lines = None
        self.color = torch.ones(N, 3, device=dev) * 0.5

        # initial obs cache
        self._last_states = self._get_states()

    def _reset_agents(self, mask):
        """Reset specific agents by mask."""
        if not mask.any():
            return
        n = mask.sum().item()
        dev = self.device
        self.pos[mask] = self._get_safe_pos(n)
        self.last_pos[mask] = self.pos[mask].clone()
        self.angle[mask] = torch.rand(n, device=dev) * (2 * math.pi)
        self.last_angle[mask] = self.angle[mask].clone()
        self.vel[mask] = 0
        self.forward_speed[mask] = 0
        self.energy[mask] = MAX_ENERGY
        self.alive[mask] = True
        self.respawn_timer[mask] = 0
        self.last_actions[mask] = 0
        self.episode_length_buf[mask] = 0

    # ── 位置生成 ──
    def _get_safe_pos(self, n):
        num_samples = max(n * 10, max(self._pop, PREDATOR_SIZE))
        samples = torch.empty(num_samples, 2, device=self.device)
        samples[:, 0].uniform_(RND_POS_PADDING, SCREEN_W - RND_POS_PADDING)
        samples[:, 1].uniform_(RND_POS_PADDING, SCREEN_H - RND_POS_PADDING)
        if PREDATOR_SIZE > 0:
            obstacles = torch.cat([self.pos[self.alive], self.pred_pos], dim=0)
            if obstacles.shape[0] > 0:
                dists = torch.cdist(samples, obstacles)
                min_dists = dists.min(dim=1).values
                _, top_idx = torch.topk(min_dists, k=min(n, num_samples))
                return samples[top_idx[:n]]
        return samples[:n]

    def _respawn_food(self, n):
        pad = RND_POS_PADDING
        candidates = torch.rand(n, 30, 2, device=self.device)
        candidates[..., 0] = candidates[..., 0] * (SCREEN_W - 2*pad) + pad
        candidates[..., 1] = candidates[..., 1] * (SCREEN_H - 2*pad) + pad

        all_obstacles = torch.cat([self.pos[self.alive], self.pred_pos], dim=0)
        if all_obstacles.shape[0] == 0:
            return candidates[:, 0, :]

        all_obstacles = all_obstacles.view(1, 1, -1, 2)
        dists_all = torch.norm(candidates.unsqueeze(2) - all_obstacles, dim=-1)
        min_dists = dists_all.min(dim=2)[0]
        top_k = min(5, 30)
        _, top_indices = torch.topk(min_dists, k=top_k, dim=1)
        rand_pick = torch.randint(0, top_k, (n,), device=self.device)
        final_indices = top_indices[torch.arange(n, device=self.device), rand_pick]
        return candidates[torch.arange(n, device=self.device), final_indices]

    def _update_entities(self, pos, vel, radius, min_speed, max_speed, jitter_chance=0.05):
        change_mask = torch.rand(pos.shape[0], 1, device=self.device) < jitter_chance
        vel.add_((torch.rand_like(vel) - 0.5) * (1.8 * change_mask))
        speeds = torch.norm(vel, dim=1, keepdim=True)
        new_speeds = torch.clamp(speeds, min_speed, max_speed)
        vel.mul_(new_speeds / (speeds + 1e-6))
        pos.add_(vel)
        min_bound = torch.full_like(self.bounds, radius)
        max_bound = self.bounds - radius
        out_of_bounds = (pos < min_bound) | (pos > max_bound)
        vel[out_of_bounds] *= -1.0
        pos.clamp_(min_bound, max_bound)
        return pos, vel

    # ── 觀測構建 ──
    def _get_states(self):
        N = self.num_envs
        dev = self.device
        angel = self.angle.view(N, 1)
        p = self.pos.unsqueeze(1)

        # 牆壁
        a = self.wall_A.unsqueeze(0)
        b = self.wall_B.unsqueeze(0)
        ab = b - a
        ap = p - a
        t = (torch.sum(ap * ab, dim=-1) / (torch.sum(ab * ab, dim=-1) + 1e-6)).clamp(0, 1)
        closest_points = a + t.unsqueeze(-1) * ab
        diff_wall = closest_points - p
        wall_dist = torch.norm(diff_wall, dim=-1)
        abs_ang_w = torch.atan2(diff_wall[..., 1], diff_wall[..., 0])
        rel_ang_w = abs_ang_w - angel
        wall_dist_val = (wall_dist - POP_RADIUS) / POP_PERCEPTION_RADIUS
        wall_mask = ((wall_dist - POP_RADIUS) < POP_PERCEPTION_RADIUS).float()
        wall_phys = torch.stack([
            torch.cos(rel_ang_w), torch.sin(rel_ang_w),
            wall_dist_val, torch.zeros_like(wall_dist)
        ], dim=1)
        wall_in = torch.cat([wall_phys, self.l_wall], dim=1)

        # 動態物件
        dynamic_pos = torch.cat([self.pos, self.food_pos, self.pred_pos], dim=0)
        diff_d = dynamic_pos.unsqueeze(0) - p
        dist_d = torch.norm(diff_d, dim=2)
        # 排除自己
        diag_idx = torch.arange(N, device=dev)
        if N <= dynamic_pos.shape[0]:
            dist_d[diag_idx, diag_idx] = 1e6

        abs_ang_d = torch.atan2(diff_d[..., 1], diff_d[..., 0])
        rel_ang_d = abs_ang_d - angel
        dist_val = (dist_d - POP_RADIUS - self.all_radius) / POP_PERCEPTION_RADIUS
        dyna_mask = ((dist_d - POP_RADIUS - self.all_radius) < POP_PERCEPTION_RADIUS).float()
        if N <= self._pop:
            dyna_mask[diag_idx, diag_idx] = 0.0

        energy_norm = torch.zeros(N, dist_d.shape[1], device=dev)
        energy_norm[:, :N] = (self.energy / MAX_ENERGY).view(1, -1)
        if PREDATOR_SIZE > 0:
            energy_norm[:, -PREDATOR_SIZE:] = 1.0

        phys_d = torch.stack([
            torch.cos(rel_ang_d), torch.sin(rel_ang_d),
            dist_val, energy_norm
        ], dim=1)
        dynamic_in = torch.cat([phys_d, self.l_all], dim=1)

        # 合併並取 top-k
        all_in = torch.cat([wall_in, dynamic_in], dim=2)
        all_mask = torch.cat([wall_mask, dyna_mask], dim=1)
        all_in = all_in * all_mask.unsqueeze(1)
        all_dists = all_in[:, 2, :].clone()
        all_dists[all_mask == 0] = 1e6
        num_to_take = min(all_in.shape[2], MAX_OBJ)
        _, indices = torch.topk(all_dists, k=num_to_take, dim=1, largest=False)
        indices_expanded = indices.unsqueeze(1).expand(-1, FEAT_IN_DIM, -1)
        sorted_in = torch.gather(all_in, 2, indices_expanded)
        final_mask = torch.gather(all_mask, 1, indices)
        sorted_in = sorted_in * final_mask.unsqueeze(1)

        mixed_in = torch.zeros(N, FEAT_IN_DIM, MAX_OBJ, device=dev)
        mixed_in[:, :, :num_to_take] = sorted_in

        # 自身狀態
        delta_pos_global = self.pos - self.last_pos
        cos_a = torch.cos(self.angle)
        sin_a = torch.sin(self.angle)
        dx_ego = delta_pos_global[:, 0] * cos_a + delta_pos_global[:, 1] * sin_a
        dy_ego = -delta_pos_global[:, 0] * sin_a + delta_pos_global[:, 1] * cos_a
        side_speed = torch.linalg.vecdot(self.vel, torch.stack([-sin_a, cos_a], dim=1))
        omega_yaw = (self.angle - self.last_angle) / POP_MAX_STEER

        self_in = torch.stack([
            self.forward_speed / POP_MAX_SPEED,
            side_speed / POP_MAX_SPEED,
            self.last_actions[:, 0],
            self.energy / MAX_ENERGY,
            dx_ego / POP_MAX_SPEED,
            dy_ego / POP_MAX_SPEED,
            omega_yaw
        ], dim=1)

        self.last_pos = self.pos.clone().detach()
        self.last_angle = self.angle.clone().detach()

        # 顏色 (for display)
        en_ratio = (self.energy / MAX_ENERGY).view(-1, 1)
        self.color = torch.cat([en_ratio, 0.5 * en_ratio, 1.0 - en_ratio], dim=1)

        return mixed_in, self_in

    def _build_obs(self) -> TensorDict:
        mixed_in, self_in = self._last_states
        # 展平: (N, FEAT_IN_DIM * MAX_OBJ + STATE_IN_DIM)
        flat_env = mixed_in.reshape(self.num_envs, -1)
        obs = torch.cat([flat_env, self_in], dim=1)
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)

    # ── VecEnv interface ──
    def get_observations(self) -> TensorDict:
        return self._build_obs()

    def step(self, actions: torch.Tensor):
        """
        actions: (num_envs, 2) — continuous [steer, throttle] from PPO Gaussian
        """
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        # Clamp to [-1, 1]
        actions = actions.clamp(-1, 1)

        N = self.num_envs
        dev = self.device

        was_alive = self.alive.clone()
        rewards = torch.full((N,), STEP_REWARD, device=dev)

        steer_vals = actions[:, 0]
        throttle_vals = actions[:, 1]

        # ── Agent 物理 ──
        speed_vals = torch.norm(self.vel, dim=1)
        sensitivity = torch.clamp(speed_vals / 0.2, min=0.0, max=1.0).pow(2.0)
        high_speed_damping = torch.clamp(1.875 - (speed_vals / POP_MAX_SPEED), min=0.7, max=1.0)
        steer_delta = steer_vals * POP_MAX_STEER * sensitivity * high_speed_damping
        self.angle += steer_delta * self.alive.float()
        pop_vecs = torch.stack([torch.cos(self.angle), torch.sin(self.angle)], dim=1)
        throttle_factors = torch.where(throttle_vals > 0,
                                       self.throttle_factor, self.throttle_factor * POP_BACKWARD_FACTOR)
        throttles = (throttle_vals * throttle_factors).unsqueeze(1)
        thrust = pop_vecs * throttles
        self.vel = (self.vel * (1 - POP_DAMPING_FACTOR) + thrust) * self.alive.unsqueeze(1)
        self.pos += self.vel * self.alive.unsqueeze(1)

        # ── 掠食者 ──
        if PREDATOR_SIZE > 0:
            self.pred_pos, self.pred_vel = self._update_entities(
                self.pred_pos, self.pred_vel, PREDATOR_RADIUS,
                min_speed=PREDATOR_MIN_SPEED, max_speed=PREDATOR_MAX_SPEED)
            dist_pred = torch.cdist(self.pos, self.pred_pos)
            killed = (dist_pred < POP_RADIUS + PREDATOR_RADIUS).any(dim=1) & self.alive
            rewards[killed] += KILLED_REWARD
            self._kill(killed)
            pred_mask = (dist_pred - POP_RADIUS - PREDATOR_RADIUS < POP_ALERT_RADIUS).any(dim=1) & self.alive
            rewards[pred_mask] += PREDATOR_NEARBY_REWARD
        else:
            killed = torch.zeros(N, dtype=torch.bool, device=dev)

        # ── 撞牆 ──
        p = self.pos.unsqueeze(1)
        a = self.wall_A.unsqueeze(0)
        b = self.wall_B.unsqueeze(0)
        ab = b - a
        ap = p - a
        t = (torch.sum(ap * ab, dim=-1) / (torch.sum(ab * ab, dim=-1) + 1e-6)).clamp(0, 1)
        wall_closest_points = a + t.unsqueeze(-1) * ab
        dist_to_walls = torch.norm(self.pos.unsqueeze(1) - wall_closest_points, dim=-1)
        hit_walls = dist_to_walls < POP_RADIUS
        collided = torch.any(hit_walls, dim=1) & self.alive
        rewards[collided] += COLLIDED_REWARD
        self._kill(collided)

        # ── 食物 ──
        if FOOD_SIZE > 0:
            dist_food = torch.cdist(self.pos, self.food_pos)
            hits_food = (dist_food < POP_RADIUS + FOOD_RADIUS) & self.alive.unsqueeze(1)
            if hits_food.any():
                masked_dist = torch.where(hits_food, dist_food, torch.tensor(float('inf'), device=dev))
                min_dists, closest_a_idx = torch.min(masked_dist, dim=0)
                valid_eaten_mask = min_dists != float('inf')
                f_idx = torch.where(valid_eaten_mask)[0]
                a_idx = closest_a_idx[valid_eaten_mask]
                num_eaten = len(a_idx)
                rewards.index_add_(0, a_idx, torch.full((num_eaten,), FOOD_REWARD, device=dev))
                self.energy.index_add_(0, a_idx, torch.full((num_eaten,), FOOD_ENERGY, device=dev))
                self.energy = torch.clamp(self.energy, max=MAX_ENERGY)
                self.food_pos[f_idx] = self._respawn_food(len(f_idx))
                self.eaten += len(f_idx)

        # ── 能量消耗 ──
        static_cost = 0.2 * ENERGY_DECAY
        dynamic_cost = 0.8 * ENERGY_DECAY * torch.pow(throttle_vals, 2)
        self.energy -= (static_cost + dynamic_cost) * self.alive.float()
        starved = (self.energy <= 0) & self.alive
        rewards[starved] += STARVED_REWARD
        self._kill(starved)

        # ── 移動獎勵 ──
        self.forward_speed = torch.linalg.vecdot(self.vel, pop_vecs)
        vel_mag = torch.norm(self.vel, dim=1, keepdim=True) + 1e-6
        vel_purity = torch.relu(torch.linalg.vecdot(self.vel / vel_mag, pop_vecs))
        move_reward = MOVE_REWARD * 2.5 * (torch.relu(self.forward_speed) / POP_MAX_SPEED) * vel_purity
        eff_move_factor = torch.pow(vel_purity, 4)
        move_reward = move_reward * eff_move_factor
        action_diff = actions - self.last_actions
        smooth_penalty = MOVE_REWARD * 0.2 * torch.mean(action_diff.pow(2), dim=1)
        abs_throttle = torch.abs(throttle_vals)
        throttle_penalty = MOVE_REWARD * 0.2 * torch.relu(abs_throttle - 0.75)
        steer_penalty = MOVE_REWARD * 0.3 * torch.pow(steer_vals * throttle_vals, 2)
        purity_gap = torch.relu(0.95 - vel_purity)
        spinning_bonus = MOVE_REWARD * 5.0 * torch.pow(purity_gap, 2) * torch.abs(steer_vals)
        spinning_penalty = MOVE_REWARD * 3.0 * torch.pow(steer_vals, 2) * (1.0 - vel_purity)
        omega_yaw = (self.angle - self.last_angle) / POP_MAX_STEER
        yaw_penalty = MOVE_REWARD * 20.0 * torch.abs(omega_yaw)
        total_step_reward = (move_reward - smooth_penalty - throttle_penalty
                             - steer_penalty - spinning_penalty - spinning_bonus - yaw_penalty)
        rewards += total_step_reward * self.alive.float()

        # ── 近牆痛覺 ──
        wall_min_dist, closest_wall_idx = torch.min(dist_to_walls, dim=1)
        wall_dist_ratio = (1.0 - (wall_min_dist - POP_RADIUS) / POP_ALERT_RADIUS).clamp(0.0, 1.0)
        wall_nearby_mask = (wall_dist_ratio > 0) & self.alive
        if wall_nearby_mask.any():
            wall_contacts = wall_closest_points[torch.arange(N, device=dev), closest_wall_idx]
            starts = self.pos[wall_nearby_mask]
            ends = wall_contacts[wall_nearby_mask]
            self.wall_lines = torch.stack([starts, ends], dim=1)
            rewards[wall_nearby_mask] += WALL_NEARBY_REWARD
        else:
            self.wall_lines = None

        dead_mask = killed | collided | starved

        # ── 更新觀測 ──
        self._last_states = self._get_states()

        # ── 復活 ──
        self.respawn_timer[~self.alive] -= 1
        ready = ~self.alive & (self.respawn_timer <= 0)
        if ready.any():
            self._reset_agents(ready)

        self.last_actions = actions.detach().clone()
        self.episode_length_buf += 1

        # ── dones 計算 ──
        # 對 rsl-rl: done = 死亡 或 超時
        time_out = (self.episode_length_buf >= MAX_EPISODE_STEPS) & self.alive
        dones = torch.zeros(N, device=dev)
        dones[dead_mask] = 1.0

        time_outs = torch.zeros(N, device=dev)
        time_outs[time_out] = 1.0
        dones[time_out] = 1.0

        # 對 done 的 agent 重置 episode counter
        done_all = dones.bool()
        for i in range(N):
            self._ep_reward[i] += rewards[i].item()
            if done_all[i]:
                self.total_episodes += 1
                self.reward_buf.append(float(self._ep_reward[i]))
                if len(self.reward_buf) > 200:
                    self.reward_buf.pop(0)
                self._ep_reward[i] = 0.0
                self.episode_length_buf[i] = 0

        # 統計
        self.energy_avg = self.energy_avg * 0.99 + (self.energy.sum().item() / N) * 0.01
        self.rewards_avg = self.rewards_avg * 0.99 + (rewards.sum().item() / N) * 0.01
        self.killed += killed.sum().item()
        self.collided += collided.sum().item()
        self.starved += starved.sum().item()
        self.rewards_last = rewards
        self.frames += 1

        obs_td = self._build_obs()
        rews_t = rewards
        dones_t = dones
        extras = {"time_outs": time_outs}

        return obs_td, rews_t, dones_t, extras

    def _kill(self, mask):
        self.alive[mask] = False
        n = mask.sum().item()
        if n > 0:
            self.respawn_timer[mask] = 1 if self._pop == 1 else torch.randint(
                30, 180, (n,), device=self.device)


# ═══════════════════════════════════════════════════════════════════════════
#  Config — rsl-rl PPO
# ═══════════════════════════════════════════════════════════════════════════
def make_train_cfg():
    return {
        "seed": 42,
        "num_steps_per_env": 64,
        "save_interval": 500,

        "algorithm": {
            "class_name":             "PPO",
            "gamma":                  0.97,
            "lam":                    0.95,
            "value_loss_coef":        1.0,
            "use_clipped_value_loss": True,
            "clip_param":             0.2,
            "entropy_coef":           0.01,
            "num_learning_epochs":    4,
            "num_mini_batches":       4,
            "learning_rate":          3e-4,
            "schedule":               "adaptive",
            "desired_kl":             0.01,
            "max_grad_norm":          1.0,
        },

        "actor": {
            "class_name":  "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation":  "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
            },
        },
        "critic": {
            "class_name":  "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation":  "elu",
        },

        "obs_groups": {
            "actor":  ["policy"],
            "critic": ["policy"],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  ModernGL Renderer (from survivors8.py, with fixes)
# ═══════════════════════════════════════════════════════════════════════════
class GLRenderer:
    def __init__(self, ctx, fbo_texture, w, h):
        self.ctx = ctx
        self.fbo_texture = fbo_texture
        self.w, self.h = w, h

        # --- Circle Shader (Instanced) ---
        self.circle_prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert; in vec4 in_pos_rad; in vec3 in_color;
                out vec3 v_color;
                out vec2 v_dist;
                uniform vec2 res;
                void main() {
                    vec2 p = (in_pos_rad.xy / res) * 2.0 - 1.0;
                    p.y *= -1.0;
                    vec2 scale = (in_pos_rad.z / res) * 2.0;
                    gl_Position = vec4(p + in_vert * scale, 0.0, 1.0);
                    v_color = in_color;
                    v_dist = in_vert;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color;
                in vec2 v_dist;
                out vec4 f;
                uniform bool is_hollow;
                uniform float thickness;
                void main() {
                    float d = length(v_dist);
                    if (d > 1.0) discard;
                    if (is_hollow) {
                        if (d < (1.0 - thickness)) discard;
                    }
                    f = vec4(v_color, 1.0);
                }
            """
        )
        self.circle_prog['res'].value = (w, h)

        # 圓形模板 (用 TRIANGLE_FAN: center + ring)
        n_seg = 32
        # Center vertex + ring vertices
        verts = np.zeros((n_seg + 2, 2), dtype='f4')
        verts[0] = [0.0, 0.0]  # center
        ang = np.linspace(0, 2 * np.pi, n_seg + 1, endpoint=True)
        verts[1:, 0] = np.cos(ang)
        verts[1:, 1] = np.sin(ang)
        self.vbo_circle_temp = ctx.buffer(verts)

        max_instances = 10000
        self.vbo_circle_inst = ctx.buffer(reserve=max_instances * 7 * 4)
        self.vao_circle = ctx.vertex_array(self.circle_prog, [
            (self.vbo_circle_temp, '2f', 'in_vert'),
            (self.vbo_circle_inst, '4f 3f /i', 'in_pos_rad', 'in_color')
        ])

        # --- Line Shader ---
        self.line_prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert; in vec3 in_color;
                out vec3 v_color;
                uniform vec2 res;
                void main() {
                    vec2 pixel_pos = in_vert + vec2(0.5, 0.5);
                    vec2 p = (pixel_pos / res) * 2.0 - 1.0;
                    p.y *= -1.0;
                    gl_Position = vec4(p, 0.0, 1.0);
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color; out vec4 f;
                void main() { f = vec4(v_color, 1.0); }
            """
        )
        self.line_prog['res'].value = (w, h)
        self.vbo_line = ctx.buffer(reserve=20000 * 5 * 4)
        self.vao_line = ctx.vertex_array(self.line_prog, [
            (self.vbo_line, '2f 3f', 'in_vert', 'in_color')
        ])

        # --- Text Overlay ---
        self.text_texture = ctx.texture((w, h), 4)
        self.text_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.text_prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert; in vec2 in_texcoord;
                out vec2 v_texcoord;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_texcoord = in_texcoord;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D tex;
                in vec2 v_texcoord;
                out vec4 f_color;
                void main() { f_color = texture(tex, v_texcoord); }
            """
        )
        quad = np.array([
            -1, 1, 0, 0,  -1, -1, 0, 1,  1, 1, 1, 0,
            -1, -1, 0, 1,  1, -1, 1, 1,  1, 1, 1, 0,
        ], dtype='f4')
        self.vbo_text = ctx.buffer(quad)
        self.vao_text = ctx.vertex_array(self.text_prog, [
            (self.vbo_text, '2f 2f', 'in_vert', 'in_texcoord')
        ])

        # --- Screen blit ---
        self.screen_prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert; out vec2 uv;
                void main() { uv = (in_vert + 1.0) * 0.5; gl_Position = vec4(in_vert, 0.0, 1.0); }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D tex; in vec2 uv; out vec4 fragColor;
                void main() { fragColor = texture(tex, uv); }
            """,
        )
        quad2 = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype='f4')
        self.vbo_screen = ctx.buffer(quad2)
        self.vao_screen = ctx.simple_vertex_array(self.screen_prog, self.vbo_screen, 'in_vert')

    def draw_circles(self, pos, rad, color):
        if pos.shape[0] == 0:
            return
        data = torch.cat([pos, rad, rad, color], dim=1).cpu().numpy().astype('f4')
        self.vbo_circle_inst.write(data)
        self.vao_circle.render(moderngl.TRIANGLE_FAN, instances=pos.shape[0])

    def draw_lines(self, start, end, color):
        if start.shape[0] == 0:
            return
        N = start.shape[0]
        data = torch.empty((N, 2, 5), device=start.device)
        data[:, 0, :2] = start
        data[:, 0, 2:] = color
        data[:, 1, :2] = end
        data[:, 1, 2:] = color
        self.vbo_line.write(data.cpu().numpy().astype('f4'))
        self.vao_line.render(moderngl.LINES, vertices=N * 2)

    def draw_text(self, surface):
        rgba_data = pygame.image.tostring(surface, 'RGBA')
        self.text_texture.write(rgba_data)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.text_texture.use(0)
        self.vao_text.render()

    def blit_to_screen(self, win_w, win_h):
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, win_w, win_h)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)
        self.fbo_texture.use(0)
        self.vao_screen.render(moderngl.TRIANGLE_STRIP)


# ═══════════════════════════════════════════════════════════════════════════
#  Pygame Renderer Wrapper
# ═══════════════════════════════════════════════════════════════════════════
class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(
            (WINDOW_W, WINDOW_H),
            pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        )
        pygame.display.set_caption("PPO Survivors — rsl-rl-lib v5")
        self.ctx = moderngl.create_context()
        fbo_texture = self.ctx.texture((SCREEN_W, SCREEN_H), 3)
        fbo_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.fbo = self.ctx.framebuffer(color_attachments=[fbo_texture])
        self.gl = GLRenderer(self.ctx, fbo_texture, SCREEN_W, SCREEN_H)
        self.ui_surface = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        self.font = pygame.font.SysFont("Consolas", 14)
        self.big_font = pygame.font.SysFont("Consolas", 18, bold=True)
        self._clk = pygame.time.Clock()
        self.draw_units = True
        self.draw_label = True
        self.verbose = 0

    def handle_events(self) -> dict:
        ev = {"quit": False, "pause": False, "reset": False}
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                ev["quit"] = True
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE):
                    ev["quit"] = True
                elif e.key == pygame.K_SPACE:
                    ev["pause"] = True
                elif e.key == pygame.K_r:
                    ev["reset"] = True
                elif e.key == pygame.K_u:
                    self.draw_units = not self.draw_units
                elif e.key == pygame.K_l:
                    self.draw_label = not self.draw_label
                elif e.key == pygame.K_v:
                    self.verbose = (self.verbose + 1) % 4
        return ev

    def draw(self, env: SurvivorsVecEnv, stats: dict):
        self.fbo.use()
        self.ctx.clear(0.08, 0.08, 0.08)
        self.ui_surface.fill((0, 0, 0, 0))
        dev = env.device

        if self.draw_units:
            # ── Snapshot all env state atomically to avoid race with training thread ──
            snap_pos = env.pos.clone()
            snap_alive = env.alive.clone()
            snap_energy = env.energy.clone()
            snap_color = env.color.clone()
            snap_vel = env.vel.clone()
            snap_fwd_speed = env.forward_speed.clone()
            snap_angle = env.angle.clone()
            snap_last_actions = env.last_actions.clone()
            snap_food_pos = env.food_pos.clone()
            snap_pred_pos = env.pred_pos.clone()
            snap_wall_lines = env.wall_lines.clone() if env.wall_lines is not None else None

            alive_mask = snap_alive
            dead_mask = ~alive_mask

            # 牆面
            w_starts = env.wall_v[:, 0, :]
            w_ends = env.wall_v[:, 1, :]
            num_walls = w_starts.size(0)
            wall_color = torch.tensor([[0.5, 0.0, 0.0]], device=dev).expand(num_walls, -1)
            self.gl.draw_lines(w_starts, w_ends, wall_color)

            # 食物
            if FOOD_SIZE > 0 and len(snap_food_pos) > 0:
                food_color = torch.tensor([[0.0, 1.0, 0.47]], device=dev).expand(len(snap_food_pos), -1)
                food_rad = torch.full((len(snap_food_pos), 1), FOOD_RADIUS, device=dev)
                self.gl.circle_prog['is_hollow'].value = False
                self.gl.draw_circles(snap_food_pos, food_rad, food_color)

            # 掠食者
            if PREDATOR_SIZE > 0 and len(snap_pred_pos) > 0:
                num_pred = len(snap_pred_pos)
                p_color = torch.tensor([[1.0, 0.0, 0.0]], device=dev).expand(num_pred, -1)
                p_rad_outer = torch.full((num_pred, 1), PREDATOR_RADIUS, device=dev)
                self.gl.circle_prog['is_hollow'].value = True
                self.gl.circle_prog['thickness'].value = 0.05
                self.gl.draw_circles(snap_pred_pos, p_rad_outer, p_color * 0.5)
                p_rad_inner = torch.full((num_pred, 1), PREDATOR_RADIUS * 0.25, device=dev)
                self.gl.circle_prog['is_hollow'].value = False
                self.gl.draw_circles(snap_pred_pos, p_rad_inner, p_color)

            # Dead Agents
            if dead_mask.any():
                d_pos = snap_pos[dead_mask]
                d_color = torch.where(
                    (snap_energy[dead_mask] <= 0).unsqueeze(1),
                    torch.tensor([0.23, 0.23, 0.23], device=dev),
                    torch.tensor([0.47, 0.0, 0.0], device=dev))
                d_rad = torch.full((d_pos.shape[0], 1), 3.0, device=dev)
                self.gl.circle_prog['is_hollow'].value = False
                self.gl.draw_circles(d_pos, d_rad, d_color)

            # Alive Agents
            if alive_mask.any():
                a_pos = snap_pos[alive_mask]
                a_color = snap_color[alive_mask]
                en_ratio = (snap_energy[alive_mask] / MAX_ENERGY).unsqueeze(1)
                a_rad = POP_RADIUS + 4 * en_ratio
                self.gl.circle_prog['is_hollow'].value = False
                self.gl.draw_circles(a_pos, a_rad, a_color)

                # 速度向量
                spd = snap_fwd_speed[alive_mask]
                vel = snap_vel[alive_mask]
                mask_spd = torch.abs(spd) > 0.1
                if mask_spd.any():
                    l_start = a_pos[mask_spd]
                    l_end = l_start + vel[mask_spd] * (torch.abs(spd[mask_spd]) * 2.5).unsqueeze(1)
                    l_color = torch.tensor([[0.0, 0.75, 1.0]], device=dev).expand(l_start.shape[0], -1)
                    self.gl.draw_lines(l_start, l_end, l_color)

                # 油門指示線
                throttle = snap_last_actions[alive_mask, 1]
                mask_t = throttle != 0
                if mask_t.any():
                    t_start = a_pos[mask_t]
                    t_val = throttle[mask_t]
                    t_abs = torch.abs(t_val)
                    t_ang = snap_angle[alive_mask][mask_t]
                    is_forward = t_val > 0
                    t_final_ang = torch.where(is_forward, t_ang, t_ang + math.pi)
                    t_final_len = torch.where(is_forward, 20.0 * t_abs, (20.0 * t_abs) / 3.0)
                    t_dir = torch.stack([torch.cos(t_final_ang), torch.sin(t_final_ang)], dim=1)
                    t_end = t_start + t_dir * t_final_len.unsqueeze(1)
                    num_t = t_start.size(0)
                    t_color = torch.zeros(num_t, 3, device=dev)
                    mask_back = ~is_forward
                    t_color[mask_back] = torch.tensor([1.0, 1.0, 0.0], device=dev)
                    mask_low = is_forward & (t_abs <= 0.5)
                    t_color[mask_low] = torch.tensor([0.0, 1.0, 0.0], device=dev)
                    mask_mid = is_forward & (t_abs > 0.5) & (t_abs <= 0.875)
                    t_color[mask_mid] = torch.tensor([1.0, 1.0, 1.0], device=dev)
                    mask_high = is_forward & (t_abs > 0.875)
                    t_color[mask_high] = torch.tensor([1.0, 0.0, 0.0], device=dev)
                    self.gl.draw_lines(t_start, t_end, t_color)

            # Debug wall lines
            if self.verbose >= 2 and snap_wall_lines is not None:
                w_starts = snap_wall_lines[:, 0, :]
                w_ends = snap_wall_lines[:, 1, :]
                num_w = w_starts.size(0)
                w_line_color = torch.tensor([[1.0, 0.2, 0.2]], device=dev).expand(num_w, -1)
                self.gl.draw_lines(w_starts, w_ends, w_line_color)
                w_dot_color = torch.tensor([[1.0, 1.0, 0.0]], device=dev).expand(num_w, -1)
                w_dot_rad = torch.full((num_w, 1), 3.0, device=dev)
                self.gl.circle_prog['is_hollow'].value = False
                self.gl.draw_circles(w_ends, w_dot_rad, w_dot_color)

        # ── UI text overlay ──
        THEME = {
            "label": (180, 180, 180),
            "perf": (100, 220, 180),
            "param": (255, 200, 100),
            "loss": (255, 120, 120),
            "success": (120, 255, 120)
        }

        if self.draw_label:
            it = stats.get("iteration", 0)
            ep = env.total_episodes
            rew = float(np.mean(env.reward_buf)) if env.reward_buf else 0.0

            labels = [
                ("Iteration:", f"{it:,}", THEME["perf"], True),
                ("Episodes:", f"{ep:,}", THEME["perf"], True),
                ("Frames:", f"{env.frames:,}", THEME["perf"], False),
                ("Energy:", f"{env.energy_avg:.0f}",
                 THEME["success"] if env.energy_avg > 60 else (255, 0, 0), False),
                ("Rewards:", f"{env.rewards_avg:.4f}",
                 THEME["success"] if env.rewards_avg > 0 else (255, 0, 0), False),
                ("Mean Ep Rew:", f"{rew:.3f}", THEME["perf"], False),
                ("Eaten:", f"{env.eaten:,}", THEME["success"], False),
                ("Killed:", f"{env.killed:,}", THEME["loss"], False),
                ("Collided:", f"{env.collided:,}", THEME["loss"], False),
                ("Starved:", f"{env.starved:,}", THEME["loss"], False),
                ("Alive:", f"{int(env.alive.sum())}/{env.num_envs}", THEME["success"], False),
            ]
            paused = stats.get("paused", False)
            if paused:
                labels.append(("[PAUSED]", "", (255, 255, 0), True))

            self._render_labels(labels, 10, 200)

        # keybinds hint
        hint = "SPACE=Pause R=Reset U=Units L=Labels V=Verbose Q=Quit"
        hint_surf = self.font.render(hint, True, (100, 100, 140))
        self.ui_surface.blit(hint_surf, (10, SCREEN_H - 20))

        self.gl.draw_text(self.ui_surface)

        win_w, win_h = pygame.display.get_window_size()
        self.gl.blit_to_screen(win_w, win_h)
        pygame.display.flip()
        self._clk.tick(FPS_RENDER)

    def _render_labels(self, labels, label_x, value_anchor_x, start_y=10, padding=4):
        current_y = start_y
        for text, val, val_color, bold in labels:
            font = self.big_font if bold else self.font
            lbl_surf = font.render(text, True, (180, 180, 180))
            lbl_rect = lbl_surf.get_rect(topleft=(label_x, current_y))
            val_surf = font.render(val, True, val_color)
            val_rect = val_surf.get_rect(topright=(value_anchor_x, current_y))
            self.ui_surface.blit(lbl_surf, lbl_rect)
            self.ui_surface.blit(val_surf, val_rect)
            line_height = max(lbl_rect.height, val_rect.height)
            current_y += line_height + padding

    def quit(self):
        pygame.quit()


# ═══════════════════════════════════════════════════════════════════════════
#  Training thread（讓 pygame 在 main thread 執行）
# ═══════════════════════════════════════════════════════════════════════════
class TrainState:
    def __init__(self):
        self.iteration = 0
        self.paused    = False
        self.stop      = False
        self.reset_req = False
        self.lock      = threading.Lock()


def train_loop(runner: OnPolicyRunner, env: SurvivorsVecEnv,
               state: TrainState, max_iter: int):
    alg       = runner.alg
    steps_per = runner.cfg["num_steps_per_env"]

    alg.train_mode()
    obs_td = env.get_observations()

    for it in range(max_iter):
        while True:
            with state.lock:
                if state.stop:
                    return
                if state.reset_req:
                    env._reset_all()
                    obs_td = env.get_observations()
                    state.reset_req = False
                if not state.paused:
                    break
            time.sleep(0.05)

        # Rollout
        for _ in range(steps_per):
            with torch.no_grad():
                actions = alg.act(obs_td)
            next_obs_td, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs=obs_td, rewards=rewards, dones=dones, extras=extras)
            obs_td = next_obs_td

        # Update
        with torch.no_grad():
            alg.compute_returns(obs_td)
        train_results = alg.update()
        mean_val_loss  = train_results.get("value_loss", 0.0)
        mean_surr_loss = train_results.get("surrogate_loss", 0.0)

        with state.lock:
            state.iteration = it + 1

        if (it + 1) % 50 == 0:
            ep  = env.total_episodes
            rew = float(np.mean(env.reward_buf)) if env.reward_buf else 0.0
            print(f"[Iter {it+1:>5}]  ep={ep:>5}  rew={rew:>7.3f}  "
                  f"eaten={env.eaten:>5}  killed={env.killed>5}  "
                  f"collided={env.collided:>5}  starved={env.starved:>5}  "
                  f"val={mean_val_loss:>2.4f}  surr={mean_surr_loss:>2.4f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    device    = DEVICE_STR
    env       = SurvivorsVecEnv(num_envs=NUM_ENVS, device=device)
    renderer  = Renderer()
    train_cfg = make_train_cfg()
    log_dir   = "logs/ppo_survivors"

    print("=" * 60)
    print("  PPO Survivors — rsl-rl-lib v5")
    print(f"  Agents: {env.num_envs}  ObsDim: {OBS_DIM}  Actions: {NUM_ACTIONS}")
    print(f"  Food: {FOOD_SIZE}  Predators: {PREDATOR_SIZE}")
    print(f"  MaxSteps: {MAX_EPISODE_STEPS}  EstSteps: {EST_STEPS:.0f}")
    print("=" * 60)

    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)
    print("Runner built. Training starts!")
    print("  SPACE=Pause  R=Reset  Q/ESC=Quit  U=Units  L=Labels  V=Verbose")

    state    = TrainState()
    max_iter = 10000

    t = threading.Thread(
        target=train_loop,
        args=(runner, env, state, max_iter),
        daemon=True,
    )
    t.start()

    # ── pygame 主迴圈 ──
    stats: dict = {}
    while t.is_alive():
        ev = renderer.handle_events()
        if ev["quit"]:
            with state.lock:
                state.stop = True
            break
        if ev["pause"]:
            with state.lock:
                state.paused = not state.paused
        if ev["reset"]:
            with state.lock:
                state.reset_req = True

        with state.lock:
            stats["iteration"] = state.iteration
            stats["paused"]    = state.paused

        renderer.draw(env, stats)

    t.join(timeout=3.0)
    print("\nTraining finished, closing.")
    renderer.quit()


if __name__ == "__main__":
    main()