import argparse
import math
import threading

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

import os
from pathlib import Path

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.models import MLPModel

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CAPTION = "Vectra: Apex Protocol"

# 存檔路徑
script_name = Path(__file__).stem
LOG_PATH    = f"logs/{script_name}"

# 畫面
SCREEN_W, SCREEN_H = 1280, 800
SCALE = 1.0
WINDOW_W, WINDOW_H = int(SCREEN_W * SCALE), int(SCREEN_H * SCALE)

# 代理物理
NUM_ENVS          = 1024       # VecEnv 同時跑的環境數 (每個環境內有 POP_SIZE 個 agent)
STAGE             = 5
POP_SIZE          = 1 if STAGE < 1 else 50 if STAGE < 2 else 16 if STAGE < 5 else 1
POP_MAX_SPEED     = 3.5
POP_RADIUS        = 4
POP_DAMPING_FACTOR = 0.25
POP_BACKWARD_FACTOR = 0.33
POP_MAX_STEER     = math.radians(15)
POP_PERCEPTION_RADIUS = 200
POP_PERCEPT_TEAM  = False if STAGE < 2 else True

# 生存難度
STAGE_SURVIVAL_MULTIPLIER = 2 if STAGE < 1 else 10 if STAGE < 2 else 3 if STAGE < 3 else 2 if STAGE < 4 else 1
EST_STEPS = math.sqrt(SCREEN_W**2 + SCREEN_H**2) * 0.715 / POP_MAX_SPEED * STAGE_SURVIVAL_MULTIPLIER

# 食物
FOOD_SIZE         = int(POP_SIZE * (1 if STAGE < 1 else 0.001 if STAGE < 2 else 1.5 if STAGE < 2 else 0.75 if STAGE < 4 else 0.5))
FOOD_RADIUS       = 3
MAX_ENERGY        = 100.0
FOOD_ENERGY       = 25.0
ENERGY_DECAY      = 0 if STAGE == 1 else FOOD_ENERGY / EST_STEPS
MOVE_FOOD         = False if STAGE < 4 else True

# 掠食者
PREDATOR_SIZE     = 0 if STAGE < 3 else 5 if STAGE < 4 else 8
PREDATOR_RADIUS   = 20.0
PREDATOR_MIN_SPEED = 1.5
PREDATOR_MAX_SPEED = 2.5 if STAGE < 3 else 3.0 if STAGE < 4 else 3.4
POP_ALERT_RADIUS  = max(POP_MAX_SPEED, (PREDATOR_MAX_SPEED / POP_MAX_SPEED) ** 2 * 20.58)
RND_POS_PADDING   = POP_RADIUS + POP_ALERT_RADIUS

# 獎懲
FOOD_REWARD        = 100 if STAGE < 1 else 50.0
KILLED_REWARD      = -75.0
COLLIDED_REWARD    = -60.0
STARVED_REWARD     = -70.0
MOVE_REWARD_FACTOR = 0.35
MOVE_REWARD        = FOOD_REWARD * MOVE_REWARD_FACTOR / EST_STEPS
TIME_PENALTY_FACTOR = 0 if STAGE == 1 else 0.15
STEP_REWARD        = STARVED_REWARD * TIME_PENALTY_FACTOR / EST_STEPS
WALL_NEARBY_REWARD = COLLIDED_REWARD * 0.25
PREDATOR_NEARBY_REWARD = KILLED_REWARD * 0.25

# 模型核心參數
FEAT_IN_DIM     = 8     # 每個物件特徵 [cos, sin, dist, energy, is_wall, is_food, is_team, is_pred]
STATE_IN_DIM    = 7     # 自身狀態 [前向速度, 側向速度, 前次轉向指令, 能量, dx_ego, dy_ego, omega_yaw]
ACTOR_OUT_DIM   = 2     # Actor 輸出動作 [轉向, 油門]
HIDDEN_FEAT_DIM = 64    # 特徵提取層 (Conv1d) 的輸出維度
HIDDEN_ATTN_DIM = 32    # Attention 內部隱藏層維度
HIDDEN_FC_DIM   = 256   # 後段全連接層 (MLP) 主要維度
MAX_OBJ         = 25    # 視野內最近距離中最多的環境物件數量
FEAT_FLAT_DIM   = FEAT_IN_DIM * MAX_OBJ
OBS_DIM         = FEAT_FLAT_DIM + STATE_IN_DIM  # 展平後的觀測向量大小 (提取前)
EXTRACTED_OBS_DIM = HIDDEN_FEAT_DIM + STATE_IN_DIM          # 提取後 PPO 看到的觀測維度

# PPO 最大步數 (episode truncation)
MAX_EPISODE_STEPS = int(EST_STEPS * 2)

FPS_RENDER = 60


class SurvivorsCustomModel(MLPModel):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None
    ) -> None:
        # 先調用父類初始化 (注意：此處 MLP 會暫時以原始維度初始化，我們後面會重寫它)
        super().__init__(
            obs, obs_groups, obs_set, output_dim, 
            hidden_dims, activation, obs_normalization, distribution_cfg
        )

        # 1. 空間感知層
        self.conv = nn.Sequential(
            nn.Conv1d(FEAT_IN_DIM, HIDDEN_FEAT_DIM, 1),
            nn.LayerNorm([HIDDEN_FEAT_DIM, MAX_OBJ]),
            nn.ReLU(),
            nn.Conv1d(HIDDEN_FEAT_DIM, HIDDEN_FEAT_DIM, 1),
            nn.ReLU()
        )
        # 2. Attention 機制
        self.attn_weights = nn.Sequential(
            nn.Linear(HIDDEN_FEAT_DIM, HIDDEN_ATTN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_ATTN_DIM, 1)
        )

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: None = None
    ) -> torch.Tensor:
        
        latent = super().get_latent(obs, masks, hidden_state)

        m_flat = latent[:, :FEAT_FLAT_DIM]
        s_in = latent[:, FEAT_FLAT_DIM:]
        m_in = m_flat.view(-1, FEAT_IN_DIM, MAX_OBJ)

        feat = self.conv(m_in).transpose(1, 2)
        weights = F.softmax(self.attn_weights(feat), dim=1)
        x_attn = torch.sum(weights * feat, dim=1)

        combined = torch.cat([x_attn, s_in], dim=1)
        return combined
    
    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = EXTRACTED_OBS_DIM
        return active_obs_groups, obs_dim


# ═══════════════════════════════════════════════════════════════════════════
#  VecEnv — rsl-rl 介面，內部維護 survivors8 的物理
# ═══════════════════════════════════════════════════════════════════════════
class SurvivorsVecEnv(VecEnv):
    """
    每個 env (world) 是一個獨立的模擬環境，擁有自己的食物與掠食者。
    每個 world 內有 POP_SIZE 個 agent。

    rsl-rl 看到的 num_envs = N_WORLDS * POP_SIZE (展平)。
    PPO Observation TensorDict: "policy" → (total_agents, OBS_DIM)

    資料佈局：
      - agents: (total, ...) 其中 total = N_WORLDS * POP_SIZE
                world_id = agent_idx // POP_SIZE
      - food:   (N_WORLDS, FOOD_SIZE, 2)   — 每個 world 獨立
      - pred:   (N_WORLDS, PREDATOR_SIZE, 2) — 每個 world 獨立
    """

    def __init__(self, num_envs: int = NUM_ENVS, device: str = DEVICE):
        self._W = num_envs       # N_WORLDS
        self._P = POP_SIZE       # agents per world
        total = self._W * self._P

        # ── VecEnv 必要屬性 ──
        self.num_envs           = total
        self.num_actions        = ACTOR_OUT_DIM
        self.max_episode_length = MAX_EPISODE_STEPS
        self.episode_length_buf = torch.zeros(total, dtype=torch.long, device=device)
        self.device             = device
        self.cfg                = {}

        self.screen_size = torch.tensor([SCREEN_W, SCREEN_H], device=device, dtype=torch.float)
        self.bounds = self.screen_size - 1.0
        self.throttle_factor = POP_MAX_SPEED * POP_DAMPING_FACTOR

        # agent→world 映射 (預計算，不變)
        self._world_idx = torch.arange(total, device=device) // self._P  # (total,)

        # 統計
        self.total_episodes  = 0
        self.reward_buf: list[float] = []
        self._ep_reward = torch.zeros(total, dtype=torch.float32, device=device)
        self.energy_avg  = 0.0
        self.rewards_avg = 0.0
        self.eaten   = 0
        self.killed  = 0
        self.collided = 0
        self.starved = 0
        self.frames  = 0

        # 牆壁 (所有 world 共用同一組牆)
        self._build_walls()

        # one-hot labels — 每個 agent 看到的動態物件 = 同 world 的 (POP_SIZE + FOOD_SIZE + PREDATOR_SIZE)
        n_dyn = self._P + FOOD_SIZE + PREDATOR_SIZE
        self.l_wall = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).view(1, 4, 1).expand(
            total, 4, self.wall_A.shape[0])

        label_parts = []
        if self._P > 0:
            label_parts.append(torch.tensor([0,1,0,0], device=device, dtype=torch.float).view(1,4,1).repeat(1,1,self._P))
        if FOOD_SIZE > 0:
            label_parts.append(torch.tensor([0,0,1,0], device=device, dtype=torch.float).view(1,4,1).repeat(1,1,FOOD_SIZE))
        if PREDATOR_SIZE > 0:
            label_parts.append(torch.tensor([0,0,0,1], device=device, dtype=torch.float).view(1,4,1).repeat(1,1,PREDATOR_SIZE))
        if label_parts:
            self.l_all = torch.cat(label_parts, dim=2).expand(total, -1, -1).clone()
        else:
            self.l_all = torch.zeros(total, 4, 0, device=device)

        radius_parts = []
        if self._P > 0:
            radius_parts.append(torch.full((self._P,), POP_RADIUS, device=device))
        if FOOD_SIZE > 0:
            radius_parts.append(torch.full((FOOD_SIZE,), FOOD_RADIUS, device=device))
        if PREDATOR_SIZE > 0:
            radius_parts.append(torch.full((PREDATOR_SIZE,), PREDATOR_RADIUS, device=device))
        self.all_radius = torch.cat(radius_parts) if radius_parts else torch.empty(0, device=device)

        self._reset_all()

    def _build_walls(self):
        pts = torch.tensor([
            [0, 0], [SCREEN_W-1, 0], [SCREEN_W-1, SCREEN_H-1], [0, SCREEN_H-1]
        ], dtype=torch.float32, device=self.device)
        self.wall_A = pts
        self.wall_B = torch.roll(pts, -1, 0)
        self.wall_v = torch.stack([self.wall_A, self.wall_B], dim=1)

    def _reset_all(self):
        N = self.num_envs   # total agents
        W = self._W
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
        self.last_actions = torch.zeros(N, ACTOR_OUT_DIM, device=dev)

        # 每個 world 獨立的食物與掠食者
        if PREDATOR_SIZE > 0:
            self.pred_pos = torch.rand(W, PREDATOR_SIZE, 2, device=dev) * self.screen_size
            self.pred_vel = (torch.rand(W, PREDATOR_SIZE, 2, device=dev) - 0.5) * 3.5
        else:
            self.pred_pos = torch.empty(W, 0, 2, device=dev)
            self.pred_vel = torch.empty(W, 0, 2, device=dev)

        if FOOD_SIZE > 0:
            self.food_pos = self._respawn_food_all()
            self.food_vel = (torch.rand(W, FOOD_SIZE, 2, device=dev) - 0.5) * 3.5
        else:
            self.food_pos = torch.empty(W, 0, 2, device=dev)
            self.food_vel = torch.empty(W, 0, 2, device=dev)

        self.episode_length_buf.zero_()
        self._ep_reward.zero_()
        self.wall_lines = None
        self.color = torch.ones(N, 3, device=dev) * 0.5
        self._last_states = self._get_states()

    def _reset_agents(self, mask):
        if not mask.any():
            return
        n = mask.sum().item()
        dev = self.device
        self.pos[mask] = self._get_safe_pos(mask)
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

    def _get_safe_pos(self, mask):
        """為 mask 中的 agents 生成安全位置（考慮其所屬 world 的障礙物）。"""
        n = mask.sum().item()
        dev = self.device
        # 簡化：隨機位置 + padding
        samples = torch.empty(n, 2, device=dev)
        samples[:, 0].uniform_(RND_POS_PADDING, SCREEN_W - RND_POS_PADDING)
        samples[:, 1].uniform_(RND_POS_PADDING, SCREEN_H - RND_POS_PADDING)
        return samples

    def _respawn_food_all(self):
        """為所有 world 生成食物位置。"""
        if FOOD_SIZE <= 0:
            return torch.empty(self._W, 0, 2, device=self.device)
        pad = RND_POS_PADDING
        pos = torch.rand(self._W, FOOD_SIZE, 2, device=self.device)
        pos[..., 0] = pos[..., 0] * (SCREEN_W - 2*pad) + pad
        pos[..., 1] = pos[..., 1] * (SCREEN_H - 2*pad) + pad
        return pos

    def _respawn_food_at(self, world_mask, food_mask):
        """重新生成指定 world 中被吃掉的食物。"""
        # world_mask: (W,) bool — 哪些 world 有食物被吃
        # food_mask: (W, FOOD_SIZE) bool — 哪些食物被吃
        if not world_mask.any():
            return
        pad = RND_POS_PADDING
        new_pos = torch.rand_like(self.food_pos)
        new_pos[..., 0] = new_pos[..., 0] * (SCREEN_W - 2*pad) + pad
        new_pos[..., 1] = new_pos[..., 1] * (SCREEN_H - 2*pad) + pad
        self.food_pos[food_mask] = new_pos[food_mask]

    def _update_entities(self, pos, vel, radius, min_speed, max_speed, jitter_chance=0.05):
        """pos/vel shape: (W, K, 2)"""
        if pos.shape[1] == 0:
            return pos, vel
        W, K, _ = pos.shape
        change_mask = (torch.rand(W, K, 1, device=self.device) < jitter_chance)
        vel = vel + (torch.rand_like(vel) - 0.5) * (1.8 * change_mask)
        speeds = torch.norm(vel, dim=2, keepdim=True)
        new_speeds = torch.clamp(speeds, min_speed, max_speed)
        vel = vel * (new_speeds / (speeds + 1e-6))
        pos = pos + vel
        min_bound = radius
        max_bound_x = SCREEN_W - 1.0 - radius
        max_bound_y = SCREEN_H - 1.0 - radius
        # 反彈
        vel[..., 0][(pos[..., 0] < min_bound) | (pos[..., 0] > max_bound_x)] *= -1
        vel[..., 1][(pos[..., 1] < min_bound) | (pos[..., 1] > max_bound_y)] *= -1
        pos[..., 0].clamp_(min_bound, max_bound_x)
        pos[..., 1].clamp_(min_bound, max_bound_y)
        return pos, vel

    # ── 觀測構建 ──
    def _get_states(self):
        N = self.num_envs   # total agents
        W = self._W
        P = self._P
        dev = self.device
        angel = self.angle.view(N, 1)
        p = self.pos.unsqueeze(1)  # (N, 1, 2)

        # ── 牆壁 (對所有 agent 相同) ──
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

        # ── 動態物件 (per-world) ──
        # 組合每個 world 的 (agents, food, predators)
        # agents_by_world: (W, P, 2)
        agents_bw = self.pos.view(W, P, 2)
        dyn_parts = [agents_bw]
        if FOOD_SIZE > 0:
            dyn_parts.append(self.food_pos)     # (W, FOOD_SIZE, 2)
        if PREDATOR_SIZE > 0:
            dyn_parts.append(self.pred_pos)     # (W, PRED_SIZE, 2)
        dyn_bw = torch.cat(dyn_parts, dim=1)   # (W, P+F+Pr, 2)
        n_dyn = dyn_bw.shape[1]

        # 展開 agent 視角: (N, n_dyn, 2)
        # 每個 agent 看到的是自己所屬 world 的動態物件
        dyn_expanded = dyn_bw[self._world_idx]   # (N, n_dyn, 2)

        diff_d = dyn_expanded - p                # (N, n_dyn, 2)
        dist_d = torch.norm(diff_d, dim=2)       # (N, n_dyn)

        # 排除自己：agent i 在 world w 中是第 (i % P) 個
        local_idx = torch.arange(N, device=dev) % P  # (N,)
        dist_d[torch.arange(N, device=dev), local_idx] = 1e6

        abs_ang_d = torch.atan2(diff_d[..., 1], diff_d[..., 0])
        rel_ang_d = abs_ang_d - angel
        dist_val = (dist_d - POP_RADIUS - self.all_radius) / POP_PERCEPTION_RADIUS
        dyna_mask = ((dist_d - POP_RADIUS - self.all_radius) < POP_PERCEPTION_RADIUS).float()
        # 排除自己
        dyna_mask[torch.arange(N, device=dev), local_idx] = 0.0
        # 隊友感知
        if not POP_PERCEPT_TEAM:
            dyna_mask[:, :P] = 0.0

        # 能量特徵
        energy_bw = (self.energy / MAX_ENERGY).view(W, P)  # (W, P)
        energy_norm = torch.zeros(N, n_dyn, device=dev)
        # 同 world agents 的能量
        energy_norm[:, :P] = energy_bw[self._world_idx]
        if PREDATOR_SIZE > 0:
            energy_norm[:, -PREDATOR_SIZE:] = 1.0

        phys_d = torch.stack([
            torch.cos(rel_ang_d), torch.sin(rel_ang_d),
            dist_val, energy_norm
        ], dim=1)
        dynamic_in = torch.cat([phys_d, self.l_all], dim=1)

        # ── 合併取 top-k ──
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

        # ── 自身狀態 ──
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

        en_ratio = (self.energy / MAX_ENERGY).view(-1, 1)
        self.color = torch.cat([en_ratio, 0.5 * en_ratio, 1.0 - en_ratio], dim=1)

        return mixed_in, self_in

    def _build_obs(self) -> TensorDict:
        mixed_in, self_in = self._last_states
        obs = torch.cat([mixed_in.reshape(self.num_envs, -1), self_in], dim=1)
        return TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)

    def get_observations(self) -> TensorDict:
        return self._build_obs()

    def step(self, actions: torch.Tensor):
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        actions = actions.clamp(-1, 1)

        N = self.num_envs
        W = self._W
        P = self._P
        dev = self.device
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

        # ── 掠食者 (per-world) ──
        if PREDATOR_SIZE > 0:
            self.pred_pos, self.pred_vel = self._update_entities(
                self.pred_pos, self.pred_vel, PREDATOR_RADIUS,
                min_speed=PREDATOR_MIN_SPEED, max_speed=PREDATOR_MAX_SPEED)
            # (W, P, 2) vs (W, PRED, 2) → (W, P, PRED)
            agents_bw = self.pos.view(W, P, 2)
            dist_pred = torch.cdist(agents_bw, self.pred_pos)  # (W, P, PRED)
            killed_bw = (dist_pred < POP_RADIUS + PREDATOR_RADIUS).any(dim=2)  # (W, P)
            alive_bw = self.alive.view(W, P)
            killed = (killed_bw & alive_bw).view(N)
            rewards[killed] += KILLED_REWARD
            self._kill(killed)
            pred_nearby_bw = (dist_pred - POP_RADIUS - PREDATOR_RADIUS < POP_ALERT_RADIUS).any(dim=2)
            pred_nearby = (pred_nearby_bw & alive_bw).view(N)
            rewards[pred_nearby] += PREDATOR_NEARBY_REWARD
        else:
            killed = torch.zeros(N, dtype=torch.bool, device=dev)

        # ── 撞牆 ──
        p = self.pos.unsqueeze(1)
        wa = self.wall_A.unsqueeze(0)
        wb = self.wall_B.unsqueeze(0)
        ab = wb - wa
        ap = p - wa
        t = (torch.sum(ap * ab, dim=-1) / (torch.sum(ab * ab, dim=-1) + 1e-6)).clamp(0, 1)
        wall_closest_points = wa + t.unsqueeze(-1) * ab
        dist_to_walls = torch.norm(self.pos.unsqueeze(1) - wall_closest_points, dim=-1)
        collided = (dist_to_walls < POP_RADIUS).any(dim=1) & self.alive
        rewards[collided] += COLLIDED_REWARD
        self._kill(collided)

        # ── 食物 (per-world) ──
        if FOOD_SIZE > 0:
            agents_bw = self.pos.view(W, P, 2)
            dist_food = torch.cdist(agents_bw, self.food_pos)  # (W, P, FOOD)
            alive_bw = self.alive.view(W, P)
            hits_food = (dist_food < POP_RADIUS + FOOD_RADIUS) & alive_bw.unsqueeze(2)  # (W, P, FOOD)
            if hits_food.any():
                # 每個食物只能被同 world 中最近的 agent 吃到
                masked_dist = torch.where(hits_food, dist_food, torch.tensor(float('inf'), device=dev))
                min_dists, closest_a_local = torch.min(masked_dist, dim=1)  # (W, FOOD)
                valid = min_dists != float('inf')  # (W, FOOD)
                if valid.any():
                    w_idx, f_idx = torch.where(valid)
                    a_local = closest_a_local[w_idx, f_idx]
                    a_global = w_idx * P + a_local
                    n_eaten = len(a_global)
                    rewards.index_add_(0, a_global, torch.full((n_eaten,), FOOD_REWARD, device=dev))
                    self.energy.index_add_(0, a_global, torch.full((n_eaten,), FOOD_ENERGY, device=dev))
                    self.energy.clamp_(max=MAX_ENERGY)
                    # 重生食物
                    food_eaten_mask = valid  # (W, FOOD)
                    self._respawn_food_at(valid.any(dim=1), food_eaten_mask)
                    self.eaten += n_eaten

        # ── 能量消耗 ──
        if ENERGY_DECAY > 0:
            static_cost = 0.2 * ENERGY_DECAY
            dynamic_cost = 0.8 * ENERGY_DECAY * throttle_vals.pow(2)
            self.energy -= (static_cost + dynamic_cost) * self.alive.float()
            starved = (self.energy <= 0) & self.alive
            rewards[starved] += STARVED_REWARD
            self._kill(starved)
        else:
            starved = torch.zeros(N, dtype=torch.bool, device=dev)

        # ── 移動獎勵 ──
        self.forward_speed = torch.linalg.vecdot(self.vel, pop_vecs)
        vel_mag = torch.norm(self.vel, dim=1, keepdim=True) + 1e-6
        vel_purity = torch.relu(torch.linalg.vecdot(self.vel / vel_mag, pop_vecs))
        move_reward = MOVE_REWARD * 2.5 * (torch.relu(self.forward_speed) / POP_MAX_SPEED) * vel_purity
        move_reward = move_reward * vel_purity.pow(4)
        action_diff = actions - self.last_actions
        smooth_penalty = MOVE_REWARD * 0.2 * torch.mean(action_diff.pow(2), dim=1)
        abs_throttle = torch.abs(throttle_vals)
        throttle_penalty = MOVE_REWARD * 0.2 * torch.relu(abs_throttle - 0.75)
        steer_penalty = MOVE_REWARD * 0.3 * (steer_vals * throttle_vals).pow(2)
        purity_gap = torch.relu(0.95 - vel_purity)
        spinning_bonus = MOVE_REWARD * 5.0 * purity_gap.pow(2) * torch.abs(steer_vals)
        spinning_penalty = MOVE_REWARD * 3.0 * steer_vals.pow(2) * (1.0 - vel_purity)
        omega_yaw = (self.angle - self.last_angle) / POP_MAX_STEER
        yaw_penalty = MOVE_REWARD * 20.0 * torch.abs(omega_yaw)
        rewards += (move_reward - smooth_penalty - throttle_penalty
                    - steer_penalty - spinning_penalty - spinning_bonus - yaw_penalty) * self.alive.float()

        # ── 近牆痛覺 ──
        wall_min_dist, closest_wall_idx = torch.min(dist_to_walls, dim=1)
        wall_dist_ratio = (1.0 - (wall_min_dist - POP_RADIUS) / POP_ALERT_RADIUS).clamp(0.0, 1.0)
        wall_nearby_mask = (wall_dist_ratio > 0) & self.alive
        if wall_nearby_mask.any():
            wall_contacts = wall_closest_points[torch.arange(N, device=dev), closest_wall_idx]
            self.wall_lines = torch.stack([self.pos[wall_nearby_mask], wall_contacts[wall_nearby_mask]], dim=1)
            rewards[wall_nearby_mask] += WALL_NEARBY_REWARD
        else:
            self.wall_lines = None

        dead_mask = killed | collided | starved

        self._last_states = self._get_states()

        # ── 復活 ──
        self.respawn_timer[~self.alive] -= 1
        ready = ~self.alive & (self.respawn_timer <= 0)
        if ready.any():
            self._reset_agents(ready)

        self.last_actions = actions.detach().clone()
        self.episode_length_buf += 1

        # ── dones ──
        time_out = (self.episode_length_buf >= MAX_EPISODE_STEPS) & self.alive
        dones = torch.zeros(N, device=dev)
        dones[dead_mask] = 1.0
        time_outs = torch.zeros(N, device=dev)
        time_outs[time_out] = 1.0
        dones[time_out] = 1.0

        self._ep_reward += rewards
        done_all = dones.bool()
        if done_all.any():
            for i in torch.where(done_all)[0].tolist():
                self.total_episodes += 1
                self.reward_buf.append(float(self._ep_reward[i]))
                if len(self.reward_buf) > 200:
                    self.reward_buf.pop(0)
                self._ep_reward[i] = 0.0
                self.episode_length_buf[i] = 0

        self.energy_avg = self.energy_avg * 0.99 + (self.energy.sum().item() / N) * 0.01
        self.rewards_avg = self.rewards_avg * 0.99 + (rewards.sum().item() / N) * 0.01
        self.killed += killed.sum().item()
        self.collided += collided.sum().item()
        self.starved += starved.sum().item()
        self.rewards_last = rewards
        self.frames += 1

        return self._build_obs(), rewards, dones, {"time_outs": time_outs}

    def _kill(self, mask):
        self.alive[mask] = False
        n = mask.sum().item()
        if n > 0:
            self.respawn_timer[mask] = 1 if self._P == 1 else torch.randint(30, 180, (n,), device=self.device)


# ═══════════════════════════════════════════════════════════════════════════
#  ModernGL Renderer
# ═══════════════════════════════════════════════════════════════════════════
class GLRenderer:
    def __init__(self, ctx, fbo_texture, w, h):
        import moderngl
        self.ctx = ctx
        self.fbo_texture = fbo_texture
        self.w, self.h = w, h

        self.circle_prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert; in vec4 in_pos_rad; in vec3 in_color;
                out vec3 v_color; out vec2 v_dist;
                uniform vec2 res;
                void main() {
                    vec2 p = (in_pos_rad.xy / res) * 2.0 - 1.0; p.y *= -1.0;
                    vec2 scale = (in_pos_rad.z / res) * 2.0;
                    gl_Position = vec4(p + in_vert * scale, 0.0, 1.0);
                    v_color = in_color; v_dist = in_vert;
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color; in vec2 v_dist; out vec4 f;
                uniform bool is_hollow; uniform float thickness;
                void main() {
                    float d = length(v_dist);
                    if (d > 1.0) discard;
                    if (is_hollow && d < (1.0 - thickness)) discard;
                    f = vec4(v_color, 1.0);
                }
            """
        )
        self.circle_prog['res'].value = (w, h)
        n_seg = 32
        verts = np.zeros((n_seg + 2, 2), dtype='f4')
        verts[0] = [0.0, 0.0]
        ang = np.linspace(0, 2 * np.pi, n_seg + 1, endpoint=True)
        verts[1:, 0] = np.cos(ang)
        verts[1:, 1] = np.sin(ang)
        self.vbo_circle_temp = ctx.buffer(verts)
        self.vbo_circle_inst = ctx.buffer(reserve=10000 * 7 * 4)
        self.vao_circle = ctx.vertex_array(self.circle_prog, [
            (self.vbo_circle_temp, '2f', 'in_vert'),
            (self.vbo_circle_inst, '4f 3f /i', 'in_pos_rad', 'in_color')
        ])

        self.line_prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert; in vec3 in_color; out vec3 v_color;
                uniform vec2 res;
                void main() {
                    vec2 p = ((in_vert + vec2(0.5)) / res) * 2.0 - 1.0; p.y *= -1.0;
                    gl_Position = vec4(p, 0.0, 1.0); v_color = in_color;
                }
            """,
            fragment_shader="#version 330\nin vec3 v_color; out vec4 f; void main() { f = vec4(v_color, 1.0); }"
        )
        self.line_prog['res'].value = (w, h)
        self.vbo_line = ctx.buffer(reserve=20000 * 5 * 4)
        self.vao_line = ctx.vertex_array(self.line_prog, [(self.vbo_line, '2f 3f', 'in_vert', 'in_color')])

        self.text_texture = ctx.texture((w, h), 4)
        self.text_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.text_prog = ctx.program(
            vertex_shader="#version 330\nin vec2 in_vert; in vec2 in_texcoord; out vec2 v_tc; void main() { gl_Position = vec4(in_vert, 0, 1); v_tc = in_texcoord; }",
            fragment_shader="#version 330\nuniform sampler2D tex; in vec2 v_tc; out vec4 f; void main() { f = texture(tex, v_tc); }"
        )
        quad = np.array([-1,1,0,0, -1,-1,0,1, 1,1,1,0, -1,-1,0,1, 1,-1,1,1, 1,1,1,0], dtype='f4')
        self.vbo_text = ctx.buffer(quad)
        self.vao_text = ctx.vertex_array(self.text_prog, [(self.vbo_text, '2f 2f', 'in_vert', 'in_texcoord')])

        self.screen_prog = ctx.program(
            vertex_shader="#version 330\nin vec2 in_vert; out vec2 uv; void main() { uv = (in_vert+1.0)*0.5; gl_Position = vec4(in_vert,0,1); }",
            fragment_shader="#version 330\nuniform sampler2D tex; in vec2 uv; out vec4 f; void main() { f = texture(tex, uv); }"
        )
        self.vbo_screen = ctx.buffer(np.array([-1,-1,1,-1,-1,1,1,1], dtype='f4'))
        self.vao_screen = ctx.simple_vertex_array(self.screen_prog, self.vbo_screen, 'in_vert')

    def draw_circles(self, pos, rad, color):
        if pos.shape[0] == 0: return
        data = torch.cat([pos, rad, rad, color], dim=1).cpu().numpy().astype('f4')
        self.vbo_circle_inst.write(data)
        import moderngl
        self.vao_circle.render(moderngl.TRIANGLE_FAN, instances=pos.shape[0])

    def draw_lines(self, start, end, color):
        if start.shape[0] == 0: return
        import moderngl
        N = start.shape[0]
        data = torch.empty((N, 2, 5), device=start.device)
        data[:, 0, :2] = start; data[:, 0, 2:] = color
        data[:, 1, :2] = end;   data[:, 1, 2:] = color
        self.vbo_line.write(data.cpu().numpy().astype('f4'))
        self.vao_line.render(moderngl.LINES, vertices=N*2)

    def draw_text(self, surface):
        import pygame, moderngl
        self.text_texture.write(pygame.image.tostring(surface, 'RGBA'))
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.text_texture.use(0); self.vao_text.render()

    def blit_to_screen(self, win_w, win_h):
        import moderngl
        self.ctx.screen.use(); self.ctx.viewport = (0, 0, win_w, win_h)
        self.ctx.disable(moderngl.DEPTH_TEST); self.ctx.disable(moderngl.CULL_FACE); self.ctx.disable(moderngl.BLEND)
        self.fbo_texture.use(0); self.vao_screen.render(moderngl.TRIANGLE_STRIP)


# ═══════════════════════════════════════════════════════════════════════════
#  Pygame Renderer
# ═══════════════════════════════════════════════════════════════════════════
class Renderer:
    def __init__(self):
        import pygame, moderngl
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
        pygame.display.set_caption(CAPTION)
        self.ctx = moderngl.create_context()
        fbo_tex = self.ctx.texture((SCREEN_W, SCREEN_H), 3)
        fbo_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.fbo = self.ctx.framebuffer(color_attachments=[fbo_tex])
        self.gl = GLRenderer(self.ctx, fbo_tex, SCREEN_W, SCREEN_H)
        self.ui_surface = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
        self.font = pygame.font.SysFont("Consolas", 14)
        self.big_font = pygame.font.SysFont("Consolas", 18, bold=True)
        self._clk = pygame.time.Clock()
        self.draw_units = True
        self.draw_label = True
        self.verbose = 0

    def handle_events(self) -> dict:
        import pygame
        ev = {"quit": False, "pause": False, "reset": False}
        for e in pygame.event.get():
            if e.type == pygame.QUIT: ev["quit"] = True
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE): ev["quit"] = True
                elif e.key == pygame.K_SPACE: ev["pause"] = True
                elif e.key == pygame.K_r: ev["reset"] = True
                elif e.key == pygame.K_u: self.draw_units = not self.draw_units
                elif e.key == pygame.K_l: self.draw_label = not self.draw_label
                elif e.key == pygame.K_v: self.verbose = (self.verbose + 1) % 4
        return ev

    def draw(self, env: SurvivorsVecEnv, stats: dict):
        import pygame
        self.fbo.use()
        self.ctx.clear(0.08, 0.08, 0.08)
        self.ui_surface.fill((0, 0, 0, 0))
        dev = env.device

        # ── Snapshot (只渲染 world 0 的 agents) ──
        P = env._P
        snap_pos = env.pos[:P].clone()
        snap_alive = env.alive[:P].clone()
        snap_energy = env.energy[:P].clone()
        snap_color = env.color[:P].clone()
        snap_vel = env.vel[:P].clone()
        snap_fwd = env.forward_speed[:P].clone()
        snap_angle = env.angle[:P].clone()
        snap_act = env.last_actions[:P].clone()
        snap_food = env.food_pos[0].clone() if FOOD_SIZE > 0 else None       # (FOOD_SIZE, 2)
        snap_pred = env.pred_pos[0].clone() if PREDATOR_SIZE > 0 else None   # (PRED_SIZE, 2)
        snap_wl = env.wall_lines.clone() if env.wall_lines is not None else None

        if self.draw_units:
            alive = snap_alive; dead = ~alive
            # 牆
            nw = env.wall_v.size(0)
            self.gl.draw_lines(env.wall_v[:,0,:], env.wall_v[:,1,:],
                               torch.tensor([[.5,0,0]], device=dev).expand(nw,-1))
            # 食物
            if snap_food is not None and len(snap_food) > 0:
                nf = len(snap_food)
                self.gl.circle_prog['is_hollow'].value = False
                self.gl.draw_circles(snap_food, torch.full((nf,1), FOOD_RADIUS, device=dev),
                                     torch.tensor([[0,1,.47]], device=dev).expand(nf,-1))
            # 掠食者
            if snap_pred is not None and len(snap_pred) > 0:
                np_ = len(snap_pred)
                pc = torch.tensor([[1,0,0]], device=dev).expand(np_,-1)
                self.gl.circle_prog['is_hollow'].value = True
                self.gl.circle_prog['thickness'].value = 0.05
                self.gl.draw_circles(snap_pred, torch.full((np_,1), PREDATOR_RADIUS, device=dev), pc*0.5)
                self.gl.circle_prog['is_hollow'].value = False
                self.gl.draw_circles(snap_pred, torch.full((np_,1), PREDATOR_RADIUS*0.25, device=dev), pc)
            # 死亡
            if dead.any():
                dp = snap_pos[dead]
                dc = torch.where((snap_energy[dead]<=0).unsqueeze(1),
                                 torch.tensor([.23,.23,.23], device=dev),
                                 torch.tensor([.47,0,0], device=dev))
                self.gl.circle_prog['is_hollow'].value = False
                self.gl.draw_circles(dp, torch.full((dp.shape[0],1),3.0, device=dev), dc)
            # 存活
            if alive.any():
                ap = snap_pos[alive]
                ac = snap_color[alive]
                er = (snap_energy[alive]/MAX_ENERGY).unsqueeze(1)
                ar = POP_RADIUS + 4*er
                self.gl.circle_prog['is_hollow'].value = False
                self.gl.draw_circles(ap, ar, ac)
                # 速度線
                spd = snap_fwd[alive]; vel = snap_vel[alive]
                ms = torch.abs(spd) > 0.1
                if ms.any():
                    ls = ap[ms]; le = ls + vel[ms]*(torch.abs(spd[ms])*2.5).unsqueeze(1)
                    self.gl.draw_lines(ls, le, torch.tensor([[0,.75,1]], device=dev).expand(ls.shape[0],-1))
                # 油門線
                thr = snap_act[alive, 1]
                mt = thr != 0
                if mt.any():
                    ts = ap[mt]
                    tv = thr[mt]
                    ta = torch.abs(tv)
                    tang = snap_angle[alive][mt]
                    fwd = tv > 0
                    fa = torch.where(fwd, tang, tang+math.pi)
                    fl = torch.where(fwd, 20*ta, 20*ta/3)
                    td = torch.stack([torch.cos(fa), torch.sin(fa)], dim=1)
                    te = ts + td*fl.unsqueeze(1)
                    tc = torch.zeros(ts.size(0), 3, device=dev)
                    tc[~fwd] = torch.tensor([1.0, 1.0, 0.0], device=dev)
                    tc[fwd&(ta<=.5)] = torch.tensor([0.0, 1.0, 0.0], device=dev)
                    tc[fwd&(ta>.5)&(ta<=.875)] = torch.tensor([1.0, 1.0, 1.0], device=dev)
                    tc[fwd&(ta>.875)] = torch.tensor([1.0, 0.0, 0.0], device=dev)
                    self.gl.draw_lines(ts, te, tc)
            # 牆壁感應線
            if self.verbose >= 2 and snap_wl is not None:
                nwl = snap_wl.size(0)
                self.gl.draw_lines(snap_wl[:,0,:], snap_wl[:,1,:],
                                   torch.tensor([[1,.2,.2]], device=dev).expand(nwl,-1))
                self.gl.circle_prog['is_hollow'].value = False
                self.gl.draw_circles(snap_wl[:,1,:], torch.full((nwl,1),3.0, device=dev),
                                     torch.tensor([[1,1,0]], device=dev).expand(nwl,-1))

        # ── UI ──
        TH = {"l":(180,180,180), "p":(100,220,180), "s":(120,255,120), "f":(255,120,120)}
        if self.draw_label:
            it = stats.get("iteration", 0)
            ep = env.total_episodes
            rew = float(np.mean(env.reward_buf)) if env.reward_buf else 0.0
            labels = [
                ("Iteration:", f"{it:,}", TH["p"], True),
                ("Episodes:", f"{ep:,}", TH["p"], True),
                ("Frames:", f"{env.frames:,}", TH["p"], False),
                ("Energy:", f"{env.energy_avg:.0f}", TH["s"] if env.energy_avg>60 else (255,0,0), False),
                ("Rewards:", f"{env.rewards_avg:.4f}", TH["s"] if env.rewards_avg>0 else (255,0,0), False),
                ("Mean Ep Rew:", f"{rew:.3f}", TH["p"], False),
                ("Eaten:", f"{env.eaten:,}", TH["s"], False),
                ("Killed:", f"{env.killed:,}", TH["f"], False),
                ("Collided:", f"{env.collided:,}", TH["f"], False),
                ("Starved:", f"{env.starved:,}", TH["f"], False),
                ("Alive:", f"{int(snap_alive.sum())}/{env.num_envs}", TH["s"], False),
            ]
            if stats.get("paused"):
                labels.append(("[PAUSED]","", (255,255,0), True))
            y = 10
            for txt, val, vc, bold in labels:
                ft = self.big_font if bold else self.font
                ls = ft.render(txt, True, (180,180,180)); vs = ft.render(val, True, vc)
                self.ui_surface.blit(ls, (10, y))
                self.ui_surface.blit(vs, vs.get_rect(topright=(200, y)))
                y += max(ls.get_height(), vs.get_height()) + 4

        hs = self.font.render("SPACE=Pause R=Reset U=Units L=Labels V=Verbose Q=Quit", True, (100, 100, 140))
        self.ui_surface.blit(hs, (10, SCREEN_H-20))
        self.gl.draw_text(self.ui_surface)
        self.gl.blit_to_screen(*pygame.display.get_window_size())
        pygame.display.flip()
        self._clk.tick(FPS_RENDER)

    def quit(self):
        import pygame;
        pygame.quit()


# ═══════════════════════════════════════════════════════════════════════════
#  Training thread
# ═══════════════════════════════════════════════════════════════════════════
class TrainState:
    def __init__(self):
        self.iteration = 0
        self.paused = False
        self.stop = False
        self.reset_req = False
        self.lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════
#  Config — rsl-rl PPO (標準 MLPModel，吃特徵提取後的觀測)
# ═══════════════════════════════════════════════════════════════════════════
def make_train_cfg():
    return {
        "seed": 42,
        "num_steps_per_env": 24,
        "save_interval": 1000,

        "obs_groups": {
            "actor":  ["policy"],
            "critic": ["policy"],
        },

        "algorithm": {
            "class_name":             "PPO",
            "num_learning_epochs":    5,
            "num_mini_batches":       4,
            "gamma":                  0.97,
            "lam":                    0.95,
        },

        "actor": {
            "class_name":  "survivors10.SurvivorsCustomModel",
            "hidden_dims": [HIDDEN_FC_DIM, HIDDEN_FC_DIM],
            "activation":  "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
            },
        },
        "critic": {
            "class_name":  "survivors10.SurvivorsCustomModel",
            "hidden_dims": [HIDDEN_FC_DIM, HIDDEN_FC_DIM],
            "activation":  "elu",
        },
    }


def train_loop(runner: OnPolicyRunner, max_iter: int):
    runner.learn(max_iter)


def get_latest_checkpoint(path):
    import glob
    import re
    files = glob.glob(os.path.join(path, "model_*.pt"))
    if not files:
        return None
    # 提取數字並找出最大值
    iters = [int(re.findall(r'model_(\d+).pt', f)[0]) for f in files]
    return max(iters)


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description=CAPTION)
    parser.add_argument("-e", "--epoch", type=str, default=None, help="一個訓練週期")
    parser.add_argument("-s", "--steps", type=int, default=float('inf'), help="達到此步數時退出")
    parser.add_argument("-r", "--record", action="store_true", default=False, help="啟動即開始錄影")
    parser.add_argument("--demo", action="store_true", default=False, help="模型性能展示")
    parser.add_argument("--frames", type=int, default=float('inf'), help="模型性能展示幀數")
    parser.add_argument("--headless", action="store_true", default=False, help="無頭模式")
    args = parser.parse_args()

    env = SurvivorsVecEnv(num_envs=NUM_ENVS, device=DEVICE)
    train_cfg = make_train_cfg()

    print("=" * 60)
    print(f"  {CAPTION}")
    print(f"  Stage: {STAGE}  Worlds: {NUM_ENVS}  PopPerWorld: {POP_SIZE}  TotalAgents: {env.num_envs}")
    print(f"  Actions: {ACTOR_OUT_DIM}  RawObs: {OBS_DIM}  ExtractedObs: {EXTRACTED_OBS_DIM}")
    print(f"  Food: {FOOD_SIZE}  Predators: {PREDATOR_SIZE}  EnergyDecay: {ENERGY_DECAY:.4f}")
    print(f"  MaxSteps: {MAX_EPISODE_STEPS}  EstSteps: {EST_STEPS:.0f}")
    print("=" * 60)

    runner = OnPolicyRunner(env, train_cfg, log_dir=LOG_PATH, device=DEVICE)
    print("Runner built. Training starts!")

    latest_idx = get_latest_checkpoint(LOG_PATH)
    if latest_idx is not None:
        file_path = os.path.join(LOG_PATH, f"model_{latest_idx}.pt")
        runner.load(file_path)
        print(f"[Load] {file_path} iteration={runner.current_learning_iteration:,}")

    state = TrainState()
    max_iter = 1000

    if args.headless:
        try:
            runner.learn(max_iter)
        except KeyboardInterrupt:
            pass
        finally:
            file_path = os.path.join(LOG_PATH, f"model_{runner.current_learning_iteration}.pt")
            runner.save(file_path)
            print(f"[Save] {file_path} iteration={runner.current_learning_iteration}")
    else:
        renderer = Renderer()
        t = threading.Thread(
            target=train_loop,
            args=(runner, max_iter),
            daemon=True,
        )
        t.start()

        stats = {}
        try:
            while t.is_alive():
                with state.lock:
                    stats["iteration"] = runner.current_learning_iteration
                renderer.draw(env, stats)
        except KeyboardInterrupt:
            pass
        finally:
            file_path = os.path.join(LOG_PATH, f"model_{runner.current_learning_iteration}.pt")
            runner.save(file_path)
            print(f"[Save] {file_path} iteration={runner.current_learning_iteration}")

        t.join(timeout=3.0)
        print("\nTraining finished, closing.")
        renderer.quit()


if __name__ == "__main__":
    main()