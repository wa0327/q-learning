from pathlib import Path
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit
import numpy as np
import pygame
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
import argparse

script_name = Path(__file__).stem
BASE_PATH = f"weights/{script_name}"
SAVE_PATH = f"{BASE_PATH}/{script_name}"
FINAL_PATH = f"{SAVE_PATH}_final"
FINAL_FULL_PATH = f"{FINAL_PATH}.zip"
INTERRUPTED_PATH = f"{SAVE_PATH}_interrupted"
LOG_PATH = f"logs/{script_name}"
CAPTION = "Vectra: Apex Protocol"
os.makedirs(BASE_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)

# ==========================================
# 參數設定
# ==========================================
STAGE = 2 # 1:學走路 2:學吃飯 3:學初級躲避 4:學高級躲避 5:煉獄模式
SCREEN_W, SCREEN_H = 1280, 800
POP_MAX_SPEED = 3.5
POP_RADIUS = 4
POP_DAMPING_FACTOR = 0.25
POP_BACKWARD_FACTOR = 0.33
POP_MAX_STEER = math.radians(15)
POP_PERCEPTION_RADIUS = 200
# 生存難度系數
STAGE_SURVIVAL_MULTIPLIER = 100 if STAGE < 2 else 5 if STAGE < 3 else 2 if STAGE < 4 else 1
EST_STEPS = math.sqrt(SCREEN_W**2 + SCREEN_H**2) * 0.715 / POP_MAX_SPEED * STAGE_SURVIVAL_MULTIPLIER # 根據速度與「生存者需跑完對角線 71.5%」的目標，推算出所需步數

# 環境物件設定
FOOD_SIZE = 5 if STAGE < 2 else 50 if STAGE < 3 else 25 if STAGE < 5 else 5
FOOD_RADIUS = 3
MAX_ENERGY = 100.0
FOOD_ENERGY = 25.0
ENERGY_DECAY = FOOD_ENERGY / EST_STEPS
MOVE_FOOD = False if STAGE < 4 else True
PREDATOR_SIZE = 0 if STAGE < 3 else 5 if STAGE < 4 else 8 if STAGE < 5 else 16
PREDATOR_RADIUS = 20.0
PREDATOR_MIN_SPEED = 1.5
PREDATOR_MAX_SPEED = 2.5 if STAGE < 4 else 3.0 if STAGE < 5 else 3.2
POP_ALERT_RADIUS = max(POP_MAX_SPEED, (PREDATOR_MAX_SPEED / POP_MAX_SPEED) ** 2 * 20.58)
RND_POS_PADDING = POP_RADIUS + POP_ALERT_RADIUS  # 隨機取位邊距
WALL_SIZE = 0

# 獎懲設定
FOOD_REWARD = 50.0    # 吃到食物
KILLED_REWARD = -75   # 被殺
COLLIDED_REWARD = -60 # 撞死
STARVED_REWARD = -70  # 餓死
MOVE_REWARD_FACTOR = 2.0 if STAGE < 2 else 0.35 # [移動總獎勵]與[最大獎勵]的佔比，數值越低對模型越驅策
MOVE_REWARD = FOOD_REWARD * MOVE_REWARD_FACTOR / EST_STEPS # 移動基礎獎勵
TIME_PENALTY_FACTOR = 0 if STAGE == 1 else 0.15 # [餓死前的總懲罰]與[餓死懲罰]的佔比，數值越低對模型來說越划算
STEP_REWARD = STARVED_REWARD * TIME_PENALTY_FACTOR / EST_STEPS # 每步時間獎懲
WALL_NEARBY_REWARD = COLLIDED_REWARD * 0.25     # 近牆懲罰, 提早預警危險
PREDATOR_NEARBY_REWARD = KILLED_REWARD * 0.25   # 近敵懲罰, 提早預警危險

# 模型核心參數
FEAT_IN_DIM = 8         # 每個物件特微 [cos, sin, dist, energy, is_wall, is_food, is_team, is_pred]
STATE_IN_DIM = 7        # 自身狀態 [前向速度, 側向速度, 前次轉向指令, 能量, dx_ego, dy_ego, omega_yaw]
ACTOR_OUT_DIM = 2       # Actor 輸出層數，輸出動作 [轉向, 油門]
HIDDEN_FEAT_DIM = 64    # 特徵提取層 (Conv1d) 的輸出維度(環境特徵)
HIDDEN_ATTN_DIM = 32    # Attention 內部的隱藏層維度
HIDDEN_FC_DIM = 256     # 後段全連接層 (MLP) 的主要維度(決策輸出)
MAX_OBJ = 25            # 視野內最近距離中最多的環境物件數量

# ==========================================
# 輔助數學函數 (NumPy 版本)
# ==========================================
def point_to_line_segment(p, a, b):
    ab = b - a
    ap = p - a
    ab_norm_sq = np.dot(ab, ab)
    if ab_norm_sq == 0:
        return a, np.linalg.norm(p - a)
    t = np.dot(ap, ab) / ab_norm_sq
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    dist = np.linalg.norm(closest - p)
    return closest, dist

# ==========================================
# Gymnasium 環境定義
# ==========================================
class SurvivorsEnv(gym.Env):
    """
    自訂的 Gymnasium 環境，遵循標準介面: reset, step, render, close
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, render_fps=120):
        super().__init__()
        self.render_mode = render_mode
        self.render_fps = render_fps
        
        # 動作空間：[轉向, 油門]，範圍 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(ACTOR_OUT_DIM,), dtype=np.float32)
        
        # 使用 Dict Space 分離環境物件特徵與自身狀態
        self.observation_space = spaces.Dict({
            "env_features": spaces.Box(low=-np.inf, high=np.inf, shape=(FEAT_IN_DIM * MAX_OBJ,), dtype=np.float32),
            "self_features": spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_IN_DIM,), dtype=np.float32)
        })
        
        self.move_food = MOVE_FOOD
        self.move_predator = True
        self.verbose = 0

        # 牆壁定義 (對應原本的 add_wall_group)
        self.walls = [
            (np.array([0, 0]), np.array([SCREEN_W-1, 0])),
            (np.array([SCREEN_W-1, 0]), np.array([SCREEN_W-1, SCREEN_H-1])),
            (np.array([SCREEN_W-1, SCREEN_H-1]), np.array([0, SCREEN_H-1])),
            (np.array([0, SCREEN_H-1]), np.array([0, 0]))
        ]
        
        # 渲染相關
        self.window = None
        self.clock = None
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 初始化代理狀態
        self.pos = np.random.rand(2) * np.array([SCREEN_W, SCREEN_H])
        self.last_pos = self.pos.copy()
        self.angle = np.random.rand() * 2 * np.pi
        self.last_angle = self.angle
        self.vel = np.zeros(2)
        self.forward_speed = 0.0
        self.energy = MAX_ENERGY
        self.last_action = np.zeros(2)
        
        # 初始化環境物件
        self.food_pos = np.random.rand(FOOD_SIZE, 2) * np.array([SCREEN_W, SCREEN_H])
        self.food_vel = (np.random.rand(FOOD_SIZE, 2) - 0.5) * 3.5
        self.pred_pos = np.random.rand(PREDATOR_SIZE, 2) * np.array([SCREEN_W, SCREEN_H])
        self.pred_vel = (np.random.rand(PREDATOR_SIZE, 2) - 0.5) * 3.5
        
        return self._get_obs(), {}

    def step(self, action):
        reward = STEP_REWARD
        terminated = False
        truncated = False
        death_reason = None
        
        # 取得控制值
        steer_val = np.clip(action[0], -1.0, 1.0)
        throttle_val = np.clip(action[1], -1.0, 1.0)

        # POP移動 (Agents)
        # 速度與舵效計算
        speed_val = np.linalg.norm(self.vel)
        # 基礎舵效：(0 -> 0.2 速度區間)
        sensitivity = np.clip(speed_val / 0.2, 0.0, 1.0) ** 2.0
        # 高速衰減：(MAX_SPEED * 0.875 -> MAX_SPEED 速度區間)
        high_speed_damping = np.clip(1.875 - (speed_val / POP_MAX_SPEED), 0.7, 1.0)
        # 計算轉角增量並更新        
        steer_delta = steer_val * POP_MAX_STEER * sensitivity * high_speed_damping
        self.angle += steer_delta
        # 計算車頭向量
        pop_vec = np.array([np.cos(self.angle), np.sin(self.angle)])
        # 換算油門與推力
        throttle_factor = POP_MAX_SPEED * POP_DAMPING_FACTOR
        if throttle_val < 0:
            throttle_factor *= POP_BACKWARD_FACTOR
        # 推力 = 向量 * 油門
        thrust = pop_vec * (throttle_val * throttle_factor)
        # 速度更新：V = V * (1 - Damping) + Thrust
        self.vel = self.vel * (1 - POP_DAMPING_FACTOR) + thrust
        # 位置更新：P = P + V
        self.pos += self.vel
        self.pos = np.clip(self.pos, POP_RADIUS, np.array([SCREEN_W, SCREEN_H]) - POP_RADIUS)

        # 掠食者移動 (Predators)
        if PREDATOR_SIZE > 0:
            if self.move_predator:
                self._update_entities(self.pred_pos, self.pred_vel, PREDATOR_RADIUS, PREDATOR_MIN_SPEED, PREDATOR_MAX_SPEED)
            # 距離計算
            dists_to_pred = np.linalg.norm(self.pred_pos - self.pos, axis=1)
            # 被吃了
            if np.any(dists_to_pred < (POP_RADIUS + PREDATOR_RADIUS)):
                reward += KILLED_REWARD
                terminated = True
                death_reason = 'killed'
            # 靠近告警
            elif np.any(dists_to_pred < POP_ALERT_RADIUS):
                reward += PREDATOR_NEARBY_REWARD

        # 牆牆處理
        if not terminated:
            for w_a, w_b in self.walls:
                _, dist = point_to_line_segment(self.pos, w_a, w_b)
                if dist < POP_RADIUS:
                    reward += COLLIDED_REWARD
                    terminated = True
                    death_reason = 'collided'
                    break
                elif dist < POP_ALERT_RADIUS:
                    reward += WALL_NEARBY_REWARD
                    break

        # 食物移動 (Food)
        if FOOD_SIZE > 0:
            if self.move_food:
                self._update_entities(self.food_pos, self.food_vel, FOOD_RADIUS, 0.5, 1.0)
            if not terminated:
                dists_to_food = np.linalg.norm(self.food_pos - self.pos, axis=1)
                eaten = dists_to_food < (POP_RADIUS + FOOD_RADIUS)
                if np.any(eaten):
                    num_eaten = np.sum(eaten)
                    reward += FOOD_REWARD * num_eaten
                    self.energy = min(MAX_ENERGY, self.energy + FOOD_ENERGY * num_eaten)
                    # 重生食物
                    self.food_pos[eaten] = np.random.rand(num_eaten, 2) * np.array([SCREEN_W, SCREEN_H])

        # 能量消耗
        if not terminated:
            static_cost = 0.2 * ENERGY_DECAY
            dynamic_cost = 0.8 * ENERGY_DECAY * (throttle_val ** 2)
            self.energy -= (static_cost + dynamic_cost)
            # 能量耗盡
            if self.energy <= 0:
                reward += STARVED_REWARD
                terminated = True
                death_reason = 'starved'

        # --- 移動獎勵計算 ---
        if not terminated:
            # 有效前進速度
            self.forward_speed = np.dot(self.vel, pop_vec)
            # 前進純度
            vel_mag = np.linalg.norm(self.vel) + 1e-6
            vel_purity = max(0.0, np.dot(self.vel / vel_mag, pop_vec))
            # 1. 基礎移動獎勵
            move_reward = MOVE_REWARD * 2.5 * (max(0, self.forward_speed) / POP_MAX_SPEED) * vel_purity
            # 如果「側滑」太嚴重（速度方向與車頭方向不一），直接扣除所有移動獎勵
            # 我們定義一個「有效係數」，如果 purity < 0.8，move_reward 快速衰減
            eff_move_factor = vel_purity ** 4
            move_reward *= eff_move_factor
            # 2. 動作變動量懲罰 (要求動作平滑度)
            action_diff = action - self.last_action
            smooth_penalty = MOVE_REWARD * 0.2 * np.sum(action_diff ** 2)
            # 3. 控制油門效率懲罰
            abs_throttle = np.abs(throttle_val)
            throttle_penalty = MOVE_REWARD * 0.2 * max(0.0, abs_throttle - 0.75)
            # 4. 轉向懲罰
            steer_penalty = MOVE_REWARD * 0.3 * ((steer_val * throttle_val) ** 2)
            # 5. 角速度懲罰，如果原地轉圈(forward_speed小)，但steer大，扣分加劇
            # 引入【旋轉臨界懲罰】：當夾角過大(purity過低)，懲罰指數級上升
            # 當 purity=1.0 時，bonus_spin=0；當 purity=0.8 時，開始劇烈扣分
            purity_gap = max(0.0, 0.95 - vel_purity)
            spinning_bonus = MOVE_REWARD * 5.0 * (purity_gap ** 2) * abs(steer_val)
            spinning_penalty = MOVE_REWARD * 3.0 * (steer_val ** 2) * (1.0 - vel_purity)
            # 6. 物理角速度的懲罰
            omega_yaw = (self.angle - self.last_angle) / POP_MAX_STEER
            yaw_penalty = MOVE_REWARD * 20.0 * abs(omega_yaw)
            # 7. 整合獎勵
            total_step_reward = move_reward - smooth_penalty - throttle_penalty - steer_penalty - spinning_penalty - spinning_bonus - yaw_penalty
            # print(f'move:{self.forward_speed.item():.2f}|{vel_purity.item():.2f}|{eff_move_factor.item():.2f}|{move_reward.item()/1e-4:.0f} '
            #       f'smooth:{torch.sum(action_diff.abs()).item():.2f}|{smooth_penalty.item()/1e-4:.0f} '
            #       f'throttle:{abs_throttle.item():.2f}|{throttle_penalty.item()/1e-4:.0f} '
            #       f'steer:{steer_vals.item():.2f}|{steer_penalty.item()/1e-4:.0f} '
            #       f'spinning:{spinning_bonus.item()/1e-4:.0f}|{spinning_penalty.item()/1e-4:.0f} '
            #       f'yaw:{omega_yaw.item():.4f}|{yaw_penalty.item()/1e-4:.0f} '
            #       f'total:{total_step_reward.item()/1e-4:.0f}')
            reward += total_step_reward

        # 更新歷史
        self.last_action = action.copy()
        obs = self._get_obs()
        info = {
            "death_reason": death_reason
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _update_entities(self, pos_arr, vel_arr, radius, min_speed, max_speed):
        if len(pos_arr) == 0:
            return
        # 加入擾動
        change_mask = np.random.rand(len(pos_arr), 1) < 0.05
        vel_arr += (np.random.rand(*vel_arr.shape) - 0.5) * (1.8 * change_mask)
        # 限制速度
        speeds = np.linalg.norm(vel_arr, axis=1, keepdims=True) + 1e-6
        vel_arr *= np.clip(speeds, min_speed, max_speed) / speeds
        pos_arr += vel_arr
        # 邊界反彈
        out_min = pos_arr < radius
        out_max = pos_arr > np.array([SCREEN_W, SCREEN_H]) - radius
        vel_arr[out_min | out_max] *= -1.0
        np.clip(pos_arr, radius, np.array([SCREEN_W, SCREEN_H]) - radius, out=pos_arr)

    def _get_obs(self):
        # [cos, sin, dist, energy, is_wall, is_food, is_team, is_pred]
        objects = []
        
        for w_a, w_b in self.walls:
            closest, dist = point_to_line_segment(self.pos, w_a, w_b)
            if dist - POP_RADIUS < POP_PERCEPTION_RADIUS:
                rel_ang = np.arctan2(closest[1]-self.pos[1], closest[0]-self.pos[0]) - self.angle
                dist_val = (dist - POP_RADIUS) / POP_PERCEPTION_RADIUS
                objects.append([np.cos(rel_ang), np.sin(rel_ang), dist_val, 0.0, 1.0, 0.0, 0.0, 0.0])
                
        for f_pos in self.food_pos:
            dist = np.linalg.norm(f_pos - self.pos)
            if dist - POP_RADIUS - FOOD_RADIUS < POP_PERCEPTION_RADIUS:
                rel_ang = np.arctan2(f_pos[1]-self.pos[1], f_pos[0]-self.pos[0]) - self.angle
                dist_val = (dist - POP_RADIUS - FOOD_RADIUS) / POP_PERCEPTION_RADIUS
                objects.append([np.cos(rel_ang), np.sin(rel_ang), dist_val, 0.0, 0.0, 1.0, 0.0, 0.0])

        for p_pos in self.pred_pos:
            dist = np.linalg.norm(p_pos - self.pos)
            if dist - POP_RADIUS - PREDATOR_RADIUS < POP_PERCEPTION_RADIUS:
                rel_ang = np.arctan2(p_pos[1]-self.pos[1], p_pos[0]-self.pos[0]) - self.angle
                dist_val = (dist - POP_RADIUS - PREDATOR_RADIUS) / POP_PERCEPTION_RADIUS
                objects.append([np.cos(rel_ang), np.sin(rel_ang), dist_val, 1.0, 0.0, 0.0, 0.0, 1.0])

        objects.sort(key=lambda x: x[2])
        objects = objects[:MAX_OBJ]
        while len(objects) < MAX_OBJ:
            objects.append([0.0]*8)
            
        env_features = np.array(objects, dtype=np.float32).T # Shape: (8, 25) 以符合 Conv1d
        env_features = np.array(env_features, dtype=np.float32).flatten()

        delta_pos = self.pos - self.last_pos
        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        self_features = np.array([
            self.forward_speed / POP_MAX_SPEED,
            np.dot(self.vel, np.array([-sin_a, cos_a])) / POP_MAX_SPEED,
            self.last_action[0],
            self.energy / MAX_ENERGY,
            (delta_pos[0] * cos_a + delta_pos[1] * sin_a) / POP_MAX_SPEED,
            (-delta_pos[0] * sin_a + delta_pos[1] * cos_a) / POP_MAX_SPEED,
            (self.angle - self.last_angle) / POP_MAX_STEER
        ], dtype=np.float32)

        self.last_pos = self.pos.copy()
        self.last_angle = self.angle

        return {"env_features": env_features, "self_features": self_features}

    def render(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.SCALED)
            pygame.display.set_caption(CAPTION)
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((SCREEN_W, SCREEN_H))
        canvas.fill((20, 20, 20))

        for w_a, w_b in self.walls:
            pygame.draw.line(canvas, (128, 0, 0), w_a, w_b, 2)

        for f_pos in self.food_pos:
            pygame.draw.circle(canvas, (0, 255, 120), f_pos.astype(int), FOOD_RADIUS)

        for p_pos in self.pred_pos:
            pygame.draw.circle(canvas, (255, 0, 0), p_pos.astype(int), PREDATOR_RADIUS, 2)
            pygame.draw.circle(canvas, (255, 0, 0), p_pos.astype(int), int(PREDATOR_RADIUS*0.25))

        en_ratio = max(0, self.energy / MAX_ENERGY)
        color = (int(255*en_ratio), int(128*en_ratio), int(255*(1-en_ratio)))
        pygame.draw.circle(canvas, color, self.pos.astype(int), POP_RADIUS + int(4*en_ratio))
        
        end_pos = self.pos + np.array([np.cos(self.angle), np.sin(self.angle)]) * 15
        pygame.draw.line(canvas, (255, 255, 255), self.pos.astype(int), end_pos.astype(int), 2)

        self.window.blit(canvas, canvas.get_rect())
        
        # 處理 pygame 事件避免視窗無回應
        pygame.event.pump()
        if self.render_mode == "human":
            # 這會確保展示時不會太快，且視窗不會失去響應
            pygame.display.flip()
            self.clock.tick(self.render_fps)
        elif self.render_mode == "rgb_array":
            # 如果是為了錄影，就不需要 tick，直接回傳像素矩陣
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

# ==========================================
# 自訂特徵提取器
# ==========================================
class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim=HIDDEN_FC_DIM):
        super().__init__(observation_space, features_dim)
        
        # 空間感知層
        self.conv = nn.Sequential(
            nn.Conv1d(FEAT_IN_DIM, HIDDEN_FEAT_DIM, 1),
            nn.LayerNorm([HIDDEN_FEAT_DIM, MAX_OBJ]), 
            nn.ReLU(),
            nn.Conv1d(HIDDEN_FEAT_DIM, HIDDEN_FEAT_DIM, 1),
            nn.ReLU()
        )

        # Attention 機制
        self.attn_weights = nn.Sequential(
            nn.Linear(HIDDEN_FEAT_DIM, HIDDEN_ATTN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_ATTN_DIM, 1)
        )
        
        # 決策融合層
        self.feat_norm = nn.LayerNorm(HIDDEN_FEAT_DIM + STATE_IN_DIM)
        
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_FEAT_DIM + STATE_IN_DIM, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        env_feat = observations["env_features"] # Shape: (Batch, FEAT_IN_DIM x MAX_OBJ)
        env_feat = env_feat.view(-1, FEAT_IN_DIM, MAX_OBJ)
        self_feat = observations["self_features"] # Shape: (Batch, 7)
        
        feat = self.conv(env_feat).transpose(1, 2) 
        weights = F.softmax(self.attn_weights(feat), dim=1)
        x_attn = torch.sum(feat * weights, dim=1)
        
        combined = self.feat_norm(torch.cat([x_attn, self_feat], dim=1))
        return self.fc(combined)

# ==========================================
# 訓練中看見畫面的 Callback
# ==========================================
class RenderEvalCallback(BaseCallback):
    """
    定期暫停無頭訓練，開啟畫面展示目前模型的行為
    """
    def __init__(self, eval_freq: int, render_steps: int = 500, verbose=0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.render_steps = render_steps
        self.eval_env = SurvivorsEnv(render_mode="human") # 建立一個帶畫面的獨立環境

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            print(f"\n[Callback] 訓練步數達 {self.n_calls}，開啟畫面展示學習成果...")
            obs, _ = self.eval_env.reset()
            for _ in range(self.render_steps):
                # 取得預測動作 (Deterministic)
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                if terminated or truncated:
                    obs, _ = self.eval_env.reset()
            print("[Callback] 展示結束，繼續並行訓練...\n")
        return True

class DeathAnalysisCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.death_counts = {"killed": 0, "collided": 0, "starved": 0}

    def _on_step(self) -> bool:
        # SB3 會把每個環境的 info 放在 self.locals["infos"]
        for info in self.locals["infos"]:
            if "death_reason" in info:
                reason = info["death_reason"]
                if reason in self.death_counts:
                    self.death_counts[reason] += 1
                    
        # 每隔一段時間紀錄到 TensorBoard
        if self.n_calls % 1000 == 0:
            for reason, count in self.death_counts.items():
                self.logger.record(f"deaths/{reason}", count)

        return True
    
def make_env(rank, render_mode=None):
    def _init():
        env = SurvivorsEnv(render_mode=render_mode)
        check_env(env, warn=True)

        max_steps = None
        if STAGE == 1:
            max_steps = 3000
        elif STAGE == 2:
            max_steps = 5000
        elif STAGE == 3:
            max_steps = 10000
        elif STAGE == 4:
            max_steps = 8000
        if max_steps:
            env = TimeLimit(env, max_episode_steps=max_steps)

        env = Monitor(env)
        return env
    
    return _init

# ==========================================
# 主程式執行區塊
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=CAPTION)
    parser.add_argument("-e", "--epoch", type=str, default=None, help="一個訓練週期")
    parser.add_argument("-s", "--steps", type=int, default=float('inf'), help="達到此步數時退出")
    parser.add_argument("--vec", type=int, default=16, help="並行環境數量")
    parser.add_argument("-r", "--record", action="store_true", default=False, help="啟動即開始錄影")
    parser.add_argument("--demo", action="store_true", default=False, help="模型性能展示")
    parser.add_argument("--frames", type=int, default=float('inf'), help="模型性能展示幀數")
    parser.add_argument("--ui", action="store_true", default=False, help="UI模式")
    args = parser.parse_args()

    if args.demo:
        if not os.path.exists(FINAL_FULL_PATH):
            print(f"錯誤：找不到權重檔案 {FINAL_FULL_PATH}")
        else:
            # 展示訓練成果 (開啟渲染)
            venv = DummyVecEnv([make_env(0, render_mode='human')])
            model = SAC.load(FINAL_FULL_PATH, env=venv)
            # print(model.policy)
            obs = venv.reset()
            try:
                running = True
                while running:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_ESCAPE:
                                running = False
                            elif event.key == pygame.K_f:
                                val = venv.envs[0].get_wrapper_attr("move_food")
                                venv.envs[0].set_wrapper_attr("move_food", not val)
                            elif event.key == pygame.K_m:
                                val = venv.envs[0].get_wrapper_attr("move_predator")
                                venv.envs[0].set_wrapper_attr("move_predator", not val)
                            elif event.key == pygame.K_v:
                                val = venv.envs[0].get_wrapper_attr("verbose")
                                venv.envs[0].set_wrapper_attr("verbose", (val + 1) % 4)

                    action, _ = model.predict(obs, deterministic=True)
                    obs, rewards, dones, infos = venv.step(action)
                    if dones:
                        obs = venv.reset()
            except KeyboardInterrupt:
                pass

            venv.close()
    else:
        # 建立並行環境 (Vectorized Environments)
        if args.vec > 1:
            venv = [make_env(i) for i in range(args.vec)]
            venv = SubprocVecEnv(venv)
        else:
            if args.ui:
                venv = make_env(0, render_mode='human')
            else:
                venv = make_env(0)
            venv = DummyVecEnv([venv])

        if os.path.exists(FINAL_FULL_PATH):
            print("加載舊模型繼續訓練...")
            model = SAC.load(FINAL_FULL_PATH, env=venv)
        else:
            print("建立新模型...")
            model = SAC(
                "MultiInputPolicy",
                venv,
                learning_rate=0.0003,
                buffer_size=200000,
                learning_starts=100,
                batch_size=512,
                tau=0.005  ,
                gamma=0.97,
                train_freq=(1, 'step'),
                gradient_steps=1,
                ent_coef='auto',
                target_entropy='auto',
                tensorboard_log=LOG_PATH,
                policy_kwargs=dict(
                    features_extractor_class=CustomFeaturesExtractor,
                    features_extractor_kwargs=dict(features_dim=HIDDEN_FC_DIM),
                    net_arch=dict(
                        pi=[HIDDEN_FC_DIM],
                        qf=[HIDDEN_FC_DIM]
                    ),
                    activation_fn=nn.ReLU
                ),
                verbose=2,
                device='cuda'
            )

        callbacks = [DeathAnalysisCallback()]
        if args.epoch:
            callbacks.append(CheckpointCallback(
                save_freq=10000, 
                save_path=f'{BASE_PATH}', 
                name_prefix=f'{script_name}_{args.epoch}'
            ))

        print(f"啟動 {args.vec} 個進程進行並行訓練... (按 Ctrl+C 提早中斷)")
        try:
            # 進行訓練
            model.learn(
                total_timesteps=args.steps * args.vec,
                callback=CallbackList(callbacks),
                log_interval=4,
                tb_log_name=args.epoch if args.epoch else "SAC",
                progress_bar=True
            )
            model.save(FINAL_PATH)
            print(f"訓練結束，模型儲存至 {FINAL_PATH}。")
            if args.epoch:
                model.save(f"{SAVE_PATH}_{args.epoch}")
        except KeyboardInterrupt:
            print("訓練提早中斷，正在儲存模型...")
            model.save(INTERRUPTED_PATH)

        venv.close()
