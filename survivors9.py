import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import math
import os
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

# ==========================================
# 參數設定 (保留原先的核心設定)
# ==========================================
STAGE = 2
SCREEN_W, SCREEN_H = 1280, 800
SCALE = 1.0  # 簡化顯示倍率
POP_MAX_SPEED = 3.5
POP_RADIUS = 4
POP_DAMPING_FACTOR = 0.25
POP_BACKWARD_FACTOR = 0.33
POP_MAX_STEER = math.radians(15)
POP_PERCEPTION_RADIUS = 200

# 環境物件設定
FOOD_SIZE = 50 if STAGE < 5 else 1
FOOD_RADIUS = 3
MAX_ENERGY = 100.0
FOOD_ENERGY = 25.0
ENERGY_DECAY = 0 if STAGE == 1 else FOOD_ENERGY / 500.0 # 簡化預估步數

PREDATOR_SIZE = 0 if STAGE < 3 else 5
PREDATOR_RADIUS = 20.0
PREDATOR_MIN_SPEED = 1.5
PREDATOR_MAX_SPEED = 2.5 if STAGE < 3 else 3.0
POP_ALERT_RADIUS = max(POP_MAX_SPEED, (PREDATOR_MAX_SPEED / POP_MAX_SPEED) ** 2 * 20.58)

# 獎懲設定
FOOD_REWARD = 100 if STAGE < 1 else 50.0
KILLED_REWARD = -75.0
COLLIDED_REWARD = -60.0
STARVED_REWARD = -70.0
MOVE_REWARD_FACTOR = 0.35
EST_STEPS = 500.0
MOVE_REWARD = FOOD_REWARD * MOVE_REWARD_FACTOR / EST_STEPS
TIME_PENALTY_FACTOR = 0 if STAGE == 1 else 0.15
STEP_REWARD = STARVED_REWARD * TIME_PENALTY_FACTOR / EST_STEPS
WALL_NEARBY_REWARD = COLLIDED_REWARD * 0.25
PREDATOR_NEARBY_REWARD = KILLED_REWARD * 0.25

MAX_OBJ = 25
FEAT_IN_DIM = 8
STATE_IN_DIM = 7

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

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        
        # 動作空間：[轉向, 油門]，範圍 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # 觀察空間：將原來的 (FEAT_IN_DIM x MAX_OBJ) 與 (STATE_IN_DIM) 攤平
        obs_dim = FEAT_IN_DIM * MAX_OBJ + STATE_IN_DIM
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
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
        
        self.frames = 0
        
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self.frames += 1
        reward = STEP_REWARD
        terminated = False
        truncated = False
        
        steer_val = np.clip(action[0], -1.0, 1.0)
        throttle_val = np.clip(action[1], -1.0, 1.0)

        # 1. 更新 Agent 物理
        speed_val = np.linalg.norm(self.vel)
        sensitivity = np.clip(speed_val / 0.2, 0.0, 1.0) ** 2.0
        high_speed_damping = np.clip(1.875 - (speed_val / POP_MAX_SPEED), 0.7, 1.0)
        
        steer_delta = steer_val * POP_MAX_STEER * sensitivity * high_speed_damping
        self.angle += steer_delta
        
        pop_vec = np.array([np.cos(self.angle), np.sin(self.angle)])
        throttle_factor = POP_MAX_SPEED * POP_DAMPING_FACTOR
        if throttle_val < 0:
            throttle_factor *= POP_BACKWARD_FACTOR
            
        thrust = pop_vec * (throttle_val * throttle_factor)
        self.vel = self.vel * (1 - POP_DAMPING_FACTOR) + thrust
        self.pos += self.vel
        
        # 確保在邊界內
        self.pos = np.clip(self.pos, POP_RADIUS, np.array([SCREEN_W, SCREEN_H]) - POP_RADIUS)

        # 2. 更新環境物件 (簡單反彈邏輯)
        self._update_entities(self.food_pos, self.food_vel, FOOD_RADIUS, 0.5, 1.0)
        self._update_entities(self.pred_pos, self.pred_vel, PREDATOR_RADIUS, PREDATOR_MIN_SPEED, PREDATOR_MAX_SPEED)

        # 3. 碰撞與獎懲邏輯
        # 撞牆
        for w_a, w_b in self.walls:
            _, dist = point_to_line_segment(self.pos, w_a, w_b)
            if dist < POP_RADIUS:
                reward += COLLIDED_REWARD
                terminated = True
            elif dist < POP_ALERT_RADIUS:
                reward += WALL_NEARBY_REWARD

        # 吃食物
        if FOOD_SIZE > 0:
            dists_to_food = np.linalg.norm(self.food_pos - self.pos, axis=1)
            eaten = dists_to_food < (POP_RADIUS + FOOD_RADIUS)
            if np.any(eaten):
                num_eaten = np.sum(eaten)
                reward += FOOD_REWARD * num_eaten
                self.energy = min(MAX_ENERGY, self.energy + FOOD_ENERGY * num_eaten)
                # 重生食物
                self.food_pos[eaten] = np.random.rand(num_eaten, 2) * np.array([SCREEN_W, SCREEN_H])

        # 被掠食者吃
        if PREDATOR_SIZE > 0:
            dists_to_pred = np.linalg.norm(self.pred_pos - self.pos, axis=1)
            if np.any(dists_to_pred < (POP_RADIUS + PREDATOR_RADIUS)):
                reward += KILLED_REWARD
                terminated = True
            elif np.any(dists_to_pred < POP_ALERT_RADIUS):
                reward += PREDATOR_NEARBY_REWARD

        # 能量消耗與餓死
        static_cost = 0.2 * ENERGY_DECAY
        dynamic_cost = 0.8 * ENERGY_DECAY * (throttle_val ** 2)
        self.energy -= (static_cost + dynamic_cost)
        
        if self.energy <= 0:
            reward += STARVED_REWARD
            terminated = True

        # 4. 移動品質獎勵 (精簡版)
        self.forward_speed = np.dot(self.vel, pop_vec)
        vel_mag = np.linalg.norm(self.vel) + 1e-6
        vel_purity = max(0.0, np.dot(self.vel / vel_mag, pop_vec))
        move_reward = MOVE_REWARD * 2.5 * (max(0, self.forward_speed) / POP_MAX_SPEED) * vel_purity
        eff_move_factor = vel_purity ** 4
        move_reward *= eff_move_factor
        
        # 動作平滑懲罰等... (轉換為單一代理的計算)
        action_diff = action - self.last_action
        smooth_penalty = MOVE_REWARD * 0.2 * np.sum(action_diff ** 2)
        reward += (move_reward - smooth_penalty)

        # 更新歷史
        self.last_action = action.copy()
        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def _update_entities(self, pos_arr, vel_arr, radius, min_speed, max_speed):
        if len(pos_arr) == 0: return
        pos_arr += vel_arr
        
        # 邊界反彈
        out_min = pos_arr < radius
        out_max = pos_arr > np.array([SCREEN_W, SCREEN_H]) - radius
        vel_arr[out_min | out_max] *= -1.0
        np.clip(pos_arr, radius, np.array([SCREEN_W, SCREEN_H]) - radius, out=pos_arr)

    def _get_obs(self):
        """建構觀察矩陣並將其攤平"""
        objects = []
        
        # 1. 牆壁特徵
        for w_a, w_b in self.walls:
            closest, dist = point_to_line_segment(self.pos, w_a, w_b)
            if dist - POP_RADIUS < POP_PERCEPTION_RADIUS:
                diff = closest - self.pos
                abs_ang = np.arctan2(diff[1], diff[0])
                rel_ang = abs_ang - self.angle
                dist_val = (dist - POP_RADIUS) / POP_PERCEPTION_RADIUS
                # [cos, sin, dist, energy, is_wall, is_food, is_team, is_pred]
                feat = [np.cos(rel_ang), np.sin(rel_ang), dist_val, 0.0, 1.0, 0.0, 0.0, 0.0]
                objects.append(feat)
                
        # 2. 食物特徵
        for f_pos in self.food_pos:
            dist = np.linalg.norm(f_pos - self.pos)
            if dist - POP_RADIUS - FOOD_RADIUS < POP_PERCEPTION_RADIUS:
                diff = f_pos - self.pos
                abs_ang = np.arctan2(diff[1], diff[0])
                rel_ang = abs_ang - self.angle
                dist_val = (dist - POP_RADIUS - FOOD_RADIUS) / POP_PERCEPTION_RADIUS
                feat = [np.cos(rel_ang), np.sin(rel_ang), dist_val, 0.0, 0.0, 1.0, 0.0, 0.0]
                objects.append(feat)

        # 3. 掠食者特徵
        for p_pos in self.pred_pos:
            dist = np.linalg.norm(p_pos - self.pos)
            if dist - POP_RADIUS - PREDATOR_RADIUS < POP_PERCEPTION_RADIUS:
                diff = p_pos - self.pos
                abs_ang = np.arctan2(diff[1], diff[0])
                rel_ang = abs_ang - self.angle
                dist_val = (dist - POP_RADIUS - PREDATOR_RADIUS) / POP_PERCEPTION_RADIUS
                feat = [np.cos(rel_ang), np.sin(rel_ang), dist_val, 1.0, 0.0, 0.0, 0.0, 1.0]
                objects.append(feat)

        # 排序並取前 MAX_OBJ 個
        objects.sort(key=lambda x: x[2]) # 依距離排序
        objects = objects[:MAX_OBJ]
        
        # 不足 MAX_OBJ 則補零
        while len(objects) < MAX_OBJ:
            objects.append([0.0]*8)
            
        # 攤平環境特徵 (8 * 25 = 200)
        env_feat = np.array(objects, dtype=np.float32).flatten()

        # 4. 自身狀態 (7 維)
        delta_pos = self.pos - self.last_pos
        cos_a, sin_a = np.cos(self.angle), np.sin(self.angle)
        dx_ego = delta_pos[0] * cos_a + delta_pos[1] * sin_a
        dy_ego = -delta_pos[0] * sin_a + delta_pos[1] * cos_a
        side_speed = np.dot(self.vel, np.array([-sin_a, cos_a]))
        omega_yaw = (self.angle - self.last_angle) / POP_MAX_STEER

        self_feat = np.array([
            self.forward_speed / POP_MAX_SPEED,
            side_speed / POP_MAX_SPEED,
            self.last_action[0],
            self.energy / MAX_ENERGY,
            dx_ego / POP_MAX_SPEED,
            dy_ego / POP_MAX_SPEED,
            omega_yaw
        ], dtype=np.float32)

        self.last_pos = self.pos.copy()
        self.last_angle = self.angle

        # 合併輸出
        return np.concatenate([env_feat, self_feat])

    def render(self):
        if self.render_mode is None:
            gym.logger.warn("You are calling render method without specifying any render mode.")
            return

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((SCREEN_W, SCREEN_H))
        canvas.fill((20, 20, 20))

        # 畫牆壁
        for w_a, w_b in self.walls:
            pygame.draw.line(canvas, (128, 0, 0), w_a, w_b, 2)

        # 畫食物
        for f_pos in self.food_pos:
            pygame.draw.circle(canvas, (0, 255, 120), f_pos.astype(int), FOOD_RADIUS)

        # 畫掠食者
        for p_pos in self.pred_pos:
            pygame.draw.circle(canvas, (255, 0, 0), p_pos.astype(int), PREDATOR_RADIUS, 2)
            pygame.draw.circle(canvas, (255, 0, 0), p_pos.astype(int), int(PREDATOR_RADIUS*0.25))

        # 畫 Agent
        en_ratio = max(0, self.energy / MAX_ENERGY)
        color = (int(255*en_ratio), int(128*en_ratio), int(255*(1-en_ratio)))
        pygame.draw.circle(canvas, color, self.pos.astype(int), POP_RADIUS + int(4*en_ratio))
        
        # 畫 Agent 方向線
        end_pos = self.pos + np.array([np.cos(self.angle), np.sin(self.angle)]) * 15
        pygame.draw.line(canvas, (255, 255, 255), self.pos.astype(int), end_pos.astype(int), 2)

        self.window.blit(canvas, canvas.get_rect())
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

# ==========================================
# 執行與訓練邏輯 (Stable Baselines 3)
# ==========================================
if __name__ == "__main__":
    # 建立目錄
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # 1. 檢查自定義環境是否符合 Gym 規範
    env = SurvivorsEnv()
    check_env(env, warn=True)
    print("環境檢查通過！符合 Gymnasium 標準。")

    # 2. 建立 SAC 模型
    # MlpPolicy 會自動根據 env.observation_space 與 env.action_space 建立對應的神經網路
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        buffer_size=100000,
        batch_size=256,
        ent_coef='auto', # 自動調整 Entropy (對應你原本的 log_alpha 機制)
        gamma=0.97,
        tau=0.005,
        tensorboard_log="./logs/sac_survivors_tb/"
    )

    # 3. 訓練設定
    # 每 10000 步存一次檔
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path="./models/", 
        name_prefix="rl_model"
    )

    print("開始訓練... (可在終端機按 Ctrl+C 提早中斷並查看展示)")
    try:
        # 進行訓練
        model.learn(total_timesteps=300000, callback=checkpoint_callback)
        model.save("models/sac_survivors_final")
    except KeyboardInterrupt:
        print("訓練提早中斷，正在儲存模型...")
        model.save("models/sac_survivors_interrupted")

    # 4. 展示訓練成果 (開啟渲染)
    print("=== 啟動展示模式 ===")
    demo_env = SurvivorsEnv(render_mode="human")
    obs, info = demo_env.reset()
    
    for i in range(2000): # 跑 2000 步看看
        # 透過模型預測動作 (deterministic=True 讓動作不再帶有探索噪聲)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = demo_env.step(action)
        
        if terminated or truncated:
            obs, info = demo_env.reset()

    demo_env.close()