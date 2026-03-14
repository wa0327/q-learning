import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
from datetime import datetime

script_name = Path(__file__).stem
CAPTION = "Vectra: Apex Protocol"
# --- 參數設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCREEN_W, SCREEN_H = 900, 700
SAVE_PATH = f"{script_name}.pt"

# 環境參數
POP_SIZE = 16
FOOD_SIZE = POP_SIZE * 2
PREDATOR_SIZE = 3
PREDATOR_RADIUS = 10.0
PREDATOR_CIRCLE = 50.0
PREDATOR_MIN_SPEED = 1.5
PREDATOR_MAX_SPEED = 2.5
MAX_ENERGY = 100.0      # 能量最大總值
FOOD_ENERGY = 25.0      # 食物補充能量
ENERGY_DECAY = FOOD_ENERGY / 250 # 每步消耗能量
PERCEPTION_RADIUS = 200 # 視野感知半徑
ALERT_RADIUS = 100      # 警戒敵人半徑
WALL_SENSE_RADIUS = 50  # 牆壁感知半徑
TEAM_RADIUS = 30        # 友方過近半徑
THROTTLE_FACTOR = 0.6
DAMPING_FACTOR = 0.85

# 獎懲設定
STEP_REWARD = -0.01     # 每步
FOOD_REWARD = 15.0      # 吃到食物
PRED_REWARD = -50.0     # 被吃
WALL_REWARD = -50.0     # 撞牆
STARVED_REWARD = -35.0  # 餓死

# DDPG 核心參數
GAMMA = 0.98
TAU = 0.005 # 軟更新係數
LR_ACTOR = 0.0003
LR_CRITIC = 0.001
MEMORY_SIZE = 50000
BATCH_SIZE = 256
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9995
MAX_OBJ = 5 # 最大環境物件數量
FEAT_DIM = 12 # 每個物件特微 [cos, sin, score] x 四種類型

# --- DDPG 網路架構 ---
# --- Actor 網路：策略決策者 ---
# 基礎型
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.conv = nn.Conv1d(FEAT_DIM, 32, 1) 
        self.fc = nn.Sequential(
            # 32 (卷積特徵) + 3 (自身狀態: 速度, 0, 能量) = 35
            nn.Linear(35, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2), # 輸出 2: [Steer(轉向), Throttle(油門)]
            nn.Tanh()         # 將輸出壓縮至 [-1, 1]
        )

    def forward(self, m_in, s_in):
        """
        m_in: [Batch, MAX_OBJ, FEAT_DIM]  (MAX_OBJ 個物體)
        s_in: [Batch, 3]                  (自身狀態)
        action: [Batch, 2]                (動作)
        """
        feat = F.relu(self.conv(m_in))
        feat = torch.max(feat, dim=2)[0]
        combined = torch.cat([feat, s_in], dim=1)
        return self.fc(combined)

# Max + Mean 混合型
class ActorPlus(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(FEAT_DIM, 32, 1) 
        # 維度：32(Max) + 32(Mean) + 3(Self) = 67
        self.fc = nn.Sequential(
            nn.Linear(67, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )

    def forward(self, m_in, s_in):
        """
        m_in: [Batch, MAX_OBJ, FEAT_DIM]  (MAX_OBJ 個物體)
        s_in: [Batch, 3]                  (自身狀態)
        action: [Batch, 2]                (動作)
        """
        feat = F.relu(self.conv(m_in))
        
        # 關鍵差異：同時保留最強威脅與整體分佈
        feat_max = torch.max(feat, dim=2)[0]
        feat_avg = torch.mean(feat, dim=2)

        combined = torch.cat([feat_max, feat_avg, s_in], dim=1)
        return self.fc(combined)

# 注意機制型
class ActorAttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(FEAT_DIM, 32, 1)
        self.attn_weights = nn.Linear(32, 1) 
        self.fc = nn.Sequential(
            nn.Linear(32 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )

    def forward(self, m_in, s_in):
        feat = F.relu(self.conv(m_in))
        feat = feat.transpose(1, 2)
        weights = F.softmax(self.attn_weights(feat), dim=1)
        x_attn = torch.sum(feat * weights, dim=1)
        combined = torch.cat([x_attn, s_in], dim=1)
        return self.fc(combined)

# Transformer 結構型
class ActorTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(FEAT_DIM, 32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, batch_first=True, dropout=0
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.fc = nn.Sequential(
            nn.Linear(32 + 32 + 3, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Tanh()
        )

    def forward(self, m_in, s_in):
        """
        m_in: [Batch, MAX_OBJ, FEAT_DIM]  (MAX_OBJ 個物體)
        s_in: [Batch, 3]                  (自身狀態)
        action: [Batch, 2]                (動作)
        """
        x = m_in.transpose(1, 2)
        x = self.embedding(m_in)
        x = self.transformer(x)

        # 雙重感知池化
        x_max = torch.max(x, dim=1)[0]  # 捕捉突出的單一物件 (如近處敵人)
        x_mean = torch.mean(x, dim=1)   # 捕捉全局分佈 (如食物群落)
        
        combined = torch.cat([x_max, x_mean, s_in], dim=1)
        return self.fc(combined)
    
# --- Critic 網路：價值評估者 ---
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv = nn.Conv1d(FEAT_DIM, 32, 1)
        self.attn_weights = nn.Linear(32, 1)
        self.fc = nn.Sequential(
            # 32 (感應特徵) + 3 (自身狀態) + 2 (動作) = 37
            nn.Linear(32 + 3 + 2, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, m_in, s_in, action):
        feat = F.relu(self.conv(m_in))
        feat = feat.transpose(1, 2)
        weights = F.softmax(self.attn_weights(feat), dim=1)
        x_attn = torch.sum(feat * weights, dim=1)
        combined = torch.cat([x_attn, s_in, action], dim=1)
        return self.fc(combined)
    
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        
        self.m_states = torch.zeros((capacity, FEAT_DIM, MAX_OBJ), device=DEVICE)
        self.s_states = torch.zeros((capacity, 3), device=DEVICE)
        self.next_m_states = torch.zeros((capacity, FEAT_DIM, MAX_OBJ), device=DEVICE)
        self.next_s_states = torch.zeros((capacity, 3), device=DEVICE)
        self.actions = torch.zeros((capacity, 2), device=DEVICE)
        self.rewards = torch.zeros(capacity, device=DEVICE)
        self.dones = torch.zeros(capacity, device=DEVICE)
        
        self.idx = 0
        self.size = 0

    def push(self, m_s, s_s, action, reward, n_m_s, n_s_s, done):
        self.m_states[self.idx].copy_(m_s)
        self.s_states[self.idx].copy_(s_s)
        self.next_m_states[self.idx].copy_(n_m_s)
        self.next_s_states[self.idx].copy_(n_s_s)
        self.actions[self.idx].copy_(action)
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=DEVICE)
        return (
            self.m_states[indices],
            self.s_states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_m_states[indices],
            self.next_s_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size

class OUNoise:
    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state, dtype=torch.float).to(DEVICE)
    
# --- 模擬環境 ---
class RLSimulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), pygame.SCALED)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 14)
        self.big_font = pygame.font.SysFont("Consolas", 18, bold=True)
        self.fps = 240
        self.update_caption()
        
        # 初始化 DDPG 網路
        self.init_network(ActorAttentionPooling, Critic)
        self.brain_path = f"{script_name}_{self.actor.__class__.__name__}.pt"
        
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.ou_noise = OUNoise(2) # 2 個動作維度
        self.a_loss_val = 0.0
        self.c_loss_val = 0.0
        self.rewards_avg = 0.0
        self.hit_pred = 0
        self.hit_wall = 0
        self.starved = 0
        self.max_speed = ((1.0 + 1.0) * THROTTLE_FACTOR) / (1 - DAMPING_FACTOR)
        print(f'=== Max speed: {self.max_speed:.2f} ===')
        self.reset_env()
        self.load_state()
        
        # 定義四面牆的方位向量, 左牆: (-1, 0), 右牆: (1, 0), 上牆: (0, -1), 下牆: (0, 1)
        self.wall_normals = torch.tensor([
            [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]
        ], device=DEVICE)

    def update_caption(self):
        pygame.display.set_caption(f"{CAPTION} | FPS:{self.fps}")

    def init_network(self, actor=None, critic=None):
        if actor is None:
            actor = self.actor.__class__
        if critic is None:
            critic = self.critic.__class__
        self.actor = actor().to(DEVICE)
        self.actor_target = actor().to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = critic().to(DEVICE)
        self.critic_target = critic().to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.steps = 0
        self.epsilon = EPSILON_START

        print(f"[{self.actor.__class__.__name__}] Network & Optimizer init complete.")
        
    def reset_env(self):
        self.last_actions = torch.zeros((POP_SIZE, 2), device=DEVICE)
        self.vel = torch.zeros((POP_SIZE, 2), device=DEVICE)
        self.angle = torch.rand(POP_SIZE, device=DEVICE) * (2 * np.pi)
        self.energy = torch.full((POP_SIZE,), MAX_ENERGY, device=DEVICE, dtype=torch.float)
        self.alive = torch.ones(POP_SIZE, dtype=torch.bool, device=DEVICE)
        self.respawn_timer = torch.zeros(POP_SIZE, dtype=torch.long, device=DEVICE) 
        self.screen_size = torch.tensor([SCREEN_W, SCREEN_H], device=DEVICE, dtype=torch.float)
        self.bounds = self.screen_size - 1.0
        self.pos = torch.rand(POP_SIZE, 2, device=DEVICE) * self.screen_size
        self.food_pos = torch.rand(FOOD_SIZE, 2, device=DEVICE) * self.screen_size
        self.food_vel = (torch.rand(FOOD_SIZE, 2, device=DEVICE) - 0.5) * 3.5
        self.pred_pos = torch.rand(PREDATOR_SIZE, 2, device=DEVICE) * self.screen_size
        self.pred_vel = (torch.rand(PREDATOR_SIZE, 2, device=DEVICE) - 0.5) * 3.5
    
    def get_states(self):
        # --- 1. 處理牆壁 (Wall: Channel 0-2) ---
        wall_in = torch.zeros((POP_SIZE, 3, MAX_OBJ), device=DEVICE)

        d_left = self.pos[:, 0]
        d_right = SCREEN_W - self.pos[:, 0]
        d_top = self.pos[:, 1]
        d_bottom = SCREEN_H - self.pos[:, 1]
        wall_dists = torch.stack([d_left, d_right, d_top, d_bottom], dim=1) 
        
        wall_angles = torch.atan2(self.wall_normals[:, 1], self.wall_normals[:, 0]).unsqueeze(0) - self.angle.unsqueeze(1)
        wall_presence = 0.1 * (1.0 - wall_dists / PERCEPTION_RADIUS).clamp(min=0)
        wall_threat = torch.pow((1.0 - wall_dists / WALL_SENSE_RADIUS).clamp(min=0), 2)
        wall_mask = (wall_dists < PERCEPTION_RADIUS).float()
        
        wall_phys = torch.stack([torch.cos(wall_angles), torch.sin(wall_angles), torch.max(wall_presence, wall_threat)], dim=2) * wall_mask.unsqueeze(-1)
        wall_in[:, :, :4] = wall_phys.transpose(1, 2)[:, :, :MAX_OBJ]

        # --- 2. 處理食物 (Food: Channel 3-5) ---
        food_in = torch.zeros((POP_SIZE, 3, MAX_OBJ), device=DEVICE)
        if FOOD_SIZE > 0:
            dist_food = torch.cdist(self.pos, self.food_pos)
            k_f = min(MAX_OBJ, FOOD_SIZE)
            val_f, f_idx = torch.topk(dist_food, k_f, largest=False)
            
            selected_food_pos = self.food_pos[f_idx] 
            f_diff = selected_food_pos - self.pos.unsqueeze(1)
            f_ang = torch.atan2(f_diff[..., 1], f_diff[..., 0]) - self.angle.unsqueeze(1)
            
            food_score = (1.0 - val_f / PERCEPTION_RADIUS).clamp(min=0) ** 2 # 越靠近值分數越高
            food_mask = (val_f < PERCEPTION_RADIUS).float()
            
            food_phys = torch.stack([torch.cos(f_ang), torch.sin(f_ang), food_score], dim=2) * food_mask.unsqueeze(-1)
            food_in[:, :, :k_f] = food_phys.transpose(1, 2)

        # --- 4. 處理敵人 (Predator: Channel 9-11) ---
        pred_in = torch.zeros((POP_SIZE, 3, MAX_OBJ), device=DEVICE)
        if PREDATOR_SIZE > 0:
            dist_pred = torch.cdist(self.pos, self.pred_pos)
            k_p = min(MAX_OBJ, PREDATOR_SIZE)
            val_p, p_idx = torch.topk(dist_pred, k_p, largest=False)
            
            selected_pred_pos = self.pred_pos[p_idx]
            p_diff = selected_pred_pos - self.pos.unsqueeze(1)
            p_ang = torch.atan2(p_diff[..., 1], p_diff[..., 0]) - self.angle.unsqueeze(1)
            
            midpoint = (ALERT_RADIUS + PERCEPTION_RADIUS) / 2 # 取感知與警戒的中間值，作為威脅感「爆發」的中點。
            pred_threat = 1.0 / (1.0 + torch.exp((val_p - midpoint) / 15.0))
            pred_threat = torch.where(val_p <= ALERT_RADIUS, torch.ones_like(pred_threat), pred_threat)
            pred_mask = (val_p < PERCEPTION_RADIUS).float()
            
            pred_phys = torch.stack([torch.cos(p_ang), torch.sin(p_ang), pred_threat], dim=2) * pred_mask.unsqueeze(-1)
            pred_in[:, :, :k_p] = pred_phys.transpose(1, 2)

        # --- 3. 處理隊友 (Team: Channel 6-8) ---
        team_in = torch.zeros((POP_SIZE, 3, MAX_OBJ), device=DEVICE)
        if POP_SIZE > 1:
            dist_agents = torch.cdist(self.pos, self.pos)
            dist_agents.fill_diagonal_(999.0)
            k_t = min(MAX_OBJ, POP_SIZE - 1)
            val_t, t_idx = torch.topk(dist_agents, k_t, largest=False)
            
            selected_team_pos = self.pos[t_idx]
            t_diff = selected_team_pos - self.pos.unsqueeze(1)
            t_ang = torch.atan2(t_diff[..., 1], t_diff[..., 0]) - self.angle.unsqueeze(1)
            
            team_presence = 0.1 * (1.0 - val_t / PERCEPTION_RADIUS).clamp(min=0)
            team_threat = torch.pow((1.0 - val_t / TEAM_RADIUS).clamp(min=0), 2)
            team_mask = (val_t < PERCEPTION_RADIUS).float()
            
            team_phys = torch.stack([torch.cos(t_ang), torch.sin(t_ang), torch.max(team_presence, team_threat)], dim=2) * team_mask.unsqueeze(-1)
            team_in[:, :, :k_t] = team_phys.transpose(1, 2)

        # --- 5. 合併與自身狀態 ---
        mixed_in = torch.cat([wall_in, food_in, team_in, pred_in], dim=1)
        
        speed = torch.norm(self.vel, dim=1) / self.max_speed
        last_steer = self.last_actions[:, 0]
        self_in = torch.stack([speed, last_steer, self.energy / MAX_ENERGY], dim=1)

        return (mixed_in, self_in)
    
    def update(self, move_food, move_predator):
        was_alive = self.alive.clone()
        current_states = self.get_states()
        
        # 取得連續動作並加入探索噪音
        with torch.no_grad():
            m, s = current_states
            actions = self.actor(m, s)
            noise = self.ou_noise.sample() * self.epsilon
            actions = torch.clamp(actions + noise, -1.0, 1.0)
            # 包含噪音的實際執行值。
            # 為了讓 Actor 的輸入與 Critic 的評估基準一致，給予「實際執行值」能讓神經網路更快學會「我的行為與環境變化」之間的關聯。
            self.last_actions = actions.detach().clone()

        rewards = torch.full((POP_SIZE,), STEP_REWARD, device=DEVICE)
            
        # 物理運算 (載具動力學)
        for i in range(POP_SIZE):
            if not self.alive[i]:
                continue
            
            speed_val = torch.norm(self.vel[i])
            steer_val = actions[i][0]     # -1~1
            throttle_val = actions[i][1]

            # 映射至 -0.15 到 0.15 弧度，舵效：速度越快轉向越明顯，但極速時轉向半徑應變大
            # 1. 基礎舵效：隨速度上升 (0 -> 2.0 速度區間)
            sensitivity = torch.clamp(speed_val / 2.0, min=0.0, max=1.0)
            # 2. 高速衰減：速度過高時，強行降低角速度以增大轉向半徑 (2.0 -> 4.0 速度區間)
            # 當速度從 2.0 升到 4.0，衰減係數從 1.0 降到 0.6
            high_speed_damping = torch.clamp(1.5 - (speed_val / self.max_speed), min=0.6, max=1.0)
            steer = steer_val * 0.15 * sensitivity * high_speed_damping
            self.angle[i] += steer

            # Action[1] 為油門 (-1 到 1 映射至 0 到 THROTTLE_FACTOR 加速度)
            throttle = throttle_val * (THROTTLE_FACTOR if throttle_val > 0 else THROTTLE_FACTOR / 3)
            # 向量動力學：計算當前車頭朝向單位向量
            forward_vec = torch.tensor([torch.cos(self.angle[i]), torch.sin(self.angle[i])], device=DEVICE)
            # 推力 = 向量 X 油門
            thrust = forward_vec * throttle
            self.vel[i] = self.vel[i] * DAMPING_FACTOR + thrust
            
            # 更新位置
            self.pos[i] += self.vel[i]
            # 能量消耗
            self.energy[i] -= ENERGY_DECAY * torch.abs(throttle_val)

            # --- 移動獎勵
            # 1. 計算「有效前進速度」：將實際速度向量投影到車頭方向
            forward_speed = torch.dot(self.vel[i], forward_vec)

            # 2. 基礎移動獎勵：改用 forward_speed
            move_reward = forward_speed * 0.125

            # 3. 嚴格的轉向懲罰
            steer_penalty = 0.05 * torch.pow(steer_val, 2) * (speed_val / self.max_speed)

            # 3. 靜止/低效懲罰 (Lazy Penalty)
            # 如果有效前進速度太低，就給予負分，逼它動起來
            lazy_penalty = torch.clamp(0.4 - forward_speed, min=0.0) * 0.2

            # 4. 高速與油門懲罰 (維持你原有的速度限制邏輯)
            throttle_penalty = 0.0
            throttle_threshold = 0.875
            if throttle_val > throttle_threshold:
                throttle_penalty = 0.5 * torch.pow((throttle_val - throttle_threshold) / (1.0 - throttle_threshold), 2)

            # 5. 最終移動獎勵整合
            # 計算方式：(有效前進 * 轉向效率) - 懶惰代價 - 超速代價
            rewards[i] += move_reward - steer_penalty - lazy_penalty - throttle_penalty

        # 邊界碰撞處理 (反彈)
        hit_w_x = (self.pos[:, 0] <= 0) | (self.pos[:, 0] >= SCREEN_W - 1)
        hit_w_y = (self.pos[:, 1] <= 0) | (self.pos[:, 1] >= SCREEN_H - 1)
        hit_wall = (hit_w_x | hit_w_y) & self.alive
        self.vel[hit_w_x, 0] *= -0.5
        self.vel[hit_w_y, 1] *= -0.5
        self.pos[:, 0] = torch.clamp(self.pos[:, 0], 0, SCREEN_W - 1)
        self.pos[:, 1] = torch.clamp(self.pos[:, 1], 0, SCREEN_H - 1)

        d_left = self.pos[:, 0]
        d_right = (SCREEN_W - 1) - self.pos[:, 0]
        d_top = self.pos[:, 1]
        d_bottom = (SCREEN_H - 1) - self.pos[:, 1]
        wall_dists = torch.stack([d_left, d_right, d_top, d_bottom], dim=1)
        wall_ratios = (1.0 - wall_dists / WALL_SENSE_RADIUS).clamp(min=0.0, max=1.0)
        wall_penalty = torch.sum(0.2 * wall_ratios, dim=1)
        rewards -= (wall_penalty * self.alive.float())

        # 更新掠食者 (Predators)
        if PREDATOR_SIZE > 0 and move_predator:
            self.pred_pos, self.pred_vel = self.update_entities(
                self.pred_pos, self.pred_vel, min_speed=PREDATOR_MIN_SPEED, max_speed=PREDATOR_MAX_SPEED
            )

        # 更新食物 (Food) - 假設食物也會移動
        if FOOD_SIZE > 0 and move_food:
            self.food_pos, self.food_vel = self.update_entities(
                self.food_pos, self.food_vel, min_speed=0.5, max_speed=1.0
            )
            
        # 食物碰撞
        dist_f = torch.cdist(self.pos, self.food_pos)
        hits_f = (dist_f < 10.0) & self.alive.unsqueeze(1)
        if hits_f.any():
            # 1. 建立遮罩距離矩陣：把「沒碰到」或「已死亡」的距離變成無限大 (inf)
            masked_dist = torch.where(hits_f, dist_f, torch.tensor(float('inf'), device=DEVICE))
            
            # 2. 沿著 Agent 維度 (dim=0) 找最小值
            # min_dists: 每個食物被碰到的最短距離 (沒被碰到的會是 inf)
            # closest_a_idx: 每個食物對應距離最近的 Agent 索引
            min_dists, closest_a_idx = torch.min(masked_dist, dim=0)
            
            # 3. 過濾出「真正有被吃到」的食物 (距離不是 inf 的)
            valid_eaten_mask = min_dists != float('inf')
            
            # 取出最終要結算的 食物索引 與 Agent 索引
            f_idx = torch.where(valid_eaten_mask)[0] 
            a_idx = closest_a_idx[valid_eaten_mask]
            
            # 4. 處理獎勵與能量 (安全累加)
            # 注意：a_idx 仍可能有重複 (如果同一個 Agent 技壓群雄，同時離 3 個食物最近)
            reward_increment = torch.full((len(a_idx),), FOOD_REWARD, device=DEVICE)
            rewards.index_add_(0, a_idx, reward_increment)
            energy_increment = torch.full((len(a_idx),), FOOD_ENERGY, device=DEVICE)
            self.energy.index_add_(0, a_idx, energy_increment)
            self.energy = torch.clamp(self.energy, max=MAX_ENERGY)
            
            # 5. 重生食物
            # 注意：這裡的 f_idx 保證是不重複的 (因為每個食物只會選出一個最近的 Agent)
            # 所以直接用 f_idx 即可，不需要再做 unique
            self.food_pos[f_idx] = self.get_risky_pos(len(f_idx))
            
        # 隊友排斥，排除自己與自己的距離 (對角線設為大值)
        if POP_SIZE > 1:
            dist_agents = torch.cdist(self.pos, self.pos).fill_diagonal_(999.0)
            # 如果靠太近，給予懲罰
            team_ratio = (1.0 - dist_agents / TEAM_RADIUS).clamp(min=0.0, max=1.0)
            team_penalty = torch.sum(torch.pow(team_ratio, 2), dim=1)
            rewards -= (team_penalty * self.alive.float())

        # 掠食者碰撞
        if PREDATOR_SIZE > 0:
            dist_p = torch.cdist(self.pos, self.pred_pos)
            danger_ratio = (1.0 - dist_p / ALERT_RADIUS).clamp(min=0.0, max=1.0)
            danger_penalty = torch.sum(torch.pow(danger_ratio, 2), dim=1)
            rewards -= (danger_penalty * self.alive.float())
            
            hit_pred = (dist_p < PREDATOR_CIRCLE).any(dim=1) & self.alive
        else:
            hit_pred = torch.zeros(POP_SIZE, dtype=torch.bool, device=DEVICE)

        starved = (self.energy <= 0) & self.alive

        dead_mask = hit_wall | hit_pred | starved
        if dead_mask.any():
            remaining_dead = dead_mask.clone()
            rewards[hit_pred] += PRED_REWARD
            remaining_dead &= ~hit_pred

            wall_mask = remaining_dead & hit_wall
            rewards[wall_mask] += WALL_REWARD
            remaining_dead &= ~wall_mask

            starve_mask = remaining_dead & starved
            rewards[starve_mask] += STARVED_REWARD

            self.alive[dead_mask] = False
            self.vel[dead_mask] = 0.0 # 死掉後速度歸零
            self.respawn_timer[dead_mask] = 1 if POP_SIZE == 1 else torch.randint(60, 360, (dead_mask.sum(),), device=DEVICE)
        
        next_states = self.get_states()
        
        # 把經驗推入 Replay Buffer
        for i in range(POP_SIZE):
            if not was_alive[i]: 
                continue # 忽略死屍的經驗
            self.memory.push(
                current_states[0][i],
                current_states[1][i],
                actions[i], 
                rewards[i], 
                next_states[0][i], 
                next_states[1][i], 
                dead_mask[i]
            )

        if self.steps % 2 == 0:
            self.optimize_model()

        ready_to_respawn = ~self.alive & (self.respawn_timer <= 0)
        if ready_to_respawn.any():
            indices = torch.where(ready_to_respawn)[0]
            self.pos[indices] = self.get_saftest_pos(len(indices))
            self.alive[indices] = True
            self.energy[indices] = MAX_ENERGY
            self.vel[indices] = 0.0

        self.respawn_timer[~self.alive] -= 1
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        self.steps += 1
        self.rewards_avg = self.rewards_avg * 0.99 + (rewards.sum().item() / POP_SIZE) * 0.01
        self.hit_pred += hit_pred.sum().item()
        self.hit_wall += hit_wall.sum().item()
        self.starved += starved.sum().item()
        self.rewards = rewards.detach().cpu().numpy()

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        m_b, s_b, a_b, r_b, nm_b, ns_b, d_b = self.memory.sample(BATCH_SIZE)

        # --- 更新 Critic ---
        with torch.no_grad():
            next_actions = self.actor_target(nm_b, ns_b)
            target_q = self.critic_target(nm_b, ns_b, next_actions).squeeze(1)
            target_value = r_b + (GAMMA * target_q * (1 - d_b.float()))
            
        current_q = self.critic(m_b, s_b, a_b).squeeze(1)
        critic_loss = F.mse_loss(current_q, target_value)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        self.c_loss_val = critic_loss.item()

        # --- 更新 Actor ---
        predicted_actions = self.actor(m_b, s_b)
        actor_loss = -self.critic(m_b, s_b, predicted_actions).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        self.a_loss_val = actor_loss.item()

        # --- 軟更新 Target 網路 (Soft Update) ---
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

    def get_saftest_pos(self, n):
        """
        一次為 n 個實體尋找安全位置。
        n: 需要生成的點數量。
        """
        pad = 10.0
        # 候選樣本數
        num_samples = max(n * 10, PREDATOR_SIZE)
        samples = torch.empty((num_samples, 2), device=DEVICE)
        samples[:, 0].uniform_(pad, SCREEN_W - pad)
        samples[:, 1].uniform_(pad, SCREEN_H - pad)

        # 如果有掠食者，進行距離篩選
        if PREDATOR_SIZE > 0:
            # 計算矩陣：(候選點數, 掠食者數)
            dists = torch.cdist(samples, self.pred_pos)
            
            # 找到每個候選點離「最近掠食者」的距離
            min_dists = dists.min(dim=1).values
            
            # 從中選出最遠（最安全）的前 n 個點
            _, top_indices = torch.topk(min_dists, k=n)
            
            # 如果需要的 n 比 top_indices 多（雖然機率極低），用 repeat 補齊
            return samples[top_indices]
        else:
            # 無掠食者時，直接隨機回傳 n 個點
            return samples[:n]

    def get_risky_pos(self, n):
        """
        找尋離掠食者最近，但距離至少大於 min_threshold 的位置。
        """
        if n <= 0: return torch.empty((0, 2), device=DEVICE)

        pad = 10.0
        # 增加樣本數
        num_samples = max(n * 20, PREDATOR_SIZE) 
        samples = torch.empty((num_samples, 2), device=DEVICE)
        samples[:, 0].uniform_(pad, SCREEN_W - pad)
        samples[:, 1].uniform_(pad, SCREEN_H - pad)

        if PREDATOR_SIZE:
            # 1. 計算所有樣本到掠食者的最近距離
            dists = torch.cdist(samples, self.pred_pos)
            min_dists = dists.min(dim=1).values
            
            # 2. 建立遮罩：找出符合「距離 > 50」條件的點
            valid_mask = min_dists >= 50.0

            # 3. 處理符合條件的樣本
            if valid_mask.any():
                valid_min_dists = min_dists[valid_mask]
                valid_samples = samples[valid_mask]
                
                # 4. 反向找「最近」: largest=False
                # 這會回傳 valid_min_dists 中數值最小的前 k 個
                _, top_indices = torch.topk(valid_min_dists, k=n, largest=False)
                return valid_samples[top_indices]
            else:
                # 如果全部點都離掠食者太近 (極端情況)，退而求其次回傳最遠的點
                _, top_indices = torch.topk(min_dists, k=n, largest=True)
                return samples[top_indices]
        else:
            # 無掠食者時，隨機回傳
            return samples[:n]
        
    def update_entities(self, pos, vel, min_speed, max_speed, jitter_chance=0.05):
        # 1. 向量化隨機擾動 (使用雜湊式隨機避免 if 分支)
        change_mask = torch.rand(pos.shape[0], 1, device=DEVICE) < jitter_chance
        # 直接在原位修改 vel，減少記憶體配置
        vel.add_((torch.rand_like(vel) - 0.5) * (1.8 * change_mask))

        # 2. 速度約束 (In-place 操作優化)
        speeds = torch.norm(vel, dim=1, keepdim=True)
        # 使用 clamp_ 進行原位限制，並處理除以零
        new_speeds = torch.clamp(speeds, min_speed, max_speed)
        vel.mul_(new_speeds / (speeds + 1e-6))

        # 3. 更新位置
        pos.add_(vel)

        # 4. 高效邊界處理 (使用 torch.abs_ 搭配 clamp_)
        # 檢查是否超出邊界：若超出，速度反向
        # 邏輯：如果 pos < 0 或 pos > bounds，速度就乘上 -1
        out_of_bounds = (pos < 0) | (pos > self.bounds)
        vel[out_of_bounds] *= -1.0
        
        # 將位置限制在畫布內
        pos.clamp_(torch.zeros_like(self.bounds), self.bounds)

        return pos, vel

    def respawn_food(self, f_idx, a_idx):
        num_to_respawn = len(f_idx)
        pad = 10.0  # 避免生在牆縫裡，Agent 很難吃到
        
        # 1. 為每個需要重生的食物生成一組候選點 (假設每個食物給 10 個候選)
        num_samples = 10
        candidates = torch.empty((num_to_respawn, num_samples, 2), device=DEVICE)
        candidates[..., 0].uniform_(pad, SCREEN_W - pad)
        candidates[..., 1].uniform_(pad, SCREEN_H - pad)

        # 2. 獲取對應 Agent 的位置 (假設每個食物索引 f_idx 對應一個 Agent)
        # 這裡需要確保你的 f_idx 邏輯能對應到正確的 Agent 位置
        # 假設是一對一，或者我們直接取 Agent 的平均位置作為避開點
        target_agent_pos = self.pos[a_idx].unsqueeze(1) # (num_res, 1, 2)

        # 3. 計算每個候選點到 Agent 的距離
        dists = torch.norm(candidates - target_agent_pos, dim=2) # (num_res, num_samples)

        # 4. 篩選策略：找距離在 [PERCEPTION_RADIUS, Max] 之間的
        # 我們先找出每個食物最遠的候選點作為保底
        farthest_idx = torch.argmax(dists, dim=1)
        
        # 建立一個 Mask，過濾掉太近的 (小於 200)
        valid_mask = dists > PERCEPTION_RADIUS
        
        final_pos = torch.empty((num_to_respawn, 2), device=DEVICE)
        
        for i in range(num_to_respawn):
            valid_indices = torch.where(valid_mask[i])[0]
            if len(valid_indices) > 0:
                # 在合格的點中隨機選一個 (增加多樣性)
                sel_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))]
                final_pos[i] = candidates[i, sel_idx]
            else:
                # 如果都太近，就取最遠的那個
                final_pos[i] = candidates[i, farthest_idx[i]]

        return final_pos
        
    def save_state(self):
        torch.save({
            'pos': self.pos,
            'last_actions': self.last_actions,
            'energy': self.energy,
            'alive': self.alive,
            'respawn_timer': self.respawn_timer,
            'food_pos': self.food_pos,
            'pred_pos': self.pred_pos,
            'rewards_avg': self.rewards_avg,
            'hit_pred': self.hit_pred,
            'hit_wall': self.hit_wall,
            'starved': self.starved
        }, SAVE_PATH)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'steps': self.steps,
            'eps': self.epsilon
        }, self.brain_path)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}][Saved] Steps: {self.steps}, A-Loss: {self.a_loss_val:.4f}, C-Loss: {self.c_loss_val:.4f}, Rewards-Avg: {self.rewards_avg:.4f}, Hit_Pred: {self.hit_pred}, Hit_Wall: {self.hit_wall}, Starved: {self.starved}.")

    def load_state(self):
        if os.path.exists(SAVE_PATH):
            try:
                state = torch.load(SAVE_PATH, map_location=DEVICE)
                self.pos = state['pos']
                self.last_actions = state['last_actions']
                self.energy = state['energy']
                self.alive = state['alive']
                self.respawn_timer = state['respawn_timer']
                self.food_pos = state['food_pos']
                self.pred_pos = state['pred_pos']
                self.rewards_avg = state['rewards_avg']
                self.hit_pred = state['hit_pred']
                self.hit_wall = state['hit_wall']
                self.starved = state['starved']
                print(f"--- [Loaded] Load completed ---")
            except Exception as e:
                print(f"--- [Error] Loading failed: {e} ---")

        if os.path.exists(self.brain_path):
            try:
                brain_state = torch.load(self.brain_path, map_location=DEVICE)
                self.actor.load_state_dict(brain_state['actor'])
                self.actor_target.load_state_dict(self.actor.state_dict())
                self.critic.load_state_dict(brain_state['critic'])
                self.critic_target.load_state_dict(self.critic.state_dict())
                self.steps = brain_state['steps']
                self.epsilon = brain_state['eps']
                print(f"--- [Loaded] brain weights {self.brain_path}, steps {self.steps:,} ---")
                return True
            except Exception as e:
                print(f"--- [Error] brain weights loading failed: {e} ---")


    def draw(self, label_only, draw_perception, draw_alert, verbose):
        self.screen.fill((20, 20, 25))

        if not label_only:
            for f in self.food_pos.cpu().numpy():
                pygame.draw.circle(self.screen, (0, 255, 120), f.astype(int), 3)

            for p in self.pred_pos.cpu().numpy(): 
                pygame.draw.circle(self.screen, (255, 50, 50), p.astype(int), PREDATOR_CIRCLE, 1)
                pygame.draw.circle(self.screen, (255, 50, 50), p.astype(int), PREDATOR_RADIUS)

            p_np = self.pos.cpu().numpy()
            a_np = self.alive.cpu().numpy()
            e_np = self.energy.cpu().numpy()
            t_np = self.respawn_timer.cpu().numpy()
            act_np = self.last_actions.cpu().numpy()
            vel_np = self.vel.cpu().numpy()
            ang_np = self.angle.cpu().numpy()

            for i, p in enumerate(p_np):
                pos_tuple = (int(p[0]), int(p[1]))

                if not a_np[i]:
                    dead_color = (60, 60, 60) if e_np[i] <= 0 else (120, 0, 0)
                    pygame.draw.circle(self.screen, dead_color, pos_tuple, 3)
                    timer_surface = self.font.render(f"{t_np[i]:.0f}", True, dead_color)
                    timer_rect = timer_surface.get_rect(center=(pos_tuple[0], pos_tuple[1] - 15))
                    self.screen.blit(timer_surface, timer_rect)
                    continue

                en_ratio = e_np[i] / MAX_ENERGY
                r = int(255 * en_ratio)          # 越飽越紅
                g = int(128 * en_ratio)          # 飽的時候帶點橘色感，不飽就變暗
                b = int(255 * (1 - en_ratio))    # 越餓越藍
                color = (r, g, b)
                radius = int(4 + 4 * en_ratio)
                pygame.draw.circle(self.screen, color, pos_tuple, radius)

                act = act_np[i]
                throttle = act[1]
                vel = vel_np[i]
                speed = np.linalg.norm(vel)

                # 顯示能量數值
                if verbose >= 1:
                    text_surface = self.font.render(f"{e_np[i]:.0f}", True, (255, 255, 255))
                    text_rect = text_surface.get_rect(center=(pos_tuple[0], pos_tuple[1] - radius - 8))
                    self.screen.blit(text_surface, text_rect)

                # 顯示除錯訊息
                if verbose >= 2:
                    dbg_text = f"{throttle:.2f} {self.rewards[i]:.4f}"
                    dbg_surface = self.font.render(dbg_text, True, (255, 255, 255))
                    dbg_rect = dbg_surface.get_rect(center=(pos_tuple[0], pos_tuple[1] + radius + 8))
                    self.screen.blit(dbg_surface, dbg_rect)

                # 繪製慣性方向
                if speed > 0:
                    v_line_length = 2 + speed * 1.1
                    # 終點座標 = 當前位置 + 速度向量 * 縮放係數
                    vel_end_p = (
                        int(p[0] + vel[0] * v_line_length), 
                        int(p[1] + vel[1] * v_line_length)
                    )
                    inertia_color = (0, 191, 255) # 深天藍色
                    pygame.draw.line(self.screen, inertia_color, pos_tuple, vel_end_p, 3)
                    
                # 繪製方向及油門指示線
                if throttle != 0:
                    power = abs(throttle)
                    angle = ang_np[i]
                    if throttle > 0:
                        line_width = 1
                        if power <= 0.5:
                            line_color = (0, 255, 0)
                        elif power <= 0.875:
                            line_color = (255, 255, 255)
                        else:
                            line_color = (255, 0, 0)
                    else:
                        power /= 3
                        angle += np.pi
                        line_width = 5
                        line_color = (255, 255, 0)

                    line_length = 25 * power
                    end_p = (
                        int(p[0] + np.cos(angle) * line_length),
                        int(p[1] + np.sin(angle) * line_length)
                    )
                    pygame.draw.line(self.screen, line_color, pos_tuple, end_p, line_width)

                # 畫出視野範圍
                if draw_perception:
                    pygame.draw.circle(self.screen, color, pos_tuple, PERCEPTION_RADIUS, 1)
                # 畫出警戒範圍
                if draw_alert:
                    pygame.draw.circle(self.screen, color, pos_tuple, ALERT_RADIUS, 1)
                    pygame.draw.circle(self.screen, color, pos_tuple, WALL_SENSE_RADIUS, 1)
                    pygame.draw.circle(self.screen, color, pos_tuple, TEAM_RADIUS, 1)

        THEME = {
            "label": (180, 180, 180),    # 淺灰色 (標籤專用)
            "perf": (100, 220, 180),     # 薄荷綠 (性能指標)
            "param": (255, 200, 100),    # 亮橘黃 (關鍵參數)
            "loss": (255, 120, 120),     # 柔和紅 (損失/負面指標)
            "success": (120, 255, 120)   # 翠綠色 (獎勵/存活)
        }
        ui_labels = [
            ("Steps:", f"{self.steps:,}", THEME["perf"], True),
            ("FPS:", f"{int(self.clock.get_fps())}", THEME["perf"], False),
            ("Epsilon:", f"{self.epsilon:.4f}", THEME["param"], False),
            ("A-Loss:", f"{self.a_loss_val:.3f}", THEME["loss"], False),
            ("C-Loss:", f"{self.c_loss_val:.3f}", THEME["loss"], False),
            ("Rewards:", f"{self.rewards_avg:.3f}", THEME["success"] if self.rewards_avg > 0 else (255, 0, 0), False),
            ("Hit Pred:", f"{self.hit_pred:,}", THEME["loss"], False),
            ("Hit Wall:", f"{self.hit_wall:,}", THEME["loss"], False),
            ("Starved:", f"{self.starved:,}", THEME["loss"], False),
            ("Alive:", f"{int(self.alive.sum())}/{POP_SIZE}", THEME["success"], False)
        ]
        current_y = 10     # 起始高度
        col_1_x = 10   # Label 開始位置
        col_2_x = 200  # Value 結束位置 (右對齊線)
        for i, (text, val, val_color, bold) in enumerate(ui_labels):
            font = self.big_font if bold else self.font
            
            lbl_surf = font.render(text, True, THEME["label"])
            lbl_rect = lbl_surf.get_rect(topleft=(col_1_x, current_y))
            val_surf = font.render(val, True, val_color)
            val_rect = val_surf.get_rect(topright=(col_2_x, current_y))
            
            self.screen.blit(lbl_surf, lbl_rect)
            self.screen.blit(val_surf, val_rect)

            line_height = max(lbl_rect.height, val_rect.height)
            current_y += line_height + 4

        pygame.display.flip()
        self.clock.tick(self.fps)

    def run(self):
        running = True
        is_paused = False
        label_only = False
        draw_alert = False
        draw_perception = False
        move_food = False
        move_predator = True
        verbose = 0

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_UP:
                        self.fps += 5
                        self.update_caption()
                    elif event.key == pygame.K_DOWN:
                        self.fps -= 5
                        self.update_caption()
                    elif event.key == pygame.K_EQUALS:
                        self.fps = 240
                        self.update_caption()
                    elif event.key == pygame.K_MINUS:
                        self.fps = 5
                        self.update_caption()
                    elif event.key == pygame.K_r:
                        self.reset_env()
                    # 檢查 大寫 R (Shift + R)
                    if event.key == pygame.K_r and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                        self.init_network()
                        self.steps = 0
                        self.rewards_avg = 0.0
                        self.hit_pred = 0
                        self.hit_wall = 0
                        self.starved = 0
                    elif event.key == pygame.K_SPACE:
                        is_paused = not is_paused
                    elif event.key == pygame.K_h:
                        label_only = not label_only
                    elif event.key == pygame.K_a:
                        draw_alert = not draw_alert
                    elif event.key == pygame.K_p:
                        draw_perception = not draw_perception
                    elif event.key == pygame.K_f:
                        move_food = not move_food
                    elif event.key == pygame.K_m:
                        move_predator = not move_predator
                    elif event.key == pygame.K_v:
                        verbose = (verbose + 1) % 3

            if not is_paused:
                self.update(move_food, move_predator)
                if self.steps % 5000 == 0:
                    self.save_state()

            self.draw(label_only, draw_perception, draw_alert, verbose)

        self.save_state()
        pygame.quit()

if __name__ == "__main__":
    sim = RLSimulation()
    sim.run()
