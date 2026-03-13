import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path

script_name = Path(__file__).stem
CAPTION = "Vectra: Apex Protocol"
# --- 參數設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCREEN_W, SCREEN_H = 900, 700
SAVE_PATH = f"{script_name}.pt"

# 環境參數
POP_SIZE = 16
FOOD_SIZE = 32
PREDATOR_SIZE = 8
MAX_ENERGY = 100.0
FOOD_ENERGY = 10.0
ENERGY_DECAY = 0.035
PERCEPTION_RADIUS = 200 # 視野感知半徑
ALERT_RADIUS = 100      # 敵方懲罰半徑
TEAM_RADIUS = 20        # 友方懲罰半徑

# DDPG 核心參數
GAMMA = 0.98
TAU = 0.005 # 軟更新係數
LR_ACTOR = 0.003
LR_CRITIC = 0.001
MEMORY_SIZE = 500000
BATCH_SIZE = 256
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9995
MAX_OBJ = 20 # 最大環境物件數量
FEAT_DIM = 7 # 每個物件特微 [cos, sin, score, 牆=1, 隊友=1, 食物=1, 敵人=1]

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
        x = m_in.transpose(1, 2)
        feat = F.relu(self.conv(x))
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
        x = m_in.transpose(1, 2)
        feat = F.relu(self.conv(x))
        
        # 關鍵差異：同時保留最強威脅與整體分佈
        feat_max = torch.max(feat, dim=2)[0]
        feat_avg = torch.mean(feat, dim=2)

        combined = torch.cat([feat_max, feat_avg, s_in], dim=1)
        return self.fc(combined)

# 注意機制型
class ActorAttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.obj_enc = nn.Linear(FEAT_DIM, 32)
        self.self_enc = nn.Linear(3, 32)

        # 使用 MultiheadAttention 讓模型可以同時關注多種目標
        self.mha = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(32 + 3, 64), # 32(全局特徵) + 3(自身狀態 s_in)
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
        # 1. 編碼物件與自身狀態
        obj_feat = F.relu(self.obj_enc(m_in))   # [Batch, N, 32]
        query = self.self_enc(s_in).unsqueeze(1) # [Batch, 1, 32]
        
        # 2. 以「自身狀態」為 Query，去物件中找尋資訊
        # attn_out 是加權後的全局特徵
        attn_out, _ = self.mha(query, obj_feat, obj_feat) 
        attn_out = attn_out.squeeze(1) # [Batch, 32]
        
        # 3. 結合並輸出
        combined = torch.cat([attn_out, s_in], dim=1)
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
        """
        m_in: [Batch, N, FEAT_DIM]  (N 個物體)
        s_in: [Batch, 3]     (自身狀態)
        action: [Batch, 2]   (要評估的動作)
        """
        # 1. 提取物體特徵: [Batch, FEAT_DIM, N] -> [Batch, 32, N]
        x = m_in.transpose(1, 2)
        feat = F.relu(self.conv(x))
        
        # 2. Attention 權重計算: [Batch, N, 32]
        feat = feat.transpose(1, 2)
        weights = F.softmax(self.attn_weights(feat), dim=1)
        
        # 3. 融合為全局特徵: [Batch, 32]
        x_attn = torch.sum(feat * weights, dim=1)
        
        # 4. 結合所有資訊: [Batch, 32 + 3 + 2] = [Batch, 37]
        combined = torch.cat([x_attn, s_in, action], dim=1)
        return self.fc(combined)
    
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        
        self.m_states = torch.zeros((capacity, MAX_OBJ, FEAT_DIM), device=DEVICE)
        self.s_states = torch.zeros((capacity, 3), device=DEVICE)
        self.next_m_states = torch.zeros((capacity, MAX_OBJ, FEAT_DIM), device=DEVICE)
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
        self.brain_path = f"{self.actor.__class__.__name__}.pt"
        
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.ou_noise = OUNoise(2) # 2 個動作維度
        self.epsilon = EPSILON_START
        self.steps = 0
        self.a_loss_val = 0.0
        self.c_loss_val = 0.0
        self.rewards_avg = 0.0
        self.hit_pred = 0
        self.hit_wall = 0
        self.starved = 0

        # 對應維度: [牆, 隊友, 食物, 敵人]
        self.tag_wall = torch.tensor([1, 0, 0, 0], device=DEVICE).view(1, 1, 4)
        self.tag_team = torch.tensor([0, 1, 0, 0], device=DEVICE).view(1, 1, 4)
        self.tag_food = torch.tensor([0, 0, 1, 0], device=DEVICE).view(1, 1, 4)
        self.tag_pred = torch.tensor([0, 0, 0, 1], device=DEVICE).view(1, 1, 4)

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

        print(f"[{self.actor.__class__.__name__}] Network & Optimizer init complete.")
        
    def reset_env(self):
        self.last_actions = torch.zeros((POP_SIZE, 2), device=DEVICE)
        self.vel = torch.zeros((POP_SIZE, 2), device=DEVICE)
        self.angle = torch.rand(POP_SIZE, device=DEVICE) * (2 * np.pi)        
        self.energy = torch.full((POP_SIZE,), MAX_ENERGY, device=DEVICE, dtype=torch.float)
        self.alive = torch.ones(POP_SIZE, dtype=torch.bool, device=DEVICE)
        self.respawn_timer = torch.zeros(POP_SIZE, dtype=torch.long, device=DEVICE) 
        self.screen_size = torch.tensor([SCREEN_W, SCREEN_H], device=DEVICE, dtype=torch.float)
        self.pos = torch.rand(POP_SIZE, 2, device=DEVICE) * self.screen_size
        self.food_pos = torch.rand(FOOD_SIZE, 2, device=DEVICE) * self.screen_size
        self.food_vel = (torch.rand(FOOD_SIZE, 2, device=DEVICE) - 0.5) * 3.5
        self.pred_pos = torch.rand(PREDATOR_SIZE, 2, device=DEVICE) * self.screen_size
        self.pred_vel = (torch.rand(PREDATOR_SIZE, 2, device=DEVICE) - 0.5) * 3.5

    def get_states(self):
        """
        將環境觀察值轉換為神經網路輸入張量。
        輸出:
            mixed_in: [Batch, MAX_OBJ, FEAT_DIM] -> 環境物體特徵
            self_in:  [Batch, 3]                 -> 自身狀態特徵
        """
        # --- 1. 預分配輸出的特徵池 (固定 Shape [Batch, MAX_OBJ, 7]) ---
        # 預先填充 0，這樣就自動處理了 Padding 邏輯
        mixed_in = torch.zeros((POP_SIZE, MAX_OBJ, 7), device=DEVICE, dtype=torch.float32)
        # 用於追踪當前 buffer 填到了哪個位置
        current_idx = 0

        # ---------------------------------------------------------
        # 定義內部函數：將處理好的特徵填入 mixed_in
        # ---------------------------------------------------------
        def fill_buffer(features):
            num_new = features.shape[1]
            space_left = MAX_OBJ - current_idx
            if space_left <= 0:
                return current_idx
            
            actual_add = min(num_new, space_left)
            mixed_in[:, current_idx : current_idx + actual_add, :] = features[:, :actual_add, :]
            return current_idx + actual_add

        # --- 2. 處理牆壁 (邊界感應) ---
        d_left = self.pos[:, 0]
        d_right = SCREEN_W - self.pos[:, 0]
        d_top = self.pos[:, 1]
        d_bottom = SCREEN_H - self.pos[:, 1]
        wall_dists = torch.stack([d_left, d_right, d_top, d_bottom], dim=1) 
        
        # 僅保留半徑內的牆壁資訊
        wall_mask = (wall_dists < PERCEPTION_RADIUS).float()
        wall_angles = torch.atan2(self.wall_normals[:, 1], self.wall_normals[:, 0]).unsqueeze(0) - self.angle.unsqueeze(1)
        # 評分機制：越近分數越高 (1.0~0.0)，範圍外強制為 0
        wall_scores = (1.0 - wall_dists / PERCEPTION_RADIUS).clamp(0, 1.0) * wall_mask
        
        wall_phys = torch.stack([torch.cos(wall_angles), torch.sin(wall_angles), wall_scores], dim=2)
        # 組合物理特徵與 Tag [Batch, 4, 7]
        wall_in = torch.cat([wall_phys, self.tag_wall.expand(POP_SIZE, 4, 4)], dim=2)
        wall_in = wall_in * wall_mask.unsqueeze(-1) # 遮罩掉範圍外的
        
        current_idx = fill_buffer(wall_in)

        # --- 3. 處理食物 (資源感應) ---
        if current_idx < MAX_OBJ:
            dist_food = torch.cdist(self.pos, self.food_pos)
            food_mask = (dist_food < PERCEPTION_RADIUS).float()
            
            f_diff = self.food_pos.unsqueeze(0) - self.pos.unsqueeze(1)
            f_ang = torch.atan2(f_diff[..., 1], f_diff[..., 0]) - self.angle.unsqueeze(1)
            food_scores = torch.pow((1.0 - dist_food / PERCEPTION_RADIUS).clamp(0, 1.0), 2) * food_mask
            
            food_phys = torch.stack([torch.cos(f_ang), torch.sin(f_ang), food_scores], dim=2)
            food_in = torch.cat([food_phys, self.tag_food.expand(POP_SIZE, food_phys.shape[1], 4)], dim=2)
            food_in = food_in * food_mask.unsqueeze(-1)
            
            current_idx = fill_buffer(food_in)

        # --- 4. 處理隊友 (群體感應) ---
        if current_idx < MAX_OBJ and POP_SIZE > 1:
            dist_agents = torch.cdist(self.pos, self.pos)
            dist_agents.fill_diagonal_(999.0)
            team_mask = (dist_agents < PERCEPTION_RADIUS).float()
            
            t_diff = self.pos.unsqueeze(0) - self.pos.unsqueeze(1)
            t_ang = torch.atan2(t_diff[..., 1], t_diff[..., 0]) - self.angle.unsqueeze(1)
            team_scores = (1.0 - dist_agents / PERCEPTION_RADIUS).clamp(0, 1.0) * team_mask
            
            team_phys = torch.stack([torch.cos(t_ang), torch.sin(t_ang), team_scores], dim=2)
            team_in = torch.cat([team_phys, self.tag_team.expand(POP_SIZE, team_phys.shape[1], 4)], dim=2)
            team_in = team_in * team_mask.unsqueeze(-1)
            
            current_idx = fill_buffer(team_in)

        # --- 5. 處理敵人 (威脅感應) ---
        if current_idx < MAX_OBJ and self.pred_pos.shape[0] > 0:
            dist_pred = torch.cdist(self.pos, self.pred_pos)
            pred_mask = (dist_pred < PERCEPTION_RADIUS).float()
            
            p_diff = self.pred_pos.unsqueeze(0) - self.pos.unsqueeze(1)
            p_ang = torch.atan2(p_diff[..., 1], p_diff[..., 0]) - self.angle.unsqueeze(1)
            pred_scores = (1.0 - dist_pred / PERCEPTION_RADIUS).clamp(0, 1.0) * pred_mask * 1.5
            
            pred_phys = torch.stack([torch.cos(p_ang), torch.sin(p_ang), pred_scores], dim=2)
            pred_in = torch.cat([pred_phys, self.tag_pred.expand(POP_SIZE, pred_phys.shape[1], 4)], dim=2)
            pred_in = pred_in * pred_mask.unsqueeze(-1)
            
            current_idx = fill_buffer(pred_in)

        # --- 7. 處理自身狀態 (Self State) ---
        # 包含：標準化後的速率、最後一次轉向動作、剩餘能量比例
        speed = torch.norm(self.vel, dim=1) / 4.0
        last_steer = self.last_actions[:, 0]
        energy_ratio = self.energy / MAX_ENERGY
        self_in = torch.stack([speed, last_steer, energy_ratio], dim=1)

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

        rewards = torch.full((POP_SIZE,), -0.01, device=DEVICE)
            
        # 物理運算 (載具動力學)
        for i in range(POP_SIZE):
            if not self.alive[i]:
                continue
            
            speed_val = torch.norm(self.vel[i])
            steer_val = actions[i][0]     # -1 ~ 1
            throttle_val = actions[i][1]  # -1 ~ 1

            # 映射至 -0.15 到 0.15 弧度，舵效：速度越快轉向越明顯，但極速時轉向半徑應變大
            steer = steer_val * 0.15 * torch.clamp(speed_val / 2.0, 0, 1.0) 
            self.angle[i] += steer

            # Action[1] 為油門 (-1 到 1 映射至 0 到 0.3 加速度)
            throttle = (throttle_val + 1.0) * 0.3
            # 向量動力學：計算當前車頭朝向單位向量
            forward_vec = torch.tensor([torch.cos(self.angle[i]), torch.sin(self.angle[i])], device=DEVICE)
            # 推力 = 向量 X 油門
            thrust = forward_vec * throttle
            self.vel[i] = self.vel[i] * 0.85 + thrust
            
            # 更新位置
            self.pos[i] += self.vel[i]

            # --- 移動獎勵
            # 1. 計算「有效前進速度」：將實際速度向量投影到車頭方向
            # 這是最重要的指標：只有往正前方跑才給高分
            forward_speed = torch.dot(self.vel[i], forward_vec)

            # 2. 基礎移動獎勵：改用 forward_speed
            # 只有 forward_speed > 0 時才給分，徹底封殺原地打轉和倒車刷分
            move_reward = torch.clamp(forward_speed, min=0.0) * 0.05 * torch.pow(1.0 - torch.abs(steer_val), 2)

            # 3. 嚴格的轉向懲罰：使用平方係數
            # 當 steer_val 為 0 時，效率 1.0；當 steer_val 為 1.0 時，效率直接降為 0
            # 這樣 AI 發現大角度轉向會讓 move_gain 歸零，它會學會非必要不打死舵
            steer_efficiency = torch.pow(1.0 - torch.abs(steer_val), 2)

            # 3. 靜止/低效懲罰 (Lazy Penalty)
            # 如果有效前進速度太低，就給予負分，逼它動起來
            lazy_penalty = -0.15 if forward_speed < 0.4 else 0.0

            # 4. 高速與油門懲罰 (維持你原有的速度限制邏輯)
            throttle_penalty = 0.0
            if speed_val > 3.5:
                throttle_penalty = 0.6 * torch.pow((speed_val - 3.5) / 0.5, 2)

            # 5. 最終移動獎勵整合
            # 計算方式：(有效前進 * 轉向效率) - 懶惰代價 - 超速代價
            rewards[i] += (move_reward * steer_efficiency) + lazy_penalty - throttle_penalty

            # 額外小技巧：鼓勵「踩油門」與「方向正」的組合
            if throttle_val > 0.5 and torch.abs(steer_val) < 0.1:
                rewards[i] += 0.05 # 這是「直線衝刺」額外獎勵

        # 邊界碰撞處理 (反彈)
        hit_w_x = (self.pos[:, 0] <= 0) | (self.pos[:, 0] >= SCREEN_W - 1)
        hit_w_y = (self.pos[:, 1] <= 0) | (self.pos[:, 1] >= SCREEN_H - 1)
        hit_wall = (hit_w_x | hit_w_y) & self.alive

        d_left = self.pos[:, 0]
        d_right = SCREEN_W - self.pos[:, 0]
        d_top = self.pos[:, 1]
        d_bottom = SCREEN_H - self.pos[:, 1]
        wall_dists = torch.stack([d_left, d_right, d_top, d_bottom], dim=1)
        wall_ratio_all = (1.0 - wall_dists / PERCEPTION_RADIUS).clamp(0, 1.0)
        wall_ratio, _ = torch.max(wall_ratio_all, dim=1)
        wall_penalty = 4.0 * torch.pow(wall_ratio, 2)
        rewards -= (wall_penalty * self.alive.float())

        # 隊友排斥，排除自己與自己的距離 (對角線設為大值)
        if POP_SIZE > 1:
            dist_agents = torch.cdist(self.pos, self.pos).fill_diagonal_(999.0)
            min_dist_to_friend, _ = torch.min(dist_agents, dim=1)
            # 如果靠太近，給予懲罰
            team_mask = (min_dist_to_friend < TEAM_RADIUS) & self.alive
            rewards[team_mask] -= 0.1

        # 更新掠食者 (Predators)
        if PREDATOR_SIZE > 0 and move_predator:
            self.pred_pos, self.pred_vel = self.update_entities(
                self.pred_pos, self.pred_vel, min_speed=1.5, max_speed=3.5
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
            reward_increment = torch.full((len(a_idx),), 10.0, device=DEVICE)
            rewards.index_add_(0, a_idx, reward_increment)
            energy_increment = torch.full((len(a_idx),), FOOD_ENERGY, device=DEVICE)
            self.energy.index_add_(0, a_idx, energy_increment)
            self.energy = torch.clamp(self.energy, max=MAX_ENERGY)
            
            # 5. 重生食物
            # 注意：這裡的 f_idx 保證是不重複的 (因為每個食物只會選出一個最近的 Agent)
            # 所以直接用 f_idx 即可，不需要再做 unique
            self.food_pos[f_idx] = torch.rand(len(f_idx), 2, device=DEVICE) * self.screen_size
            
        # 掠食者碰撞
        if PREDATOR_SIZE > 0:
            dist_p = torch.cdist(self.pos, self.pred_pos)
            min_dist_p, _ = torch.min(dist_p, dim=1)
            # 只有還活著且進入警戒區的才扣分
            danger_mask = (min_dist_p < ALERT_RADIUS) & self.alive
            if danger_mask.any():
                # 懲罰函數：距離越近扣越多
                danger_ratio = 1.0 - min_dist_p[danger_mask] / ALERT_RADIUS
                danger_penalty = 8.0 * torch.pow(danger_ratio, 3)
                rewards[danger_mask] -= danger_penalty

            hit_pred = (dist_p < 22.0).any(dim=1) & self.alive
        else:
            hit_pred = torch.zeros(POP_SIZE, dtype=torch.bool, device=DEVICE)

        self.energy[self.alive] -= ENERGY_DECAY
        starved = (self.energy <= 0) & self.alive

        dead_mask = hit_wall | hit_pred | starved
        if dead_mask.any():
            rewards[dead_mask] -= 15
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
            for idx in indices:
                safe_pos = self.get_safest_pos()
                self.pos[idx] = safe_pos
                self.alive[idx] = True
                self.energy[idx] = MAX_ENERGY
                self.vel[idx] = 0.0

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

    def get_safest_pos(self):
        pad = 10.0
        samples = torch.empty((5, 2), device=DEVICE)
        samples[:, 0].uniform_(pad, SCREEN_W - pad)
        samples[:, 1].uniform_(pad, SCREEN_H - pad)

        if PREDATOR_SIZE > 0:
            dists = torch.cdist(samples, self.pred_pos)
            min_dists = torch.min(dists, dim=1).values
            best_idx = torch.argmax(min_dists)
            return samples[best_idx]
        else:
            return samples[0]

    def update_entities(self, pos, vel, min_speed, max_speed, jitter_chance=0.05):
        """
        高度優化的移動邏輯：完全向量化，無須 CPU 同步判斷。
        """
        num_entities = pos.shape[0]

        # 1. 向量化隨機擾動
        # 直接生成全體的擾動，再用 mask 決定誰要套用，避免 if change_mask.any()
        change_mask = torch.rand(num_entities, 1, device=DEVICE) < jitter_chance
        random_noise = (torch.rand(num_entities, 2, device=DEVICE) - 0.5) * 1.8
        vel += random_noise * change_mask # 只有被選中的會加上噪音

        # 2. 速度約束 (Speed Clamping) - 向量化實作
        speeds = torch.norm(vel, dim=1, keepdim=True)
        target_speeds = torch.clamp(speeds, min_speed, max_speed)
        vel = (vel / (speeds + 1e-6)) * target_speeds

        # 3. 更新位置
        pos += vel

        # 4. 向量化邊界處理 (Vectorized Boundary Handling)
        # 取得螢幕寬高，轉換為 Tensor 以進行快速廣播運算
        # 假設 self.screen_size 為 [W, H]
        bounds = self.screen_size - 1.0 

        # 檢查左/上邊界 ( < 0 )
        hit_min = pos < 0
        vel[hit_min] *= -1
        pos[hit_min] = 0

        # 檢查右/下邊界 ( > bounds )
        hit_max = pos > bounds
        vel[hit_max] *= -1
        pos[hit_max] = bounds[hit_max % 2].expand_as(pos)[hit_max] 
        # 註：上方 pos[hit_max] 賦值為了處理 X, Y 不同邊界，更簡單寫法如下：
        pos[:, 0] = torch.clamp(pos[:, 0], 0, bounds[0])
        pos[:, 1] = torch.clamp(pos[:, 1], 0, bounds[1])

        return pos, vel

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
        print(f"[Saved] Steps: {self.steps}, A-Loss: {self.a_loss_val:.4f}, C-Loss: {self.c_loss_val:.4f}, Rewards-Avg: {self.rewards_avg:.4f}, Hit_Pred: {self.hit_pred}, Hit_Wall: {self.hit_wall}, Starved: {self.starved}.")

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
                pygame.draw.circle(self.screen, (255, 50, 50), p.astype(int), 20, 1)
                pygame.draw.circle(self.screen, (255, 50, 50), p.astype(int), 4)

            p_np = self.pos.cpu().numpy()
            a_np = self.alive.cpu().numpy()
            e_np = self.energy.cpu().numpy()
            t_np = self.respawn_timer.cpu().numpy()
            act_np = self.last_actions.cpu().numpy()
            vel_np = self.vel.cpu().numpy()
            ang_np = self.angle.cpu().numpy()

            for i, p in enumerate(p_np):
                if not a_np[i]:
                    dead_color = (60, 60, 60) if e_np[i] <= 0 else (120, 0, 0)
                    pygame.draw.circle(self.screen, dead_color, p.astype(int), 3)
                    img = self.font.render(f"{t_np[i]:.0f}", True, dead_color)
                    text_rect = img.get_rect(center=(int(p[0]), int(p[1]) - 15))
                    self.screen.blit(img, text_rect)
                    continue

                pos_tuple = (int(p[0]), int(p[1]))
                en_ratio = e_np[i] / MAX_ENERGY
                r = int(255 * en_ratio)          # 越飽越紅
                g = int(128 * en_ratio)          # 飽的時候帶點橘色感，不飽就變暗
                b = int(255 * (1 - en_ratio))    # 越餓越藍
                color = (r, g, b)
                radius = int(4 + 4 * en_ratio)
                pygame.draw.circle(self.screen, color, pos_tuple, radius)

                act = act_np[i]
                speed = np.linalg.norm(vel_np[i])

                # 顯示能量數值
                if verbose >= 1:
                    energy_text = f"{e_np[i]:.0f}"
                    text_surface = self.font.render(energy_text, True, (255, 255, 255))
                    text_rect = text_surface.get_rect(center=(pos_tuple[0], pos_tuple[1] - radius - 8))
                    self.screen.blit(text_surface, text_rect)

                # 顯示除錯訊息
                if verbose >= 2:
                    ctrl_text = f"{act[0]:.2f} {act[1]:.2f} {speed:.2f} {self.rewards[i]:.4f}"
                    ctrl_surface = self.font.render(ctrl_text, True, (255, 255, 255))
                    ctrl_rect = ctrl_surface.get_rect(center=(pos_tuple[0], pos_tuple[1] + radius + 8))
                    self.screen.blit(ctrl_surface, ctrl_rect)

                # 畫出方向指示線
                throttle = act[1] + 1
                line_length = 5 + throttle * 10
                angle = ang_np[i]
                end_p = (int(p[0] + np.cos(angle) * line_length), int(p[1] + np.sin(angle) * line_length))
                if throttle <= 1.0:
                    line_color = (0, 255, 0)
                elif throttle <= 1.75:
                    line_color = (255, 255, 255)
                else:
                    line_color = (255, 0, 0)
                pygame.draw.line(self.screen, line_color, pos_tuple, end_p, 2)

                # 畫出視野範圍
                if draw_perception:
                    pygame.draw.circle(self.screen, color, pos_tuple, PERCEPTION_RADIUS, 1)
                # 畫出警戒範圍
                if draw_alert:
                    pygame.draw.circle(self.screen, color, pos_tuple, ALERT_RADIUS, 1)
                    pygame.draw.circle(self.screen, color, pos_tuple, TEAM_RADIUS, 1)

        ui_labels = [
            (f"FPS: {int(self.clock.get_fps())}", (0, 255, 0), True),
            (f"Steps: {self.steps:,}", (0, 255, 255), False),
            (f"A-Loss: {self.a_loss_val:.3f}", (255, 100, 100), False),
            (f"C-Loss: {self.c_loss_val:.3f}", (255, 100, 100), False),
            (f"Rewards: {self.rewards_avg:.3f}", (255, 100, 100), False),
            (f"Hit Pred: {self.hit_pred}", (255, 100, 100), False),
            (f"Hit Wall: {self.hit_wall}", (255, 100, 100), False),
            (f"Starved: {self.starved}", (255, 100, 100), False),
            (f"Alive: {int(self.alive.sum())}/{POP_SIZE}", (100, 255, 100), False)
        ]
        for i, (text, color, bold) in enumerate(ui_labels):
            surf = (self.big_font if bold else self.font).render(text, True, color)
            self.screen.blit(surf, (10, 10 + i * 25))

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
