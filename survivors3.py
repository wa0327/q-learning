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
POP_SIZE = 15
FOOD_SIZE = 30
PREDATOR_SIZE = 20
MAX_ENERGY = 100.0
FOOD_ENERGY = 10.0
ENERGY_DECAY = 0.03
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
        # 預分配空間 (根據你的 Actor 輸入維度)
        # s1: mixed_in [MAX_OBJ, FEAT_DIM], s2: self_in [3]
        self.m_states = np.zeros((capacity, MAX_OBJ, FEAT_DIM), dtype=np.float32)
        self.s_states = np.zeros((capacity, 3), dtype=np.float32)
        self.next_m_states = np.zeros((capacity, MAX_OBJ, FEAT_DIM), dtype=np.float32)
        self.next_s_states = np.zeros((capacity, 3), dtype=np.float32)
        self.actions = np.zeros((capacity, 2), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.idx = 0
        self.size = 0

    def push(self, m_s, s_s, action, reward, n_m_s, n_s_s, done):
        # 存入時強制轉為 CPU 上的 NumPy 陣列，切斷梯度
        self.m_states[self.idx] = m_s.detach().cpu().numpy()
        self.s_states[self.idx] = s_s.detach().cpu().numpy()
        self.next_m_states[self.idx] = n_m_s.detach().cpu().numpy()
        self.next_s_states[self.idx] = n_s_s.detach().cpu().numpy()
        self.actions[self.idx] = action.detach().cpu().numpy()
        self.rewards[self.idx] = reward.item()
        self.dones[self.idx] = done.item()
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        # 轉回 Torch Tensor 並送往 GPU (一次性處理一個 Batch，效率最高)
        return (
            torch.from_numpy(self.m_states[indices]).to(DEVICE),
            torch.from_numpy(self.s_states[indices]).to(DEVICE),
            torch.from_numpy(self.actions[indices]).to(DEVICE),
            torch.from_numpy(self.rewards[indices]).to(DEVICE),
            torch.from_numpy(self.next_m_states[indices]).to(DEVICE),
            torch.from_numpy(self.next_s_states[indices]).to(DEVICE),
            torch.from_numpy(self.dones[indices]).to(DEVICE)
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
        self.rewards = 0.0
        self.dead = 0
        self.starved = 0
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
        self.prev_pos = self.pos.clone()
        self.food_pos = torch.rand(FOOD_SIZE, 2, device=DEVICE) * self.screen_size
        self.pred_pos = torch.rand(PREDATOR_SIZE, 2, device=DEVICE) * self.screen_size
        self.pred_vel = (torch.rand(PREDATOR_SIZE, 2, device=DEVICE) - 0.5) * 3.5

    def get_states(self):
        """
        將環境觀察值轉換為神經網路輸入張量。
        輸出:
            mixed_in: [Batch, MAX_OBJ, FEAT_DIM] -> 環境物體特徵
            self_in:  [Batch, 3]                 -> 自身狀態特徵
        """
        # --- 1. 初始化類別標籤 (One-hot Encoding) ---
        # 對應維度: [牆, 隊友, 食物, 敵人]
        tag_wall = torch.tensor([1, 0, 0, 0], device=DEVICE).view(1, 1, 4)
        tag_team = torch.tensor([0, 1, 0, 0], device=DEVICE).view(1, 1, 4)
        tag_food = torch.tensor([0, 0, 1, 0], device=DEVICE).view(1, 1, 4)
        tag_pred = torch.tensor([0, 0, 0, 1], device=DEVICE).view(1, 1, 4)

        all_detected_objs = [] # 用於暫存所有類型的物體特徵池

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
        wall_in = torch.cat([wall_phys * wall_mask.unsqueeze(-1), 
                            tag_wall.expand(POP_SIZE, 4, 4) * wall_mask.unsqueeze(-1)], dim=2)
        all_detected_objs.append(wall_in)

        # --- 3. 處理食物 (資源感應) ---
        dist_food = torch.cdist(self.pos, self.food_pos)
        food_mask = (dist_food < PERCEPTION_RADIUS).float()
        
        f_diff = self.food_pos.unsqueeze(0) - self.pos.unsqueeze(1)
        f_ang = torch.atan2(f_diff[..., 1], f_diff[..., 0]) - self.angle.unsqueeze(1)
        food_scores = torch.pow((1.0 - dist_food / PERCEPTION_RADIUS).clamp(0, 1.0), 2) * food_mask
        
        food_phys = torch.stack([torch.cos(f_ang), torch.sin(f_ang), food_scores], dim=2)
        food_in = torch.cat([food_phys * food_mask.unsqueeze(-1), 
                            tag_food.expand(POP_SIZE, food_phys.shape[1], 4) * food_mask.unsqueeze(-1)], dim=2)
        all_detected_objs.append(food_in)

        # --- 4. 處理隊友 (群體感應) ---
        if POP_SIZE > 1:
            dist_agents = torch.cdist(self.pos, self.pos)
            dist_agents.fill_diagonal_(999.0) # 排除自身
            team_mask = (dist_agents < PERCEPTION_RADIUS).float()
            
            t_diff = self.pos.unsqueeze(0) - self.pos.unsqueeze(1)
            t_ang = torch.atan2(t_diff[..., 1], t_diff[..., 0]) - self.angle.unsqueeze(1)
            team_scores = (1.0 - dist_agents / PERCEPTION_RADIUS).clamp(0, 1.0) * team_mask
            
            team_phys = torch.stack([torch.cos(t_ang), torch.sin(t_ang), team_scores], dim=2)
            team_in = torch.cat([team_phys * team_mask.unsqueeze(-1), 
                                tag_team.expand(POP_SIZE, team_phys.shape[1], 4) * team_mask.unsqueeze(-1)], dim=2)
            all_detected_objs.append(team_in)

        # --- 5. 處理敵人 (威脅感應) ---
        if self.pred_pos.shape[0] > 0:
            dist_pred = torch.cdist(self.pos, self.pred_pos)
            pred_mask = (dist_pred < PERCEPTION_RADIUS).float()
            
            p_diff = self.pred_pos.unsqueeze(0) - self.pos.unsqueeze(1)
            p_ang = torch.atan2(p_diff[..., 1], p_diff[..., 0]) - self.angle.unsqueeze(1)
            # 敵人評分權重加倍 (1.5x)，確保其在 Top-K 篩選中具備更高優先權
            pred_scores = (1.0 - dist_pred / PERCEPTION_RADIUS).clamp(0, 1.0) * pred_mask * 1.5
            
            pred_phys = torch.stack([torch.cos(p_ang), torch.sin(p_ang), pred_scores], dim=2)
            pred_in = torch.cat([pred_phys * pred_mask.unsqueeze(-1), 
                                tag_pred.expand(POP_SIZE, pred_phys.shape[1], 4) * pred_mask.unsqueeze(-1)], dim=2)
            all_detected_objs.append(pred_in)

        # --- 6. 綜合特徵池化與 Top-K 篩選 ---
        # 合併所有類型的物體特徵 [Batch, Total_Detected_N, 7]
        combined_all = torch.cat(all_detected_objs, dim=1)
        
        # 根據 score (維度索引 2) 提取前 MAX_OBJ 個最重要物體
        scores = combined_all[:, :, 2]
        _, top_indices = torch.topk(scores, MAX_OBJ, dim=1, largest=True)
        
        # 根據索引重新聚合特徵，確保輸出 Shape 固定為 [Batch, 20, 7]
        idx_expanded = top_indices.unsqueeze(-1).expand(-1, -1, 7)
        mixed_in = torch.gather(combined_all, 1, idx_expanded)

        # --- 7. 處理自身狀態 (Self State) ---
        # 包含：標準化後的速率、最後一次轉向動作、剩餘能量比例
        speed = torch.norm(self.vel, dim=1) / 10.0
        last_steer = self.last_actions[:, 0]
        energy_ratio = self.energy / MAX_ENERGY
        self_in = torch.stack([speed, last_steer, energy_ratio], dim=1)

        return (mixed_in, self_in)

    def update(self, move_predator):
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
        # # 取得食物距離矩陣 (用於後續獎勵計算)
        # dist_f_all = torch.cdist(self.pos, self.food_pos)
        # min_dist_f, _ = torch.min(dist_f_all, dim=1)

        # 物理運算 (載具動力學)
        for i in range(POP_SIZE):
            if not self.alive[i]:
                continue
            
            # 計算當前速度
            speed_val = torch.norm(self.vel[i])

            # actions[0] 為轉向 (-1 到 1 映射至 -0.15 到 0.15 弧度)
            # 舵效：速度越快轉向越明顯，但極速時轉向半徑應變大
            steer_power = torch.clamp(speed_val / 2.0, 0, 1.0) 
            steer = actions[i][0] * 0.15 * steer_power
            self.angle[i] += steer

            # Action[1] 為油門 (-1 到 1 映射至 0 到 0.3 加速度)
            throttle = (actions[i][1] + 1.0) * 0.3
            # 向量動力學：計算前進方向推力
            dir_vec = torch.tensor([torch.cos(self.angle[i]), torch.sin(self.angle[i])], device=DEVICE)
            self.vel[i] = self.vel[i] * 0.85 + dir_vec * throttle
            
            # 更新位置
            self.pos[i] += self.vel[i]
            
            # # --- 精準進場獎勵 (解決繞圈問題) ---
            # if min_dist_f[i] < 50.0:
            #     # 獎勵低速接近：鼓勵精準對準食物
            #     if speed_val < 1.2:
            #         rewards[i] += 0.15
            #     # 懲罰高速衝過頭：防止因轉彎半徑太大繞圈
            #     elif speed_val > 2.2:
            #         rewards[i] -= 0.1
            # else:
            #     # 遠處時，鼓勵直線加速前進
            #     forward_bonus = (speed_val / 10.0) * (1.0 - torch.abs(actions[i][0]))
            #     rewards[i] += forward_bonus * 0.1  # 這會鼓勵直線衝刺

        # 邊界碰撞處理 (反彈)
        hit_w_x = (self.pos[:, 0] <= 0) | (self.pos[:, 0] >= SCREEN_W - 1)
        hit_w_y = (self.pos[:, 1] <= 0) | (self.pos[:, 1] >= SCREEN_H - 1)
        rewards[hit_w_x | hit_w_y] -= 0.5
        self.vel[hit_w_x, 0] *= -0.5
        self.vel[hit_w_y, 1] *= -0.5
        self.pos[:, 0] = torch.clamp(self.pos[:, 0], 0, SCREEN_W - 1)
        self.pos[:, 1] = torch.clamp(self.pos[:, 1], 0, SCREEN_H - 1)

        # 移動距離獎懲
        displacement = torch.norm(self.pos - self.prev_pos, dim=1)
        self.prev_pos = self.pos.clone()
        for i in range(POP_SIZE):
            if not self.alive[i]:
                continue
            
            self.energy[i] -= displacement[i] * 0.05

            # 只有當位移大於一定程度（代表真的在移動，不是微小震動或繞圈）才給分
            if displacement[i] > 2.0:
                steer_val = torch.abs(actions[i][0])
                move_reward = 0.2 * (1.0 - steer_val * 1.5) # 轉向超過 0.66 就會變負分
                rewards[i] += move_reward
            else:
                rewards[i] -= 0.2

            # # 額外懲罰：如果視覺完全沒東西（空曠區）卻還在轉彎
            # vision_sum = torch.sum(torch.abs(current_states[0][i]))
            # if vision_sum < 0.01 and torch.abs(actions[i][0]) > 0.3:
            #     # 空曠地區轉向懲罰
            #     rewards[i] -= torch.abs(actions[i][0]) * 0.1

        # self.energy[self.alive] -= ENERGY_DECAY
        self.respawn_timer[~self.alive] -= 1

        # 掠食者移動
        if PREDATOR_SIZE > 0 and move_predator:
            # 1. 隨機擾動：增加掠食者轉向的變幻莫測感
            num_preds = self.pred_pos.shape[0]
            change_mask = torch.rand(num_preds, device=DEVICE) < 0.05
            if change_mask.any():
                # 產生隨機加速度擾動
                random_noise = (torch.rand(int(change_mask.sum()), 2, device=DEVICE) - 0.5) * 1.8
                self.pred_vel[change_mask] += random_noise
                
                # 重新標準化速度，確保掠食者不會無限加速或停下
                speeds = torch.norm(self.pred_vel, dim=1, keepdim=True)
                target_speeds = torch.clamp(speeds, 1.5, 3.5)
                self.pred_vel = (self.pred_vel / (speeds + 1e-6)) * target_speeds

            # 2. 根據速度更新位置
            self.pred_pos += self.pred_vel
            # 3. 強力邊界反彈與座標修正 (防止掠食者卡在牆外)
            hit_left   = self.pred_pos[:, 0] <= 0
            hit_right  = self.pred_pos[:, 0] >= SCREEN_W - 1
            hit_top    = self.pred_pos[:, 1] <= 0
            hit_bottom = self.pred_pos[:, 1] >= SCREEN_H - 1
            if hit_left.any():
                self.pred_vel[hit_left, 0] *= -1
                self.pred_pos[hit_left, 0] = 0
            if hit_right.any():
                self.pred_vel[hit_right, 0] *= -1
                self.pred_pos[hit_right, 0] = SCREEN_W - 1
            if hit_top.any():
                self.pred_vel[hit_top, 1] *= -1
                self.pred_pos[hit_top, 1] = 0
            if hit_bottom.any():
                self.pred_vel[hit_bottom, 1] *= -1
                self.pred_pos[hit_bottom, 1] = SCREEN_H - 1

        # 隊友排斥，排除自己與自己的距離 (對角線設為大值)
        if POP_SIZE > 1:
            dist_agents = torch.cdist(self.pos, self.pos).fill_diagonal_(999.0)
            min_dist_to_friend, _ = torch.min(dist_agents, dim=1)
            # 如果靠太近，給予懲罰
            team_mask = (min_dist_to_friend < TEAM_RADIUS) & self.alive
            rewards[team_mask] -= 0.1

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
                danger_penalty = 1.0 * (1.0 - min_dist_p[danger_mask] / ALERT_RADIUS)
                rewards[danger_mask] -= danger_penalty

            hits_p = (dist_p < 22.0).any(dim=1) & self.alive
            rewards[hits_p] -= 10  # 被殺死
        else:
            hits_p = torch.zeros(POP_SIZE, dtype=torch.bool, device=DEVICE)

        starved = (self.energy <= 0) & self.alive
        rewards[starved] -= -8 # 餓死

        dead_mask = hits_p | starved
        if dead_mask.any():
            self.alive[dead_mask] = False
            self.vel[dead_mask] = 0.0 # 死掉後速度歸零
            self.respawn_timer[dead_mask] = torch.randint(60, 360, (dead_mask.sum(),), device=DEVICE)
        
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

        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        self.steps += 1
        self.rewards = self.rewards * 0.99 + (rewards.sum().item() / POP_SIZE) * 0.01
        self.dead += hits_p.sum().item()
        self.starved += starved.sum().item()

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

    def save_state(self):
        torch.save({
            'pos': self.pos,
            'last_actions': self.last_actions,
            'energy': self.energy,
            'alive': self.alive,
            'respawn_timer': self.respawn_timer,
            'food_pos': self.food_pos,
            'pred_pos': self.pred_pos,
            'rewards': self.rewards,
            'dead': self.dead,
            'starved': self.starved
        }, SAVE_PATH)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'steps': self.steps,
            'eps': self.epsilon
        }, self.brain_path)
        print(f"[Saved] Steps: {self.steps}, A-Loss: {self.a_loss_val:.4f}, C-Loss: {self.c_loss_val:.4f}, Rewards: {self.rewards:.4f}, Dead: {self.dead}, Starved: {self.starved}.")

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
                self.rewards = state['rewards']
                self.dead = state['dead']
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


    def draw(self, label_only, draw_perception, draw_alert):
        self.screen.fill((20, 20, 25))

        if not label_only:
            for f in self.food_pos.cpu().numpy(): pygame.draw.circle(self.screen, (0, 255, 120), f.astype(int), 3)
            for p in self.pred_pos.cpu().numpy(): 
                pygame.draw.circle(self.screen, (255, 50, 50), p.astype(int), 20, 1)
                pygame.draw.circle(self.screen, (255, 50, 50), p.astype(int), 4)

            p_np = self.pos.cpu().numpy()
            a_np = self.alive.cpu().numpy()
            e_np = self.energy.cpu().numpy()
            t_np = self.respawn_timer.cpu().numpy()

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

                # --- 顯示能量數值 ---
                energy_text = f"{e_np[i]:.0f}" 
                text_surface = self.font.render(energy_text, True, (255, 255, 255)) # 白色文字
                text_rect = text_surface.get_rect(center=(pos_tuple[0], pos_tuple[1] - radius - 8))
                self.screen.blit(text_surface, text_rect)

                # 畫出方向指示線
                angle = self.angle[i].cpu().item()
                end_p = (int(p[0] + np.cos(angle)*12), int(p[1] + np.sin(angle)*12))
                pygame.draw.line(self.screen, (255, 255, 255), pos_tuple, end_p, 2)

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
            (f"Rewards: {self.rewards:.3f}", (255, 100, 100), False),
            (f"Dead: {self.dead}", (255, 100, 100), False),
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
        move_predator = True

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
                        self.rewards = 0.0
                        self.dead = 0
                        self.starved = 0
                    elif event.key == pygame.K_SPACE:
                        is_paused = not is_paused
                    elif event.key == pygame.K_h:
                        label_only = not label_only
                    elif event.key == pygame.K_a:
                        draw_alert = not draw_alert
                    elif event.key == pygame.K_p:
                        draw_perception = not draw_perception
                    elif event.key == pygame.K_m:
                        move_predator = not move_predator

            if not is_paused:
                self.update(move_predator)
                if self.steps % 5000 == 0:
                    self.save_state()

            self.draw(label_only, draw_perception, draw_alert)

        self.save_state()
        pygame.quit()

if __name__ == "__main__":
    sim = RLSimulation()
    sim.run()
