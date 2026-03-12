import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from collections import deque
from pathlib import Path

script_name = Path(__file__).stem
CAPTION = "Vectra: Apex Protocol"
# --- 參數設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POP_SIZE = 16
SCREEN_W, SCREEN_H = 900, 700
SAVE_PATH = f"{script_name}.pt"
BRAIN_PATH = f"{script_name}_brain.pt"

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
DIST_FOOD = 5
DIST_PRED = 5

# 環境參數
FOOD_SIZE = POP_SIZE * 2
PREDATOR_SIZE = 8
MAX_ENERGY = 100.0
FOOD_ENERGY = 5.0
ENERGY_DECAY = 0.08
ALERT_RADIUS = 100 # 定義警戒半徑

# --- DDPG 網路架構 ---
# --- Actor 網路：策略決策者 ---
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()# 6 個輸入通道：
        # [0:2] 食物角度(cos, sin), [2] 食物距離
        # [3:5] 掠食者角度(cos, sin), [5] 掠食者威脅度
        self.conv = nn.Conv1d(6, 32, 1) 
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
        輸入:
        m_in: Tensor [Batch, 6, 5] -> 6個特徵，探測周圍最近的 5 個物體，這裡的 '5' 指的是 DIST_FOOD (5個) 與 DIST_PRED (5個) 對齊後的物體槽位
        s_in: Tensor [Batch, 3]    -> 自身狀態 (Speed, Constant_0, Energy)
        
        輸出:
        actions: Tensor [Batch, 2] -> 每個 Agent 的 [轉向, 加速] 連續值
        """
        # Conv1d 處理局部感應數據 -> [Batch, 32, 5]
        # Max 降維取最強特徵 -> [Batch, 32]
        x = torch.max(F.relu(self.conv(m_in)), dim=2)[0]
        
        # 結合感應特徵與自身狀態 -> [Batch, 35]
        combined = torch.cat([x, s_in], dim=1)
        
        # 輸出最終動作
        return self.fc(combined)

# --- Critic 網路：價值評估者 ---
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv = nn.Conv1d(6, 32, 1)
        self.fc = nn.Sequential(
            # 32 (感應特徵) + 3 (自身狀態) + 2 (動作) = 37
            nn.Linear(35 + 2, 64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 輸出 1: 該 狀態+動作 的預期 Q 值 (分數)
        )

    def forward(self, m_in, s_in, action):
        """
        輸入:
        m_in: Tensor [Batch, 6, 5]
        s_in: Tensor [Batch, 3]
        action: Tensor [Batch, 2] -> 來自 Actor 預測或 Memory 紀錄的動作
        
        輸出:
        q_value: Tensor [Batch, 1] -> 預測的未來總分回報
        """
        # 處理感應數據 -> [Batch, 32]
        x = torch.max(F.relu(self.conv(m_in)), dim=2)[0]
        
        # 結合特徵、自身狀態與動作數據 -> [Batch, 37]
        combined = torch.cat([x, s_in, action], dim=1)
        
        # 評估該動作在該狀態下的價值
        return self.fc(combined)
    
class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self): return len(self.buffer)

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
        self.font = pygame.font.SysFont("Consolas", 16)
        self.big_font = pygame.font.SysFont("Consolas", 20, bold=True)
        self.fps = 240
        self.update_caption()
        
        # 初始化 DDPG 網路
        self.actor = Actor().to(DEVICE)
        self.actor_target = Actor().to(DEVICE)
        self.critic = Critic().to(DEVICE)
        self.critic_target = Critic().to(DEVICE)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.ou_noise = OUNoise(2) # 2 個動作維度
        self.epsilon = EPSILON_START
        self.steps_done = 0
        self.a_loss_val = 0.0
        self.c_loss_val = 0.0
        self.gen_count = 1
        
        self.reset_env()
        self.load_state()

    def update_caption(self):
        pygame.display.set_caption(f"{CAPTION} | FPS:{self.fps}")

    def reset_env(self):
        self.gen_start_time = pygame.time.get_ticks()
        self.last_actions = torch.zeros((POP_SIZE, 2), device=DEVICE)
        self.pos = torch.rand(POP_SIZE, 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)
        self.vel = torch.zeros(POP_SIZE, 2).to(DEVICE)
        self.angle = torch.rand(POP_SIZE).to(DEVICE) * 2 * np.pi
        self.energy = torch.full((POP_SIZE,), MAX_ENERGY).to(DEVICE)
        self.alive = torch.ones(POP_SIZE, dtype=torch.bool).to(DEVICE)
        self.food_pos = torch.rand(FOOD_SIZE, 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)
        self.pred_pos = torch.rand(PREDATOR_SIZE, 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)
        self.pred_vel = (torch.rand(PREDATOR_SIZE, 2).to(DEVICE) - 0.5).to(DEVICE) * 3.5

    def get_states(self):
        dist_food = torch.cdist(self.pos, self.food_pos)
        dist_pred = torch.cdist(self.pos, self.pred_pos)
        _, f_idx = torch.topk(dist_food, DIST_FOOD, largest=False)
        _, p_idx = torch.topk(dist_pred, DIST_PRED, largest=False)

        f_diff = self.food_pos[f_idx] - self.pos.unsqueeze(1)
        f_dist = torch.norm(f_diff, dim=2)
        f_ang = torch.atan2(f_diff[:,:,1], f_diff[:,:,0]) - self.angle.unsqueeze(1)
        food_in = torch.stack([torch.cos(f_ang), torch.sin(f_ang), f_dist/1000.0], dim=1)

        p_diff = self.pred_pos[p_idx] - self.pos.unsqueeze(1)
        p_dist = torch.norm(p_diff, dim=2)
        p_ang = torch.atan2(p_diff[:,:,1], p_diff[:,:,0]) - self.angle.unsqueeze(1)
        threat_score = 100.0 / ((p_dist / 10.0)**2 + 1)
        pred_in = torch.stack([torch.cos(p_ang), torch.sin(p_ang), threat_score], dim=1)

        mixed_in = torch.cat([food_in, pred_in], dim=1)
        speed = torch.norm(self.vel, dim=1) / 10.0
        last_steer = self.last_actions[:, 0]
        self_in = torch.stack([speed, last_steer, self.energy/MAX_ENERGY], dim=1)
        
        return (mixed_in, self_in)

    def update(self):
        self.gen_ticks = pygame.time.get_ticks() - self.gen_start_time

        if not self.alive.any():
            print(f"Gen {self.gen_count} finished. Survived for {self.gen_ticks:,} ticks.")
            self.gen_count += 1
            self.reset_env()
            return

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
            
        rewards = torch.full((POP_SIZE,), 0.05, device=DEVICE) # 給予微小存活獎勵
        
        # 物理運算 (載具動力學)
        for i in range(POP_SIZE):
            if not self.alive[i]:
                continue
            
            # Action[0] 為轉向 (-1 到 1 映射至 -0.15 到 0.15 弧度)
            steer = actions[i][0] * 0.15
            # Action[1] 為油門 (-1 到 1 映射至 0 到 0.3 加速度)
            throttle = (actions[i][1] + 1.0) * 0.3
            
            self.angle[i] += steer
            dir_vec = torch.tensor([torch.cos(self.angle[i]), torch.sin(self.angle[i])], device=DEVICE)
            # 慣性與阻力：新速度 = 舊速度 * 摩擦係數 + 方向 * 油門加速度
            self.vel[i] = (self.vel[i] * 0.85) + (dir_vec * throttle)
            self.pos[i] += self.vel[i]
            
            # 鼓勵向前移動，懲罰倒退或原地不動
            speed_val = torch.norm(self.vel[i])
            if speed_val > 0.5:
                rewards[i] += 0.05

        # 邊界碰撞處理 (反彈)
        hit_w_x = (self.pos[:, 0] < 0) | (self.pos[:, 0] > SCREEN_W)
        hit_w_y = (self.pos[:, 1] < 0) | (self.pos[:, 1] > SCREEN_H)
        self.vel[hit_w_x, 0] *= -0.5
        self.vel[hit_w_y, 1] *= -0.5
        self.pos[:, 0] = torch.clamp(self.pos[:, 0], 0, SCREEN_W)
        self.pos[:, 1] = torch.clamp(self.pos[:, 1], 0, SCREEN_H)

        self.energy -= ENERGY_DECAY

        # 掠食者移動
        # 1. 隨機擾動：增加掠食者轉向的變幻莫測感
        change_mask = torch.rand(PREDATOR_SIZE, device=DEVICE) < 0.05
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
        hit_left   = self.pred_pos[:, 0] < 0
        hit_right  = self.pred_pos[:, 0] > SCREEN_W
        hit_top    = self.pred_pos[:, 1] < 0
        hit_bottom = self.pred_pos[:, 1] > SCREEN_H
        if hit_left.any():
            self.pred_vel[hit_left, 0] *= -1.0
            self.pred_pos[hit_left, 0] = 1.0
        if hit_right.any():
            self.pred_vel[hit_right, 0] *= -1.0
            self.pred_pos[hit_right, 0] = SCREEN_W - 1.0
        if hit_top.any():
            self.pred_vel[hit_top, 1] *= -1.0
            self.pred_pos[hit_top, 1] = 1.0
        if hit_bottom.any():
            self.pred_vel[hit_bottom, 1] *= -1.0
            self.pred_pos[hit_bottom, 1] = SCREEN_H - 1.0

        # 食物碰撞
        dist_f = torch.cdist(self.pos, self.food_pos)
        hits_f = (dist_f < 15.0) & self.alive.unsqueeze(1)
        if hits_f.any():
            a_idx, f_idx = torch.where(hits_f)
            rewards[a_idx] += 5.0
            self.energy[a_idx] = torch.clamp(self.energy[a_idx] + FOOD_ENERGY, max=MAX_ENERGY)
            self.food_pos[f_idx] = torch.rand(len(f_idx), 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).to(DEVICE)

        # 掠食者碰撞
        dist_p = torch.cdist(self.pos, self.pred_pos)
        min_dist_p, _ = torch.min(dist_p, dim=1)
        # 只有還活著且進入警戒區的才扣分
        danger_mask = (min_dist_p < ALERT_RADIUS) & self.alive
        if danger_mask.any():
            # 懲罰函數：距離越近扣越多
            # 當距離=22(碰撞邊緣)時，大約扣 0.4
            # 當距離=100時，扣 0.1
            danger_penalty = 0.5 * (1.0 - min_dist_p[danger_mask] / ALERT_RADIUS)
            rewards[danger_mask] -= danger_penalty

        # 區分死因的獎勵邏輯
        hits_p = (dist_p < 22.0).any(dim=1) & self.alive
        starved = (self.energy <= 0) & self.alive        
        rewards[hits_p] -= 10  # 被殺死
        rewards[starved] -= -8 # 餓死
        
        dead_mask = hits_p | starved
        if dead_mask.any():
            self.alive[dead_mask] = False
            self.vel[dead_mask] = 0.0 # 死掉後速度歸零

        next_states = self.get_states()
        
        # 把經驗推入 Replay Buffer
        for i in range(POP_SIZE):
            if not was_alive[i]: 
                continue # 忽略死屍的經驗
            s_i = (current_states[0][i:i+1], current_states[1][i:i+1])
            ns_i = (next_states[0][i:i+1], next_states[1][i:i+1])
            self.memory.push(s_i, actions[i].unsqueeze(0), rewards[i].unsqueeze(0), ns_i, dead_mask[i].unsqueeze(0))

        self.optimize_model()
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        self.steps_done += 1

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = self.memory.sample(BATCH_SIZE)
        m_b = torch.cat([x[0][0] for x in batch])
        s_b = torch.cat([x[0][1] for x in batch])
        a_b = torch.cat([x[1] for x in batch])
        r_b = torch.cat([x[2] for x in batch])
        nm_b = torch.cat([x[3][0] for x in batch])
        ns_b = torch.cat([x[3][1] for x in batch])
        d_b = torch.cat([x[4] for x in batch])

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

    def save_state(self):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'eps': self.epsilon,
            'steps': self.steps_done,
            'gen': self.gen_count,
            'gen_ticks': self.gen_ticks,
            'pos': self.pos,
            'last_actions': self.last_actions,
            'energy': self.energy,
            'alive': self.alive,
            'food_pos': self.food_pos,
            'pred_pos': self.pred_pos
        }, SAVE_PATH)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, BRAIN_PATH)
        print(f"--- [Saved] Progress to {SAVE_PATH} and {BRAIN_PATH} ---")

    def load_state(self):
        if os.path.exists(SAVE_PATH):
            try:
                state = torch.load(SAVE_PATH, map_location=DEVICE)
                self.actor.load_state_dict(state['actor'])
                self.actor_target.load_state_dict(self.actor.state_dict())
                self.critic.load_state_dict(state['critic'])
                self.critic_target.load_state_dict(self.critic.state_dict())
                self.epsilon = state['eps']
                self.steps_done = state['steps']
                self.gen_count = state['gen']
                self.gen_ticks = state['gen_ticks']
                self.pos = state['pos']
                self.last_actions = state['last_actions']
                self.energy = state['energy']
                self.alive = state['alive']
                self.food_pos = state['food_pos']
                self.pred_pos = state['pred_pos']
                print(f"--- [Loaded] Gen {self.gen_count}, Steps {self.steps_done:,} ---")
                return
            except Exception as e:
                print(f"--- [Error] Loading failed: {e} ---")

        if os.path.exists(BRAIN_PATH):
            try:
                brain_state = torch.load(BRAIN_PATH, map_location=DEVICE)
                self.actor.load_state_dict(brain_state['actor'])
                self.actor_target.load_state_dict(self.actor.state_dict())
                self.critic.load_state_dict(brain_state['critic'])
                self.critic_target.load_state_dict(self.critic.state_dict())
                print(f"--- [Loaded] Brain weights only. Experience transfered. ---")
                return True
            except Exception as e:
                print(f"--- [Error] Brain loading failed: {e} ---")


    def draw(self, draw_alert):
        self.screen.fill((20, 20, 25))
        for f in self.food_pos.cpu().numpy(): pygame.draw.circle(self.screen, (0, 255, 120), f.astype(int), 3)
        for p in self.pred_pos.cpu().numpy(): 
            pygame.draw.circle(self.screen, (255, 50, 50), p.astype(int), 20, 1)
            pygame.draw.circle(self.screen, (255, 50, 50), p.astype(int), 4)

        p_np = self.pos.cpu().numpy()
        a_np = self.alive.cpu().numpy()
        e_np = self.energy.cpu().numpy()
        
        for i, p in enumerate(p_np):
            if not a_np[i]:
                dead_color = (60, 60, 60) if e_np[i] <= 0 else (120, 0, 0)
                pygame.draw.circle(self.screen, dead_color, p.astype(int), 3)
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

            # 畫出警戒範圍
            if draw_alert:
                pygame.draw.circle(self.screen, color, pos_tuple, ALERT_RADIUS, 1)

        ui_labels = [
            (f"FPS: {int(self.clock.get_fps())}", (0, 255, 0), False),
            (f"Steps: {self.steps_done:,}", (0, 255, 255), False),
            (f"Generation: {self.gen_count} | {self.gen_ticks:,}", (200, 200, 200), False),
            (f"Actor Loss: {self.a_loss_val:.3f}", (255, 100, 100), False),
            (f"Critic Loss: {self.c_loss_val:.3f}", (255, 100, 100), False),
            (f"Alive: {int(self.alive.sum())}/{POP_SIZE}", (100, 255, 100), False)
        ]
        for i, (text, color, bold) in enumerate(ui_labels):
            surf = (self.big_font if bold else self.font).render(text, True, color)
            self.screen.blit(surf, (20, 20 + i*30))
        pygame.display.flip()
        self.clock.tick(self.fps)

    def run(self):
        running = True
        draw_ui = True
        is_paused = False
        draw_alert = False

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_t:
                        draw_ui = not draw_ui
                    elif event.key == pygame.K_a:
                        draw_alert = not draw_alert
                    elif event.key == pygame.K_SPACE:
                        is_paused = not is_paused
                    elif event.key == pygame.K_UP:
                        self.fps += 5
                        self.update_caption()
                    elif event.key == pygame.K_DOWN:
                        self.fps -= 5
                        self.update_caption()

            if not is_paused:
                self.update()
                if self.steps_done % 5000 == 0:
                    self.save_state()
                
            if draw_ui:
                self.draw(draw_alert)
            else:
                if not is_paused and self.steps_done % 1000 == 0:
                    print(f"Steps: {self.steps_done}, A-Loss: {self.a_loss_val:.4f}, C-Loss: {self.c_loss_val:.4f}")
                    
        self.save_state()
        pygame.quit()

if __name__ == "__main__":
    sim = RLSimulation()
    sim.run()