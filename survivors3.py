import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
from collections import deque

# --- 參數設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POP_SIZE = 1
SCREEN_W, SCREEN_H = 1024, 768
SAVE_PATH = "survivors3.pt"  # 統一存檔名

# RL 核心參數
GAMMA = 0.98
LR = 0.0005
MEMORY_SIZE = 50000
BATCH_SIZE = 128
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 0.9998

# 環境參數
FOOD_SIZE = 80
PREDATOR_SIZE = 5
MAX_ENERGY = 100.0
ENERGY_DECAY = 0.12

# --- DQN 腦架構 ---
class DQNBrain(nn.Module):
    def __init__(self):
        super(DQNBrain, self).__init__()
        self.food_conv = nn.Conv1d(3, 16, 1)
        self.pred_conv = nn.Conv1d(3, 16, 1)
        self.fc = nn.Sequential(
            nn.Linear(35, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3) 
        )

    def forward(self, f_in, p_in, s_in):
        f = torch.max(F.relu(self.food_conv(f_in)), dim=2)[0]
        p = torch.max(F.relu(self.pred_conv(p_in)), dim=2)[0]
        combined = torch.cat([f, p, s_in], dim=1)
        return self.fc(combined)

class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self): return len(self.buffer)

class RLSimulation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 16)
        self.big_font = pygame.font.SysFont("Consolas", 20, bold=True)
        
        # 1. 初始化模型
        self.policy_net = DQNBrain().to(DEVICE)
        self.target_net = DQNBrain().to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        self.epsilon = EPSILON_START
        self.steps_done = 0
        self.loss_val = 0.0
        self.gen_count = 1
        
        # 2. 先進行環境初始化 (確保 alive 等屬性一定存在)
        self.reset_env()
        
        # 3. 嘗試讀取存檔覆蓋狀態
        self.load_state()
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset_env(self):
        self.pos = torch.rand(POP_SIZE, 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)
        self.angle = torch.rand(POP_SIZE).to(DEVICE) * 2 * np.pi
        self.energy = torch.full((POP_SIZE,), MAX_ENERGY).to(DEVICE)
        self.alive = torch.ones(POP_SIZE, dtype=torch.bool).to(DEVICE)
        self.food_pos = torch.rand(FOOD_SIZE, 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)
        self.pred_pos = torch.rand(PREDATOR_SIZE, 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)
        self.pred_vel = (torch.rand(PREDATOR_SIZE, 2).to(DEVICE) - 0.5).to(DEVICE) * 3.5

    def get_states(self):
        dist_food = torch.cdist(self.pos, self.food_pos)
        dist_pred = torch.cdist(self.pos, self.pred_pos)
        _, f_idx = torch.topk(dist_food, 5, largest=False)
        _, p_idx = torch.topk(dist_pred, 3, largest=False)

        f_diff = self.food_pos[f_idx] - self.pos.unsqueeze(1)
        f_dist = torch.norm(f_diff, dim=2)
        f_ang = torch.atan2(f_diff[:,:,1], f_diff[:,:,0]) - self.angle.unsqueeze(1)
        food_in = torch.stack([torch.cos(f_ang), torch.sin(f_ang), f_dist/1000.0], dim=1)

        p_diff = self.pred_pos[p_idx] - self.pos.unsqueeze(1)
        p_dist = torch.norm(p_diff, dim=2)
        p_ang = torch.atan2(p_diff[:,:,1], p_diff[:,:,0]) - self.angle.unsqueeze(1)
        pred_in = torch.stack([torch.cos(p_ang), torch.sin(p_ang), torch.clamp(200.0/(p_dist+1),0,2)], dim=1)

        self_in = torch.stack([torch.zeros(POP_SIZE).to(DEVICE), torch.zeros(POP_SIZE).to(DEVICE), self.energy/MAX_ENERGY], dim=1)
        return (food_in, pred_in, self_in)

    def select_action(self, states):
        f, p, s = states
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(f, p, s).max(1)[1]
        else:
            return torch.tensor([random.randrange(3) for _ in range(POP_SIZE)], device=DEVICE)

    def update(self):
        if not self.alive.any():
            self.gen_count += 1
            if self.gen_count % 5 == 0: self.save_state()
            self.reset_env()
            return

        current_states = self.get_states()
        actions = self.select_action(current_states)
        
        rewards = torch.full((POP_SIZE,), 0.05, device=DEVICE)
        
        for i in range(POP_SIZE):
            if not self.alive[i]: continue
            if actions[i] == 0: self.angle[i] -= 0.12
            elif actions[i] == 1: self.angle[i] += 0.12
            
            speed = 4.0 if actions[i] == 2 else 2.5
            move_dir = torch.tensor([torch.cos(self.angle[i]), torch.sin(self.angle[i])], device=DEVICE)
            self.pos[i] += move_dir * speed

        self.pos[:, 0] = torch.clamp(self.pos[:, 0], 0, SCREEN_W)
        self.pos[:, 1] = torch.clamp(self.pos[:, 1], 0, SCREEN_H)
        self.energy -= ENERGY_DECAY

        self.pred_pos += self.pred_vel
        hit_wall = (self.pred_pos < 0) | (self.pred_pos > torch.tensor([SCREEN_W, SCREEN_H], device=DEVICE))
        if hit_wall.any():
            self.pred_vel[hit_wall.any(dim=1)] *= -1

        dist_f = torch.cdist(self.pos, self.food_pos)
        hits_f = (dist_f < 15.0) & self.alive.unsqueeze(1)
        if hits_f.any():
            a_idx, f_idx = torch.where(hits_f)
            rewards[a_idx] += 12.0
            self.energy[a_idx] = torch.clamp(self.energy[a_idx] + 35, max=MAX_ENERGY)
            self.food_pos[f_idx] = torch.rand(len(f_idx), 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).to(DEVICE)

        dist_p = torch.cdist(self.pos, self.pred_pos)
        hits_p = (dist_p < 22.0).any(dim=1) & self.alive
        starved = (self.energy <= 0) & self.alive
        
        dead_mask = hits_p | starved
        if dead_mask.any():
            rewards[dead_mask] -= 25.0
            self.alive[dead_mask] = False

        next_states = self.get_states()
        for i in range(POP_SIZE):
            s_i = (current_states[0][i:i+1], current_states[1][i:i+1], current_states[2][i:i+1])
            ns_i = (next_states[0][i:i+1], next_states[1][i:i+1], next_states[2][i:i+1])
            self.memory.push(s_i, actions[i].unsqueeze(0), rewards[i].unsqueeze(0), ns_i, dead_mask[i].unsqueeze(0))

        self.optimize_model()
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        if self.steps_done % 1000 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.steps_done += 1

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = self.memory.sample(BATCH_SIZE)
        
        f_b = torch.cat([x[0][0] for x in batch])
        p_b = torch.cat([x[0][1] for x in batch])
        s_b = torch.cat([x[0][2] for x in batch])
        a_b = torch.cat([x[1] for x in batch])
        r_b = torch.cat([x[2] for x in batch])
        nf_b = torch.cat([x[3][0] for x in batch])
        np_b = torch.cat([x[3][1] for x in batch])
        ns_b = torch.cat([x[3][2] for x in batch])
        d_b = torch.cat([x[4] for x in batch])

        q_values = self.policy_net(f_b, p_b, s_b).gather(1, a_b.unsqueeze(1))
        with torch.no_grad():
            max_next_q = self.target_net(nf_b, np_b, ns_b).max(1)[0]
            target_q = r_b + (GAMMA * max_next_q * (1 - d_b.float()))
        
        loss = F.smooth_l1_loss(q_values, target_q.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.loss_val = loss.item()

    def save_state(self):
        state = {
            'net': self.policy_net.state_dict(),
            'opt': self.optimizer.state_dict(),
            'eps': self.epsilon,
            'steps': self.steps_done,
            'gen': self.gen_count,
            'pos': self.pos,
            'energy': self.energy,
            'alive': self.alive,
            'food_pos': self.food_pos,
            'pred_pos': self.pred_pos
        }
        torch.save(state, SAVE_PATH)
        print(f"--- [Saved] Progress to {SAVE_PATH} ---")

    def load_state(self):
        if os.path.exists(SAVE_PATH):
            try:
                state = torch.load(SAVE_PATH, map_location=DEVICE)
                self.policy_net.load_state_dict(state['net'])
                self.optimizer.load_state_dict(state['opt'])
                self.epsilon = state['eps']
                self.steps_done = state['steps']
                self.gen_count = state.get('gen', 1)
                self.pos = state['pos']
                self.energy = state['energy']
                self.alive = state['alive']
                self.food_pos = state['food_pos']
                self.pred_pos = state['pred_pos']
                print(f"--- [Loaded] Gen {self.gen_count}, Steps {self.steps_done} ---")
                return True
            except Exception as e:
                print(f"--- [Error] Loading failed: {e} ---")
                return False
        return False

    def draw(self):
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
                continue
            pos_tuple = (int(p[0]), int(p[1]))
            en_ratio = e_np[i]/MAX_ENERGY
            color = (int(255*(1-en_ratio)), int(255*en_ratio), 255)
            pygame.draw.circle(self.screen, color, pos_tuple, 8)
            angle = self.angle[i].cpu().item()
            end_p = (int(p[0] + np.cos(angle)*12), int(p[1] + np.sin(angle)*12))
            pygame.draw.line(self.screen, (255, 255, 255), pos_tuple, end_p, 2)

        current_fps = int(self.clock.get_fps())
        ui_labels = [
            (f"DQN RL - survivors3.py", (255, 255, 255), True),
            (f"FPS: {current_fps}", (0, 255, 0), False),
            (f"Generation: {self.gen_count}", (200, 200, 200), False),
            (f"Exploration: {self.epsilon:.4f}", (0, 255, 255), False),
            (f"Loss: {self.loss_val:.6f}", (255, 100, 100), False),
            (f"Alive: {int(self.alive.sum())}/{POP_SIZE}", (100, 255, 100), False)
        ]
        for i, (text, color, bold) in enumerate(ui_labels):
            surf = (self.big_font if bold else self.font).render(text, True, color)
            self.screen.blit(surf, (20, 20 + i*30))
        pygame.display.flip()
        self.clock.tick(0)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
            self.update()
            self.draw()
        self.save_state()
        pygame.quit()

if __name__ == "__main__":
    sim = RLSimulation()
    sim.run()