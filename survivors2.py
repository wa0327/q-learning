import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random

# --- 參數設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POP_SIZE = 50
FOOD_SIZE = 100
PREDATOR_SIZE = 5
SCREEN_W, SCREEN_H = 1024, 768
FPS = 240
EVOLUTION_TICKS = 10000 
SAVE_PATH = "survivors2.pt"
PRETRAIN_PATH = "pretrain2.pt"
MAX_ENERGY = 100.0
ENERGY_DECAY = 0.12

# 感知與網路參數
NEARBY_FOOD_COUNT = 5
NEARBY_PRED_COUNT = 3
OBJ_FEATURES = 3   
SELF_FEATURES = 3  

# --- 腦架構 (CNN + FCN) ---
class HybridBrain(nn.Module):
    def __init__(self):
        super(HybridBrain, self).__init__()
        # 處理多目標的卷積層
        self.food_conv = nn.Conv1d(OBJ_FEATURES, 16, kernel_size=1)
        self.pred_conv = nn.Conv1d(OBJ_FEATURES, 16, kernel_size=1)
        
        # 融合層：16(food) + 16(pred) + 3(self) = 35
        self.fc = nn.Sequential(
            nn.Linear(35, 24),
            nn.ReLU(),
            nn.Linear(24, 2),
            nn.Tanh()
        )

    def forward(self, food_list, pred_list, self_state):
        # food_list shape: [B, 3, 5]
        f = F.relu(self.food_conv(food_list))
        f = torch.max(f, dim=2)[0] # 取出最強特徵
        
        # pred_list shape: [B, 3, 3]
        p = F.relu(self.pred_conv(pred_list))
        p = torch.max(p, dim=2)[0]
        
        combined = torch.cat([f, p, self_state], dim=1)
        return self.fc(combined)

class EvolutionSim:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption(f"CNN+FCN Evolution - {DEVICE.type.upper()}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.big_font = pygame.font.SysFont("Arial", 28, bold=True)
        
        self.generation_count = 1
        self.evolution_ticks = EVOLUTION_TICKS
        self.brains = [HybridBrain().to(DEVICE) for _ in range(POP_SIZE)]

        if not self.load_state():
            self.inject_pretrain_weights()
        
        self.init_environment()

    def inject_pretrain_weights(self):
        if os.path.exists(PRETRAIN_PATH):
            print(f"[System] No save. Injecting Expert Genes from {PRETRAIN_PATH}...")
            expert_sd = torch.load(PRETRAIN_PATH, map_location=DEVICE)
            pure_count = int(POP_SIZE * 0.1)
            for i in range(POP_SIZE):
                self.brains[i].load_state_dict(expert_sd)
                if i >= pure_count:
                    with torch.no_grad():
                        for p in self.brains[i].parameters():
                            p.add_(torch.randn(p.size()).to(DEVICE) * 0.15)
        else:
            print("[Warning] No pretrain2.pt found.")

    def init_environment(self):
        self.pos = torch.rand(POP_SIZE, 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)
        self.vel = torch.zeros(POP_SIZE, 2).to(DEVICE)
        self.angle = torch.rand(POP_SIZE).to(DEVICE) * 2 * np.pi
        self.fitness = torch.zeros(POP_SIZE).to(DEVICE)
        self.energy = torch.full((POP_SIZE,), MAX_ENERGY).to(DEVICE)
        self.alive = torch.ones(POP_SIZE, dtype=torch.bool).to(DEVICE)
        self.survival_age = torch.zeros(POP_SIZE, dtype=torch.int).to(DEVICE)
        
        self.food_pos = torch.rand(FOOD_SIZE, 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)
        self.pred_pos = torch.rand(PREDATOR_SIZE, 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)
        self.pred_vel = (torch.rand(PREDATOR_SIZE, 2).to(DEVICE) - 0.5) * 4.0

    def get_batch_sensors(self):
        dist_food = torch.cdist(self.pos, self.food_pos)
        dist_pred = torch.cdist(self.pos, self.pred_pos)
        _, f_idx = torch.topk(dist_food, NEARBY_FOOD_COUNT, largest=False)
        _, p_idx = torch.topk(dist_pred, NEARBY_PRED_COUNT, largest=False)

        f_diff = self.food_pos[f_idx] - self.pos.unsqueeze(1)
        f_dist = torch.norm(f_diff, dim=2)
        f_ang = torch.atan2(f_diff[:,:,1], f_diff[:,:,0]) - self.angle.unsqueeze(1)
        food_in = torch.stack([torch.cos(f_ang), torch.sin(f_ang), f_dist/1000.0], dim=1)

        p_diff = self.pred_pos[p_idx] - self.pos.unsqueeze(1)
        p_dist = torch.norm(p_diff, dim=2)
        p_ang = torch.atan2(p_diff[:,:,1], p_diff[:,:,0]) - self.angle.unsqueeze(1)
        pred_in = torch.stack([torch.cos(p_ang), torch.sin(p_ang), torch.clamp(200.0/(p_dist+1),0,2)], dim=1)

        self_in = torch.stack([self.vel[:,0]*0.1, self.vel[:,1]*0.1, self.energy/MAX_ENERGY], dim=1)
        return food_in, pred_in, self_in

    def update_physics(self):
        alive_mask = self.alive == True
        if not alive_mask.any(): return

        food_in, pred_in, self_in = self.get_batch_sensors()
        with torch.no_grad():
            for i in range(POP_SIZE):
                if not self.alive[i]: continue
                out = self.brains[i](food_in[i:i+1], pred_in[i:i+1], self_in[i:i+1]).squeeze()
                self.angle[i] += out[1] * 0.15
                speed = (out[0] + 1.0) * 0.15
                move_dir = torch.stack([torch.cos(self.angle[i]), torch.sin(self.angle[i])])
                self.vel[i] = self.vel[i] * 0.9 + move_dir * speed
                self.pos[i] += self.vel[i]

        self.pos[:, 0] = torch.clamp(self.pos[:, 0], 0, SCREEN_W)
        self.pos[:, 1] = torch.clamp(self.pos[:, 1], 0, SCREEN_H)
        
        self.energy[alive_mask] -= ENERGY_DECAY
        self.fitness[alive_mask] += 0.01 
        
        starved = (self.energy <= 0) & alive_mask
        if starved.any():
            self.alive[starved] = False
            self.fitness[starved] -= 100

        dist_f = torch.cdist(self.pos, self.food_pos)
        hits = (dist_f < 12.0) & self.alive.unsqueeze(1)
        if hits.any():
            a_idx, f_idx = torch.where(hits)
            self.fitness[a_idx] += 20
            self.energy[a_idx] = torch.clamp(self.energy[a_idx] + 35, max=MAX_ENERGY)
            self.food_pos[f_idx] = torch.rand(len(f_idx), 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)

        dist_p = torch.cdist(self.pos, self.pred_pos)
        killed = (dist_p < 22.0).any(dim=1) & alive_mask
        if killed.any():
            self.alive[killed] = False
            self.fitness[killed] -= 50

        self.pred_pos += self.pred_vel
        self.pred_vel[(self.pred_pos[:,0]<0)|(self.pred_pos[:,0]>SCREEN_W), 0] *= -1
        self.pred_vel[(self.pred_pos[:,1]<0)|(self.pred_pos[:,1]>SCREEN_H), 1] *= -1

    def evolve(self):
        sorted_idx = torch.argsort(self.fitness, descending=True)
        elite_idx = sorted_idx[0]
        
        self.survival_age[self.alive] += 1
        best_sd = self.brains[elite_idx].state_dict()
        print(f"[System] GEN {self.generation_count} Evolving. Best Fitness: {self.fitness[elite_idx]:.1f}")

        for i in range(POP_SIZE):
            if i == elite_idx: continue
            self.brains[i].load_state_dict(best_sd)
            with torch.no_grad():
                for p in self.brains[i].parameters():
                    p.add_(torch.randn(p.size()).to(DEVICE) * 0.12)
        
        self.save_state()
        self.generation_count += 1
        self.init_environment()
        self.evolution_ticks = EVOLUTION_TICKS

    def save_state(self):
        state = {
            'gen': self.generation_count, 
            'brains': [b.state_dict() for b in self.brains],
            'survival_age': self.survival_age
        }
        torch.save(state, SAVE_PATH)

    def load_state(self):
        if os.path.exists(SAVE_PATH):
            try:
                state = torch.load(SAVE_PATH, map_location=DEVICE)
                self.generation_count = state['gen']
                self.survival_age = state.get('survival_age', torch.zeros(POP_SIZE, dtype=torch.int).to(DEVICE))
                for i in range(POP_SIZE): self.brains[i].load_state_dict(state['brains'][i])
                print(f"[System] Successfully loaded save: GEN {self.generation_count}")
                return True
            except:
                return False
        return False

    def run(self):
        LEVEL_COLORS = [(255,255,255), (100,149,237), (200,100,255), (255,165,0), (255,215,0)]
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False

            self.update_physics()
            self.evolution_ticks -= 1
            if self.evolution_ticks <= 0 or not self.alive.any():
                self.evolve()

            self.screen.fill((20, 20, 25))
            
            # 畫場景
            for f in self.food_pos.cpu().numpy(): pygame.draw.circle(self.screen, (0, 255, 100), f.astype(int), 3)
            for p in self.pred_pos.cpu().numpy():
                pygame.draw.circle(self.screen, (255, 255, 255), p.astype(int), 15, 1)
                pygame.draw.circle(self.screen, (255, 50, 50), p.astype(int), 5)

            # 畫個體
            p_np = self.pos.cpu().numpy()
            a_np = self.alive.cpu().numpy()
            age_np = self.survival_age.cpu().numpy()
            e_np = self.energy.cpu().numpy()

            for i, p in enumerate(p_np):
                if not a_np[i]:
                    dead_color = (60, 60, 60) if e_np[i] <= 0 else (120, 0, 0)
                    pygame.draw.circle(self.screen, dead_color, p.astype(int), 3)
                    continue
                
                age = age_np[i]
                color = LEVEL_COLORS[min(age, len(LEVEL_COLORS)-1)]
                radius = 4 if age < 1 else (6 if age < 3 else 8)
                pygame.draw.circle(self.screen, color, p.astype(int), radius)
                if age >= 3:
                    pygame.draw.circle(self.screen, color, p.astype(int), radius + 5, 1)

            # UI 顯示 (包含 FPS)
            self.screen.blit(self.big_font.render(f"GEN: {self.generation_count}", True, (255,255,255)), (20, 20))
            self.screen.blit(self.font.render(f"FPS: {int(self.clock.get_fps())}", True, (0, 255, 0)), (20, 60))
            self.screen.blit(self.font.render(f"Alive: {int(self.alive.sum())}/{POP_SIZE}", True, (150, 150, 255)), (20, 90))
            self.screen.blit(self.font.render(f"Next: {self.evolution_ticks}", True, (200, 200, 200)), (20, 115))
            
            # 顯示最佳個體資訊
            best_idx = torch.argmax(self.fitness)
            champ_text = f"Best Agent: #{best_idx} | Level: {age_np[best_idx]} | Fit: {self.fitness[best_idx]:.0f}"
            self.screen.blit(self.font.render(champ_text, True, (0, 255, 150)), (400, 20))

            pygame.display.flip()
            self.clock.tick(FPS)
        
        self.save_state()
        pygame.quit()

if __name__ == "__main__":
    sim = EvolutionSim()
    sim.run()