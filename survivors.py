import pygame
import torch
import numpy as np
import random
import time

# --- 參數設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POP_SIZE = 100          # 生物數量
FOOD_SIZE = 100         # 食物數量
SCREEN_W, SCREEN_H = 1600, 1200
FPS = 60
EVOLUTION_INTERVAL = 60  # 固定每 60 秒進化一次

# 神經網路結構：輸入(5) -> 隱藏(8) -> 輸出(2)
# 輸入：[前方距離, 左方距離, 右方距離, 食物相對角度, 剩餘能量]
INPUT_SIZE = 5
HIDDEN_SIZE = 10
OUTPUT_SIZE = 2

class EvolutionSim:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption(f"NN Evolution - CUDA: {DEVICE.type.upper()}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        
        # 初始化生物
        self.pos = torch.rand(POP_SIZE, 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)
        self.vel = torch.zeros(POP_SIZE, 2).to(DEVICE)
        self.angle = torch.rand(POP_SIZE).to(DEVICE) * 2 * np.pi
        self.fitness = torch.zeros(POP_SIZE).to(DEVICE)
        
        # 追蹤誰是精英 (用於渲染)
        self.elite_indices = []
        
        # 初始化食物
        self.food_pos = torch.rand(FOOD_SIZE, 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)
        
        # 初始化神經網路權重 (CUDA)
        self.w1 = torch.randn(POP_SIZE, INPUT_SIZE, HIDDEN_SIZE).to(DEVICE)
        self.w2 = torch.randn(POP_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)
        
        # 進化計時器
        self.last_evolution_time = time.time()

    def get_sensor_data(self):
        dist_to_food = torch.cdist(self.pos, self.food_pos)
        min_dist, min_idx = torch.min(dist_to_food, dim=1)
        target_food = self.food_pos[min_idx]
        
        dir_to_food = target_food - self.pos
        angle_to_food = torch.atan2(dir_to_food[:, 1], dir_to_food[:, 0]) - self.angle
        
        sensors = torch.stack([
            torch.cos(angle_to_food), 
            torch.sin(angle_to_food),
            torch.clamp(min_dist / 500.0, 0, 1),
            (self.pos[:, 0] / SCREEN_W) * 2 - 1,
            (self.pos[:, 1] / SCREEN_H) * 2 - 1
        ], dim=1)
        return sensors

    def update_physics(self):
        sensors = self.get_sensor_data()
        
        # 神經網路前向傳播 (批次運算)
        # x = tanh(Input * W1) -> tanh(x * W2)
        h = torch.tanh(torch.bmm(sensors.unsqueeze(1), self.w1))
        out = torch.tanh(torch.bmm(h, self.w2)).squeeze(1)
        
        # 輸出控制：[前進力量, 轉向速度]
        speed = (out[:, 0] + 1) * 2.0 # 確保主要向前運動
        self.angle += out[:, 1] * 0.15
        
        new_vel = torch.stack([torch.cos(self.angle), torch.sin(self.angle)], dim=1) * speed.unsqueeze(1)
        self.vel = self.vel * 0.95 + new_vel # 慣性與阻尼
        self.pos += self.vel
        
        # 邊界反彈 (簡單物理)
        self.pos[:, 0] = torch.clamp(self.pos[:, 0], 0, SCREEN_W)
        self.pos[:, 1] = torch.clamp(self.pos[:, 1], 0, SCREEN_H)

        # 吃到食物檢測
        dist_to_food = torch.cdist(self.pos, self.food_pos)
        eaten = dist_to_food < 12.0
        for i in range(POP_SIZE):
            food_hit = torch.where(eaten[i])[0]
            if len(food_hit) > 0:
                self.fitness[i] += len(food_hit) * 10
                self.food_pos[food_hit] = torch.rand(len(food_hit), 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)

    def evolve(self):
        # 找出前 20% 精英
        num_elites = int(POP_SIZE * 0.2)
        sorted_indices = torch.argsort(self.fitness, descending=True)
        self.elite_indices = sorted_indices[:num_elites].cpu().numpy().tolist()
        elite_idx_tensor = sorted_indices[:num_elites]
        
        # 進行進化更新
        new_w1 = self.w1.clone()
        new_w2 = self.w2.clone()
        
        for i in range(POP_SIZE):
            if i not in self.elite_indices:
                # 隨機挑選一個精英作為父母
                parent_idx = elite_idx_tensor[random.randint(0, num_elites - 1)]
                # 繼承並加入突變
                new_w1[i] = self.w1[parent_idx] + torch.randn_like(self.w1[i]) * 0.15
                new_w2[i] = self.w2[parent_idx] + torch.randn_like(self.w2[i]) * 0.15
                # 失敗者位置重置
                self.pos[i] = torch.rand(2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)
        
        self.w1, self.w2 = new_w1, new_w2
        self.fitness *= 0 # 重置適應度，重新評估新一代
        self.last_evolution_time = time.time()
        print(f"Generation Evolved! Top Fitness: {torch.max(self.fitness).item()}")

    def run(self):
        running = True
        while running:
            current_time = time.time()
            time_left = EVOLUTION_INTERVAL - (current_time - self.last_evolution_time)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False

            # 定時進化觸發
            if time_left <= 0:
                self.evolve()
                time_left = EVOLUTION_INTERVAL

            self.update_physics()
            
            # --- 渲染 ---
            self.screen.fill((20, 20, 25))
            
            # 畫食物
            for f in self.food_pos.cpu().numpy():
                pygame.draw.circle(self.screen, (255, 60, 60), f.astype(int), 3)
            
            # 畫生物
            pos_np = self.pos.cpu().numpy()
            for i, p in enumerate(pos_np):
                if i in self.elite_indices:
                    # 精英：金色 + 較大 + 光環
                    pygame.draw.circle(self.screen, (255, 215, 0), p.astype(int), 6)
                    pygame.draw.circle(self.screen, (255, 215, 0), p.astype(int), 10, 1) # 光環
                else:
                    # 普通：綠色
                    pygame.draw.circle(self.screen, (46, 204, 113), p.astype(int), 3)

            # UI 資訊
            timer_text = self.font.render(f"Next Evolution in: {int(time_left)}s", True, (255, 255, 255))
            elite_text = self.font.render(f"Elites (Yellow): {int(POP_SIZE*0.2)} units", True, (255, 215, 0))
            self.screen.blit(timer_text, (20, 20))
            self.screen.blit(elite_text, (20, 50))
                
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()

if __name__ == "__main__":
    sim = EvolutionSim()
    sim.run()