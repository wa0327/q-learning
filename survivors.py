import pygame
import torch
import numpy as np
import random

# --- 參數設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POP_SIZE = 10          # 生物數量
FOOD_SIZE = 100         # 食物數量
SCREEN_W, SCREEN_H = 1600, 1200
FPS = 60

# 神經網路結構：輸入(5) -> 隱藏(8) -> 輸出(2)
# 輸入：[前方距離, 左方距離, 右方距離, 食物相對角度, 剩餘能量]
INPUT_SIZE = 5
HIDDEN_SIZE = 8
OUTPUT_SIZE = 2 # [加速度, 轉向力]

class EvolutionSim:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        
        # 初始化生物屬性 (Tensor 運算提升效率)
        self.pos = torch.rand(POP_SIZE, 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).to(DEVICE)
        self.vel = torch.zeros(POP_SIZE, 2).to(DEVICE)
        self.angle = torch.rand(POP_SIZE).to(DEVICE) * 2 * np.pi
        self.energy = torch.ones(POP_SIZE).to(DEVICE) * 100.0
        self.fitness = torch.zeros(POP_SIZE).to(DEVICE)
        
        # 初始化食物
        self.food_pos = torch.rand(FOOD_SIZE, 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).to(DEVICE)
        
        # 初始化神經網路權重 (CUDA)
        self.w1 = torch.randn(POP_SIZE, INPUT_SIZE, HIDDEN_SIZE).to(DEVICE)
        self.w2 = torch.randn(POP_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)

    def get_sensor_data(self):
        # 簡化版感測：計算最近食物的方向與距離
        # 在實際工程中，這裡可以使用 Raycasting，此處為求運行流暢先用向量計算
        dist_to_food = torch.cdist(self.pos, self.food_pos) # (POP, FOOD)
        min_dist, min_idx = torch.min(dist_to_food, dim=1)
        target_food = self.food_pos[min_idx]
        
        dir_to_food = target_food - self.pos
        angle_to_food = torch.atan2(dir_to_food[:, 1], dir_to_food[:, 0]) - self.angle
        
        # 構建輸入向量 (Normalize)
        sensors = torch.stack([
            torch.cos(angle_to_food), 
            torch.sin(angle_to_food),
            min_dist / SCREEN_W,
            self.pos[:, 0] / SCREEN_W, # 邊界感知
            self.pos[:, 1] / SCREEN_H
        ], dim=1)
        return sensors

    def update_brains(self):
        sensors = self.get_sensor_data()
        
        # 神經網路前向傳播 (批次運算)
        # x = tanh(Input * W1) -> tanh(x * W2)
        h = torch.tanh(torch.bmm(sensors.unsqueeze(1), self.w1))
        out = torch.tanh(torch.bmm(h, self.w2)).squeeze(1)
        
        # 輸出應用：out[:, 0] 是加速度, out[:, 1] 是轉向
        accel = out[:, 0] * 0.5
        self.angle += out[:, 1] * 0.2
        
        # 更新物理狀態
        new_vel = torch.stack([torch.cos(self.angle), torch.sin(self.angle)], dim=1) * accel.unsqueeze(1)
        self.vel = self.vel * 0.95 + new_vel # 慣性與阻尼
        self.pos += self.vel
        
        # 邊界處理
        self.pos[:, 0] = torch.clamp(self.pos[:, 0], 0, SCREEN_W)
        self.pos[:, 1] = torch.clamp(self.pos[:, 1], 0, SCREEN_H)

    def evaluate_and_evolve(self):
        # 碰撞檢測：生物與食物
        dist_to_food = torch.cdist(self.pos, self.food_pos)
        eaten_mask = dist_to_food < 10.0
        
        for i in range(POP_SIZE):
            eaten_indices = torch.where(eaten_mask[i])[0]
            if len(eaten_indices) > 0:
                self.fitness[i] += len(eaten_indices) * 50
                # 重置食物位置
                self.food_pos[eaten_indices] = torch.rand(len(eaten_indices), 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).to(DEVICE)

        # 簡單的代際進化 (每隔段時間或能量耗盡執行)
        # 這裡為了展示，我們每一千幀執行一次大隨機變異模擬進化
        if random.random() < 0.005:
            top_idx = torch.argsort(self.fitness, descending=True)[:int(POP_SIZE*0.2)]
            # 讓優秀個體覆蓋平庸個體並加入突變
            for i in range(POP_SIZE):
                if i not in top_idx:
                    parent_idx = random.choice(top_idx)
                    self.w1[i] = self.w1[parent_idx] + torch.randn_like(self.w1[i]) * 0.1
                    self.w2[i] = self.w2[parent_idx] + torch.randn_like(self.w2[i]) * 0.1
                    self.pos[i] = torch.rand(2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).to(DEVICE)
            self.fitness *= 0 # 重置適應度

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False

            self.update_brains()
            self.evaluate_and_evolve()
            
            # 渲染
            self.screen.fill((30, 30, 35)) # 深灰色背景
            
            # 畫食物
            for f in self.food_pos.cpu().numpy():
                pygame.draw.circle(self.screen, (255, 50, 50), f.astype(int), 3)
            
            # 畫生物
            pos_np = self.pos.cpu().numpy()
            for p in pos_np:
                pygame.draw.circle(self.screen, (50, 255, 150), p.astype(int), 4)
                
            pygame.display.flip()
            self.clock.tick(FPS)
            pygame.display.set_caption(f"Evolution Sim - Device: {DEVICE} - FPS: {self.clock.get_fps():.1f}")

        pygame.quit()

if __name__ == "__main__":
    sim = EvolutionSim()
    sim.run()