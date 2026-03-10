import pygame
import torch
import numpy as np
import random
import time
import os

# --- 參數設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POP_SIZE = 100          # 生物數量
FOOD_SIZE = 100         # 食物數量
PREDATOR_SIZE = 3       # 掠食者數量
SCREEN_W, SCREEN_H = 1024, 768
FPS = 240
EVOLUTION_INTERVAL = 30 # 固定每n秒進化一次
SAVE_PATH = "survivors.pt" # 存檔路徑

# 輸入層增加到 9：[食物cos, sin, 食物dist, X, Y, 速度X, 速度Y, 掠食者cos, 掠食者sin]
INPUT_SIZE = 9 
HIDDEN_SIZE = 16 # 稍微增加隱藏層神經元以處理更複雜的邏輯
OUTPUT_SIZE = 2

class EvolutionSim:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption(f"NN Evolution - {DEVICE.type.upper()}")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.big_font = pygame.font.SysFont("Arial", 28, bold=True)
        
        # 基本屬性初始化
        self.generation_count = 1
        self.last_evolution_time = time.time()
        self.elite_indices = []

        # 初始化掠食者 (Predators) - 隨機移動，不參與進化
        self.pred_pos = torch.rand(PREDATOR_SIZE, 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)
        self.pred_vel = (torch.rand(PREDATOR_SIZE, 2).to(DEVICE) - 0.5) * 4.0

        if not self.load_state():
            self.init_new_simulation()

    def init_new_simulation(self):
        self.pos = torch.rand(POP_SIZE, 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)
        self.vel = torch.zeros(POP_SIZE, 2).to(DEVICE)
        self.angle = torch.rand(POP_SIZE).to(DEVICE) * 2 * np.pi
        self.fitness = torch.zeros(POP_SIZE).to(DEVICE)
        self.survival_age = torch.zeros(POP_SIZE, dtype=torch.int).to(DEVICE)
        self.food_pos = torch.rand(FOOD_SIZE, 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)
        
        # 初始化神經網路權重 (CUDA)
        self.w1 = torch.randn(POP_SIZE, INPUT_SIZE, HIDDEN_SIZE).to(DEVICE)
        self.w2 = torch.randn(POP_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)

    def save_state(self):
        """儲存當前所有關鍵狀態到檔案"""
        state = {
            'generation_count': self.generation_count,
            'pos': self.pos, 'vel': self.vel, 'angle': self.angle,
            'fitness': self.fitness, 'survival_age': self.survival_age,
            'w1': self.w1, 'w2': self.w2, 'food_pos': self.food_pos,
            'pred_pos': self.pred_pos, 'elite_indices': self.elite_indices
        }
        torch.save(state, SAVE_PATH)
        print(f"\n[System] State saved to {SAVE_PATH} at Generation {self.generation_count}")

    def load_state(self):
        if os.path.exists(SAVE_PATH):
            try:
                state = torch.load(SAVE_PATH, map_location=DEVICE)
                self.generation_count = state['generation_count']
                self.pos, self.vel, self.angle = state['pos'], state['vel'], state['angle']
                self.fitness, self.survival_age = state['fitness'], state['survival_age']
                self.w1, self.w2 = state['w1'], state['w2']
                self.food_pos, self.pred_pos = state['food_pos'], state['pred_pos']
                self.elite_indices = state['elite_indices']
                print(f"[System] Successfully loaded save: Generation {self.generation_count}")
                return True
            except Exception as e:
                print(f"[Error] Failed to load save: {e}")
                return False
        return False

    def get_sensor_data(self):
        # 1. 食物資訊
        dist_to_food = torch.cdist(self.pos, self.food_pos)
        min_f_dist, min_f_idx = torch.min(dist_to_food, dim=1)
        target_food = self.food_pos[min_f_idx]
        dir_to_food = target_food - self.pos
        ang_to_food = torch.atan2(dir_to_food[:, 1], dir_to_food[:, 0]) - self.angle

        # 2. 掠食者資訊
        dist_to_pred = torch.cdist(self.pos, self.pred_pos)
        min_p_dist, min_p_idx = torch.min(dist_to_pred, dim=1)
        target_pred = self.pred_pos[min_p_idx]
        dir_to_pred = target_pred - self.pos
        ang_to_pred = torch.atan2(dir_to_pred[:, 1], dir_to_pred[:, 0]) - self.angle

        # 3. 組合感測向量
        sensors = torch.stack([
            torch.cos(ang_to_food), torch.sin(ang_to_food), 
            torch.clamp(min_f_dist / 500.0, 0, 1),
            (self.pos[:, 0] / SCREEN_W) * 2 - 1, (self.pos[:, 1] / SCREEN_H) * 2 - 1,
            self.vel[:, 0] * 0.1, self.vel[:, 1] * 0.1, # 自身速度
            torch.cos(ang_to_pred), torch.sin(ang_to_pred) # 掠食者方向
        ], dim=1)
        return sensors

    def update_physics(self):
        # 掠食者移動 (隨機漫遊)
        self.pred_pos += self.pred_vel
        # 掠食者邊界反彈
        mask_x = (self.pred_pos[:, 0] < 0) | (self.pred_pos[:, 0] > SCREEN_W)
        mask_y = (self.pred_pos[:, 1] < 0) | (self.pred_pos[:, 1] > SCREEN_H)
        self.pred_vel[mask_x, 0] *= -1
        self.pred_vel[mask_y, 1] *= -1

        # 生物決策
        sensors = self.get_sensor_data()
        
        # 神經網路前向傳播 (批次運算) x = tanh(Input * W1) -> tanh(x * W2)
        # 1. 矩陣相乘：Input * W1
        # sensors.unsqueeze(1) 將 (100, 5) 變成 (100, 1, 5) 以進行批次矩陣乘法
        h = torch.tanh(torch.bmm(sensors.unsqueeze(1), self.w1))
        # 2. 矩陣相乘：Hidden * W2
        out = torch.tanh(torch.bmm(h, self.w2)).squeeze(1)
        
        # 輸出控制：[前進力量, 轉向速度]
        speed = (out[:, 0] + 1) * 0.5 # 確保主要向前運動
        self.angle += out[:, 1] * 0.15
        new_vel = torch.stack([torch.cos(self.angle), torch.sin(self.angle)], dim=1) * speed.unsqueeze(1)
        self.vel = self.vel * 0.95 + new_vel # 慣性與阻尼
        self.pos += self.vel
        
        # 邊界反彈 (簡單物理)
        self.pos[:, 0] = torch.clamp(self.pos[:, 0], 0, SCREEN_W)
        self.pos[:, 1] = torch.clamp(self.pos[:, 1], 0, SCREEN_H)

        # 吃到食物檢測
        dist_f = torch.cdist(self.pos, self.food_pos)
        eaten = dist_f < 10.0
        for i in range(POP_SIZE):
            food_hit = torch.where(eaten[i])[0]
            if len(food_hit) > 0:
                self.fitness[i] += len(food_hit) * 10
                self.food_pos[food_hit] = torch.rand(len(food_hit), 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)

        # 2. 被掠食者碰觸懲罰 (-10 每幀)
        dist_p = torch.cdist(self.pos, self.pred_pos)
        attacked = dist_p < 25.0 # 掠食者攻擊範圍較大
        self.fitness[attacked.any(dim=1)] -= 50

    def evolve(self):
        # 找出前 20% 精英
        num_elites = int(POP_SIZE * 0.2)
        sorted_indices = torch.argsort(self.fitness, descending=True)
        current_elites = sorted_indices[:num_elites]
        self.elite_indices = current_elites.cpu().numpy().tolist()
        
        # 更新存活代數：精英 +1，其餘重置為 0
        new_survival_age = torch.zeros(POP_SIZE, dtype=torch.int).to(DEVICE)
        new_survival_age[current_elites] = self.survival_age[current_elites] + 1
        self.survival_age = new_survival_age
        
        # 進行進化更新
        new_w1 = self.w1.clone()
        new_w2 = self.w2.clone()
        for i in range(POP_SIZE):
            if i not in self.elite_indices:
                # 隨機挑選一個精英作為父母
                parent_idx = current_elites[random.randint(0, num_elites - 1)]
                # 繼承並加入突變
                new_w1[i] = self.w1[parent_idx] + torch.randn_like(self.w1[i]) * 0.15
                new_w2[i] = self.w2[parent_idx] + torch.randn_like(self.w2[i]) * 0.15
                # 失敗者位置重置
                self.pos[i] = torch.rand(2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)
                self.angle[i] = random.random() * 2 * np.pi
        
        self.w1, self.w2 = new_w1, new_w2
        self.fitness *= 0 # 重置適應度，重新評估新一代
        self.generation_count += 1
        self.last_evolution_time = time.time()
        # 進化完成後也自動存檔一次，防止意外中斷
        self.save_state()

    def run(self):
        running = True
        while running:
            # 定時進化觸發
            time_left = EVOLUTION_INTERVAL - (time.time() - self.last_evolution_time)
            if time_left <= 0:
                self.evolve()
                time_left = EVOLUTION_INTERVAL

            self.update_physics()
            self.screen.fill((20, 20, 25))
            
            # 畫食物
            for f in self.food_pos.cpu().numpy():
                pygame.draw.circle(self.screen, (255, 60, 60), f.astype(int), 3)
            
            # 繪製掠食者 (大白圓)
            for p in self.pred_pos.cpu().numpy():
                pygame.draw.circle(self.screen, (255, 255, 255), p.astype(int), 15, 2)
                pygame.draw.circle(self.screen, (255, 50, 50), p.astype(int), 5)

            # 繪製生物
            pos_np = self.pos.cpu().numpy()
            age_np = self.survival_age.cpu().numpy()
            for i, p in enumerate(pos_np):
                if i in self.elite_indices:
                    age = age_np[i]
                    # 超過 3 代的長青精英變換顏色 (紫色)
                    color = (200, 100, 255) if age >= 3 else (255, 215, 0)
                    pygame.draw.circle(self.screen, color, p.astype(int), 7)
                    pygame.draw.circle(self.screen, color, p.astype(int), 12, 1) # 外環

                    # 顯示年齡標註
                    age_surf = self.font.render(f"{age}", True, color)
                    self.screen.blit(age_surf, (p[0] - 5, p[1] - 25))
                else:
                    # 普通：綠色
                    pygame.draw.circle(self.screen, (46, 204, 113), p.astype(int), 4)

            # UI 資訊
            gen_text = self.big_font.render(f"GENERATION: {self.generation_count}", True, (255, 255, 255))
            self.screen.blit(gen_text, (20, 20))
            timer_text = self.font.render(f"Next Evolution in: {int(time_left)}s", True, (200, 200, 200))
            self.screen.blit(timer_text, (20, 60))
            fps_text = self.font.render(f"FPS: {int(self.clock.get_fps())}", True, (0, 255, 0))
            self.screen.blit(fps_text, (20, 90))
            save_info = self.font.render("Auto-saves on exit or evolution", True, (100, 100, 150))
            self.screen.blit(save_info, (20, 1150))
                
            pygame.display.flip()
            self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

        # 迴圈結束，執行存檔
        self.save_state()
        pygame.quit()

if __name__ == "__main__":
    sim = EvolutionSim()
    sim.run()