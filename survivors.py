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
PREDATOR_SIZE = 10       # 增加掠食者數量，強化生存壓力
SCREEN_W, SCREEN_H = 1024, 768
FPS = 240
EVOLUTION_TICKS = 2000 # 固定每n幀進化一次
SAVE_PATH = "survivors.pt" # 存檔路徑
MAX_ENERGY = 100.0
ENERGY_DECAY = 0.15 # 每幀扣除能量 (數值越高，進食壓力越大)

# 輸入層增加到 11：[食物cos, sin, 食物dist, X, Y, 速度X, 速度Y, 掠食者cos, 掠食者sin, 掠食者壓迫感]
INPUT_SIZE = 11 
HIDDEN_SIZE = 16 
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
        self.evolution_ticks = EVOLUTION_TICKS
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
        self.alive = torch.ones(POP_SIZE, dtype=torch.bool).to(DEVICE)
        self.energy = torch.full((POP_SIZE,), MAX_ENERGY).to(DEVICE)

        # --- 專家基因注入邏輯 ---
        expert_path = "pretrain.pt"
        if os.path.exists(expert_path):
            print(f"[System] Injecting Expert Seed from {expert_path}...")
            expert_data = torch.load(expert_path, map_location=DEVICE)
            base_w1 = expert_data['w1'] # (INPUT, HIDDEN)
            base_w2 = expert_data['w2'] # (HIDDEN, OUTPUT)

            # 擴散到 100 個個體
            # 其中 20% 保持專家原樣 (純種)，80% 加上突變 (探索者)
            new_w1 = base_w1.unsqueeze(0).repeat(POP_SIZE, 1, 1)
            new_w2 = base_w2.unsqueeze(0).repeat(POP_SIZE, 1, 1)

            # 對後 80% 進行變異
            mutation_mask = int(POP_SIZE * 0.2)
            noise_indices = range(mutation_mask, POP_SIZE)
            for i in noise_indices:
                new_w1[i] += torch.randn_like(new_w1[i]) * 0.2 # 較大的突變強度
                new_w2[i] += torch.randn_like(new_w2[i]) * 0.2
            
            self.w1 = new_w1.to(DEVICE)
            self.w2 = new_w2.to(DEVICE)
        else:
            print("[System] No expert seed found. Initializing with random weights.")
            self.w1 = torch.randn(POP_SIZE, INPUT_SIZE, HIDDEN_SIZE).to(DEVICE)
            self.w2 = torch.randn(POP_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)

    def save_state(self):
        state = {
            'generation_count': self.generation_count,
            'pos': self.pos, 'vel': self.vel, 'angle': self.angle,
            'fitness': self.fitness, 'survival_age': self.survival_age,
            'alive': self.alive, 'energy': self.energy,
            'w1': self.w1, 'w2': self.w2, 'food_pos': self.food_pos,
            'pred_pos': self.pred_pos, 'elite_indices': self.elite_indices
        }
        torch.save(state, SAVE_PATH)
        print(f"[System] State saved to {SAVE_PATH} at Generation {self.generation_count}")

    def load_state(self):
        if os.path.exists(SAVE_PATH):
            try:
                state = torch.load(SAVE_PATH, map_location=DEVICE)
                self.generation_count = state['generation_count']
                self.pos, self.vel, self.angle = state['pos'], state['vel'], state['angle']
                self.fitness, self.survival_age = state['fitness'], state['survival_age']
                self.alive, self.energy = state['alive'], state['energy']
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
        # 1. 最近食物
        dist_to_food = torch.cdist(self.pos, self.food_pos)
        min_f_dist, min_f_idx = torch.min(dist_to_food, dim=1)
        target_food = self.food_pos[min_f_idx]
        dir_to_food = target_food - self.pos
        ang_to_food = torch.atan2(dir_to_food[:, 1], dir_to_food[:, 0]) - self.angle

        # 2. 最近掠食者
        dist_to_pred = torch.cdist(self.pos, self.pred_pos)
        min_p_dist, min_p_idx = torch.min(dist_to_pred, dim=1)
        target_pred = self.pred_pos[min_p_idx]
        dir_to_pred = target_pred - self.pos
        ang_to_pred = torch.atan2(dir_to_pred[:, 1], dir_to_pred[:, 0]) - self.angle

        # 3. 壓迫感計算：距離越近，值越大 (200/dist)
        pressure = torch.clamp(200.0 / (min_p_dist + 1.0), 0, 2)

        # 4. 當前能量比例 (1.0 = 滿載, 0.0 = 瀕死)
        energy_ratio = self.energy / MAX_ENERGY

        sensors = torch.stack([
            torch.cos(ang_to_food), torch.sin(ang_to_food), 
            torch.clamp(min_f_dist / 500.0, 0, 1),
            (self.pos[:, 0] / SCREEN_W) * 2 - 1, (self.pos[:, 1] / SCREEN_H) * 2 - 1,
            self.vel[:, 0] * 0.1, self.vel[:, 1] * 0.1,
            torch.cos(ang_to_pred), torch.sin(ang_to_pred),
            pressure,
            energy_ratio
        ], dim=1)
        return sensors

    def update_physics(self):
        active_mask = self.alive == True
        if not active_mask.any():
            return 

        # 1. 能量消耗
        self.energy[active_mask] -= ENERGY_DECAY
        
        # 2. 餓死檢測
        starved = (self.energy <= 0) & active_mask
        if starved.any():
            self.alive[starved] = False
            self.fitness[starved] -= 100 # 餓死的懲罰比撞死重，強制它必須去探索
            self.energy[starved] = 0
            
        # 掠食者移動邏輯
        self.pred_pos += self.pred_vel
        mask_x = (self.pred_pos[:, 0] < 0) | (self.pred_pos[:, 0] > SCREEN_W)
        mask_y = (self.pred_pos[:, 1] < 0) | (self.pred_pos[:, 1] > SCREEN_H)
        self.pred_vel[mask_x, 0] *= -1
        self.pred_vel[mask_y, 1] *= -1

        # NN 運算
        sensors = self.get_sensor_data() 
        h = torch.tanh(torch.bmm(sensors.unsqueeze(1), self.w1))
        out = torch.tanh(torch.bmm(h, self.w2)).squeeze(1)
        
        # 物理更新
        self.angle[active_mask] += out[active_mask, 1] * 0.15
        current_speed_vol = (out[active_mask, 0] + 1) * 0.1 
        move_dir = torch.stack([torch.cos(self.angle[active_mask]), torch.sin(self.angle[active_mask])], dim=1)
        
        # 應用阻尼與推力
        self.vel[active_mask] = self.vel[active_mask] * 0.95 + move_dir * current_speed_vol.unsqueeze(1)
        self.pos[active_mask] += self.vel[active_mask]
        
        # 邊界
        self.pos[:, 0] = torch.clamp(self.pos[:, 0], 0, SCREEN_W)
        self.pos[:, 1] = torch.clamp(self.pos[:, 1], 0, SCREEN_H)

        # 生存獎勵：只要活著，每一幀給予微量加分
        self.fitness[active_mask] += 0.01

        # 檢測碰撞
        active_indices = torch.where(active_mask)[0]
        
        # 食物
        dist_f = torch.cdist(self.pos[active_mask], self.food_pos)
        eaten_mask = dist_f < 10.0
        for i_idx, real_idx in enumerate(active_indices):
            hits = torch.where(eaten_mask[i_idx])[0]
            if len(hits) > 0:
                self.fitness[real_idx] += len(hits) * 10
                self.energy[real_idx] = torch.clamp(self.energy[real_idx] + 30.0, max=MAX_ENERGY)
                self.food_pos[hits] = torch.rand(len(hits), 2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)

        # 掠食者與壓迫扣分
        dist_p = torch.cdist(self.pos[active_mask], self.pred_pos)
        
        # 壓迫感扣分：距離小於 80 開始每一幀扣分，鼓勵遠離
        close_to_pred = (dist_p < 80.0).any(dim=1)
        self.fitness[active_indices[close_to_pred]] -= 0.1

        # 直接碰撞出局
        killed_sub_mask = (dist_p < 25.0).any(dim=1)
        if killed_sub_mask.any():
            killed_real_indices = active_indices[killed_sub_mask]
            self.alive[killed_real_indices] = False
            self.fitness[killed_real_indices] -= 50

    def evolve(self):
        # 1. 取得排序索引 (從高分到低分)
        sorted_indices = torch.argsort(self.fitness, descending=True)
        
        # 2. 定義兩個候選池
        num_target = int(POP_SIZE * 0.2)
        positive_mask = self.fitness[sorted_indices] > 0
        elites_with_score = sorted_indices[positive_mask]
        
        # 判斷邏輯：如果有得分者，則僅取得分者（上限 20%）；若無得分者，則取前 20%
        if len(elites_with_score) > 0:
            # 僅取分數大於 0 的個體，且最多只取前 20% 的名額
            current_elites = elites_with_score[:num_target]
            print(f"[System] Prosperous Era: {len(current_elites)} elites with fitness > 0.")
        else:
            # 全員皆沒得分，強制取前 20% 作為種子
            current_elites = sorted_indices[:num_target]
            print(f"[System] Dark Age: No positive fitness. Using top 20% relative performers.")

        self.elite_indices = current_elites.cpu().numpy().tolist()
        num_found_elites = len(self.elite_indices)

        # 更新存活代數
        new_survival_age = torch.zeros(POP_SIZE, dtype=torch.int).to(DEVICE)
        new_survival_age[current_elites] = self.survival_age[current_elites] + 1
        self.survival_age = new_survival_age
        
        # 3. 準備新權重與平均分配
        new_w1 = self.w1.clone()
        new_w2 = self.w2.clone()
        non_elite_indices = [i for i in range(POP_SIZE) if i not in self.elite_indices]

        # 這裡的 num_found_elites 絕對會大於 0 (因為即使沒人得分也會取前 20%)
        offspring_per_elite = len(non_elite_indices) // num_found_elites
        
        for idx, parent_idx in enumerate(self.elite_indices):
            start = idx * offspring_per_elite
            # 處理最後一組餘數，確保填滿 100 個位置
            end = (idx + 1) * offspring_per_elite if idx < num_found_elites - 1 else len(non_elite_indices)
            target_group = non_elite_indices[start:end]
            for i in target_group:
                # 繼承並突變
                new_w1[i] = self.w1[parent_idx] + torch.randn_like(self.w1[i]) * 0.15
                new_w2[i] = self.w2[parent_idx] + torch.randn_like(self.w2[i]) * 0.15
                self.reset_agent(i)

        # 更新權重並重置環境
        self.w1, self.w2 = new_w1, new_w2
        self.alive[:] = True
        self.fitness *= 0 
        self.energy[:] = MAX_ENERGY
        self.generation_count += 1
        self.evolution_ticks = EVOLUTION_TICKS
        # 進化完成後也自動存檔一次，防止意外中斷
        self.save_state()
        
    def reset_agent(self, i):
        """輔助函式：重置個體狀態"""
        self.pos[i] = torch.rand(2).to(DEVICE) * torch.tensor([SCREEN_W, SCREEN_H]).float().to(DEVICE)
        self.vel[i] *= 0
        self.angle[i] = random.random() * 2 * np.pi

    def run(self):
        running = True
        while running:
            # 定時進化觸發
            self.evolution_ticks -= 1
            if self.evolution_ticks <= 0:
                self.evolve()
                self.evolution_ticks = EVOLUTION_TICKS

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
            alive_np = self.alive.cpu().numpy()
            for i, p in enumerate(pos_np):
                if not alive_np[i]:
                    # 死亡：畫成灰色，且不顯示年齡
                    pygame.draw.circle(self.screen, (80, 80, 80), p.astype(int), 3)
                    continue

                color = (200, 100, 255) if age_np[i] >= 3 else (255, 215, 0) if i in self.elite_indices else (46, 204, 113)
                radius = 7 if i in self.elite_indices else 4
                pygame.draw.circle(self.screen, color, p.astype(int), radius)
                if i in self.elite_indices:
                    pygame.draw.circle(self.screen, color, p.astype(int), radius+5, 1)

            # UI
            self.screen.blit(self.big_font.render(f"GEN: {self.generation_count}", True, (255,255,255)), (20, 20))
            self.screen.blit(self.font.render(f"Next: {int(self.evolution_ticks)}", True, (200,200,200)), (20, 60))
            self.screen.blit(self.font.render(f"Alive: {int(self.alive.sum())}/{POP_SIZE}", True, (100, 100, 150)), (20, 90))
            self.screen.blit(self.font.render(f"FPS: {int(self.clock.get_fps())}", True, (0, 255, 0)), (20, 120))
            
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