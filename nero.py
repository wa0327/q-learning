import torch
import torch.nn as nn
import numpy as np

# --- 1. 大腦架構設計 ---
class Brain(nn.Module):
    def __init__(self, input_size=5, hidden_size=8, output_size=2):
        super(Brain, self).__init__()
        # 使用簡單的全連接層
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size),
            nn.Tanh() # 輸出限制在 -1 到 1 之間
        )
        
        # 禁用梯度計算，因為我們用演化而非反向傳播
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.network(x)

# --- 2. 獎懲機制 (可輕易增刪條件) ---
def calculate_fitness(agent_data):
    """
    agent_data 包含: distance_to_target, steps_alive, hit_wall
    """
    reward = 0
    
    # 條件 1: 越靠近目標獎勵越高 (權重: 1.0)
    reward += (1.0 / (agent_data['dist'] + 0.1)) * 10 
    
    # 條件 2: 存活時間獎勵 (權重: 0.1)
    reward += agent_data['steps'] * 0.1
    
    # 條件 3: 碰撞懲罰
    if agent_data['hit_wall']:
        reward -= 50
        
    return reward

# --- 3. 演化管理器 (支援 CUDA) ---
class EvolutionManager:
    def __init__(self, pop_size=100, device='cuda'):
        self.device = device
        self.pop_size = pop_size
        # 初始化一群大腦並搬移到 GPU
        self.population = [Brain().to(device) for _ in range(pop_size)]
        
    def mutate(self, brain, rate=0.1):
        """ 對權重加入隨機噪聲來模擬突變 """
        new_brain = Brain().to(self.device)
        new_brain.load_state_dict(brain.state_dict())
        with torch.no_grad():
            for param in new_brain.parameters():
                noise = torch.randn(param.size()).to(self.device) * rate
                param.add_(noise)
        return new_brain

    def evolve(self, fitness_scores):
        # 根據分數排序，選出前 10% 作為精英
        top_indices = np.argsort(fitness_scores)[-int(self.pop_size*0.1):]
        elites = [self.population[i] for i in top_indices]
        
        new_population = []
        new_population.extend(elites) # 保留精英
        
        while len(new_population) < self.pop_size:
            parent = np.random.choice(elites)
            child = self.mutate(parent)
            new_population.append(child)
            
        self.population = new_population

# --- 4. 簡易模擬迴圈 (Demo) ---
def simulate():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    evo = EvolutionManager(pop_size=50, device=device)
    target_pos = torch.tensor([5.0, 5.0]).to(device)
    
    for gen in range(10): # 跑 10 代測試
        fitness_results = []
        
        for brain in evo.population:
            # 模擬個體在 2D 平面移動
            pos = torch.zeros(2).to(device)
            steps = 0
            
            for _ in range(50): # 每個個體給 50 步
                # 輸入：目前座標與目標座標
                input_data = torch.cat([pos, target_pos, torch.tensor([steps/50.0]).to(device)])
                action = brain(input_data) # 取得輸出 (dx, dy)
                pos += action * 0.5
                steps += 1
            
            # 計算該個體的分數
            dist = torch.norm(pos - target_pos).item()
            score = calculate_fitness({'dist': dist, 'steps': steps, 'hit_wall': False})
            fitness_results.append(score)
            
        print(f"Generation {gen} | Max Fitness: {max(fitness_results):.2f}")
        evo.evolve(fitness_results)

if __name__ == "__main__":
    simulate()