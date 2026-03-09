import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque

# --- 0. 自動偵測環境：優先使用 CUDA GPU ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"目前使用的訓練設備為: {device}")

# --- 1. 定義神經網路 (AI 的大腦) ---
# 這個網路負責「預測」：輸入目前座標，輸出往四個方向走的「期望總分」(Q值)
class QNet(nn.Module):
    def __init__(self):
        super(QNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 128),  # 輸入層：接受 x, y 兩個座標數值
            nn.ReLU(),         # 激活函數：增加非線性表達能力，模擬神經元啟動
            nn.Linear(128, 128), # 隱藏層：負責提取空間中的特徵關聯
            nn.ReLU(),
            nn.Linear(128, 4)   # 輸出層：對應 上、下、左、右 四個動作的預估價值
        )
    def forward(self, x):
        return self.network(x)

# --- 2. 初始化訓練組件 ---
model = QNet().to(device) # 實例化 AI 大腦
optimizer = optim.Adam(model.parameters(), lr=0.001) # 優化器：負責根據誤差調整網路權重
memory = deque(maxlen=5000) # 經驗回放池：記錄過去的經歷，實現您要求的「連帶優化」
gamma = 0.98    # 折扣因子：決定 AI 多看重「未來」的獎勵 (0.95 表示高度看重遠期利益)
epsilon = 0.2   # 探索率：20% 的機率隨機亂走，用來發現新路徑

# --- 3. 核心訓練邏輯：連帶優化 (Experience Replay) ---
def train_step(batch_size=64):
    if len(memory) < batch_size: return # 記憶不夠多時先不訓練
    
    # 從記憶池隨機抽取一批過去的經歷，打破時間上的連續性，讓學習更穩定
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    # 將數據轉換為 PyTorch 張量 (Tensor) 以進行數學運算
    states = torch.stack(states).to(device)
    next_states = torch.stack(next_states).to(device)
    actions = torch.tensor(actions).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    # 計算「目前預測」與「實際獎勵 + 未來預期」之間的差距
    current_q = model(states).gather(1, actions).squeeze() # 目前大腦對該動作的估值

    with torch.no_grad(): # 下一步的預測不需要計算梯度，節省 GPU 顯存
        max_next_q = model(next_states).max(1)[0]     # 下一步狀態中，大腦認為最強的價值
    
    target_q = rewards + (1 - dones) * gamma * max_next_q  # Bellman 方程：理想的目標價值

    # 計算誤差 (MSE) 並回傳給神經網路進行修正 (Backpropagation)
    loss = nn.MSELoss()(current_q, target_q)
    optimizer.zero_grad() # 清空舊梯度
    loss.backward()       # 反向傳播計算權重調整方向
    optimizer.step()      # 正式更新權重

# --- 4. 視覺化函數：呈現學習效果 ---
def plot_learning_effect(episode, path):
    plt.clf()
    # 建立一個網格，探測神經網路對整個空間的「價值評價」
    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 1, 30)
    v_map = np.zeros((30, 30))
    
    model.eval() # 切換為評估模式
    with torch.no_grad():
        for i in range(30):
            for j in range(30):
                st = torch.tensor([x[j], y[i]], dtype=torch.float32).to(device)
                v_map[i, j] = model(st).max().item() # 取得該位置最高動作價值
    model.train() # 切換回訓練模式

    # 繪製價值熱圖：藍色代表低價值，紅色代表高價值目標區
    plt.imshow(v_map, extent=[0,1,0,1], origin='lower', cmap='magma', alpha=0.8)
    path = np.array(path)
    plt.plot(path[:,0], path[:,1], 'w-', linewidth=2, label='Current Path') # 繪製 AI 本次走的軌跡
    plt.scatter([1], [1], c='cyan', s=150, edgecolors='white', label='Goal')        # 標出終點
    plt.title(f"Episode {episode} - Value Distribution & Path")
    plt.colorbar(label='Predicted Value (Q)')
    plt.pause(0.01)

# --- 5. 主訓練迴圈 (訓練 100 局) ---
plt.ion() # 開啟互動模式以即時更新圖表
for ep in range(201):
    state = torch.tensor([0.0, 0.0], dtype=torch.float32) # 每局從起點 (0,0) 開始
    path = [state.numpy()] # 記錄路徑以供繪圖
    
    for t in range(500): # 每局限制最多步數，避免 AI 無限徘徊
        # 決定動作：Epsilon-Greedy 策略 (兼顧探索與利用)
        if random.random() < epsilon: 
            action = random.randint(0, 3) # 隨機亂猜
        else: 
            with torch.no_grad():
                # 預測時需將 state 送入 GPU
                action = model(state.to(device)).argmax().item() # 根據大腦目前的經驗選最優動作

        # 模擬物理位移 (連續空間)
        next_state = state.clone()
        step_size = 0.05
        if action == 0: next_state[1] += step_size # 向上
        elif action == 1: next_state[1] -= step_size # 向下
        elif action == 2: next_state[0] -= step_size # 向左
        elif action == 3: next_state[0] += step_size # 向右
        next_state = torch.clamp(next_state, 0, 1) # 限制在 0.0 ~ 1.0 的範圍內 (牆壁限制)

        # 獎勵設計：越靠近終點，未來回報越高
        dist = torch.dist(next_state, torch.tensor([1.0, 1.0]))
        done = dist < 0.1 # 判斷是否抵達終點範圍
        reward = 20.0 if done else -0.05 # 到達終點給大獎，每走一步扣小分 (鼓勵最短路徑)
        
        # 存入記憶池
        memory.append((state, action, reward, next_state, done))
        state = next_state
        path.append(state.numpy())
        
        train_step(batch_size=128) # 執行一次神經網路學習，這會連帶優化相關的空間權重
        if done: break

    # 每 20 局更新一次圖表，觀察 AI 變聰明的過程
    if ep % 1 == 0:
        plot_learning_effect(ep, path)
        print(f"Episode {ep} 完成，路徑步數: {len(path)}")

plt.ioff()
plt.show()