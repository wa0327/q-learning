import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- 參數設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BRAIN_PATH = "pretrain3.pt"
BATCH_SIZE = 256
EPOCHS = 10000

# --- 與 survivors3.py 相同的網路架構 ---
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.conv = nn.Conv1d(6, 32, 1)
        self.fc = nn.Sequential(
            nn.Linear(35, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh()
        )

    def forward(self, m_in, s_in):
        x = torch.max(F.relu(self.conv(m_in)), dim=2)[0]
        combined = torch.cat([x, s_in], dim=1)
        return self.fc(combined)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.conv = nn.Conv1d(6, 32, 1)
        self.fc = nn.Sequential(
            nn.Linear(35 + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, m_in, s_in, action):
        x = torch.max(F.relu(self.conv(m_in)), dim=2)[0]
        combined = torch.cat([x, s_in, action], dim=1)
        return self.fc(combined)

# --- 專家邏輯 (Teacher) ---
def expert_logic(m_in, s_in):
    """
    根據假造的輸入狀態，決定最佳的轉向與油門
    m_in 結構 (6 個 Channel):
      0: 食物 Cos, 1: 食物 Sin, 2: 食物距離
      3: 掠食者 Cos, 4: 掠食者 Sin, 5: 掠食者壓迫感 (0~2)
    s_in 結構:
      0: 速度, 1: 0 (佔位), 2: 能量比例 (0~1)
    """
    batch_size = m_in.shape[0]
    targets = torch.zeros(batch_size, 2).to(DEVICE)
    
    # 提取特徵 (假設 index 0 是最近的目標)
    best_food_sin = m_in[:, 1, 0]        
    worst_pred_sin = m_in[:, 4, 0]       
    worst_pred_pressure = m_in[:, 5, 0]  
    energy_ratio = s_in[:, 2]            

    # 計算風險容忍度：越餓 (energy_ratio 越低)，越敢冒險不逃跑
    risk_tolerance = 0.8 + (1.0 - energy_ratio) * 0.6
    evade_mask = worst_pred_pressure > risk_tolerance
    
    # --- 1. 逃跑邏輯 (遇到危險) ---
    # 動作 0 (Steer): 往掠食者的反方向轉彎
    targets[evade_mask, 0] = -worst_pred_sin[evade_mask] * 2.0 
    # 動作 1 (Throttle): 油門全開 (1.0) 逃命
    targets[evade_mask, 1] = 1.0                               

    # --- 2. 覓食邏輯 (安全時) ---
    # 動作 0 (Steer): 往食物方向轉彎
    targets[~evade_mask, 0] = best_food_sin[~evade_mask] * 1.5 
    # 動作 1 (Throttle): 保持中等速度 (-0.2 對應輕微加速，根據你主程式 mapping (x+1)*0.25)
    targets[~evade_mask, 1] = -0.2                             
    
    # 限制動作範圍在 [-1, 1] 以符合 Tanh 輸出
    return targets.clamp(-1.0, 1.0)

# --- 執行預訓練 ---
def run_pretrain():
    actor = Actor().to(DEVICE)
    critic = Critic().to(DEVICE)
    
    if os.path.exists(BRAIN_PATH):
        try:
            brain_state = torch.load(BRAIN_PATH, map_location=DEVICE)
            actor.load_state_dict(brain_state['actor'])
            critic.load_state_dict(brain_state['critic'])
            print(f"--- [Loaded] Brain weights. ---")
        except Exception as e:
            print(f"--- [Error] Brain loading failed: {e} ---")

    optimizer = optim.Adam(actor.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    print(f"--- [Pretrain] Training expert Actor to {BRAIN_PATH} ---")
    
    for i in range(EPOCHS):
        # 隨機生成符合維度的 Dummy 資料 (Batch, Channels, Objects)
        m_in = torch.randn(BATCH_SIZE, 6, 5).to(DEVICE)
        
        # 修正生成的資料範圍，使其更符合物理現實
        m_in[:, 0:2, :] = torch.clamp(m_in[:, 0:2, :], -1.0, 1.0) # Cos/Sin 落在 -1~1
        m_in[:, 3:5, :] = torch.clamp(m_in[:, 3:5, :], -1.0, 1.0) # Cos/Sin 落在 -1~1
        m_in[:, 5, :] = torch.clamp(torch.abs(m_in[:, 5, :]), 0.0, 2.0) # 壓迫感 0~2
        
        # s_in (Batch, 3)
        s_in = torch.rand(BATCH_SIZE, 3).to(DEVICE)
        s_in[:, 1] = 0.0 # 中間的佔位符保持為 0
        
        # 取得專家目標動作
        target = expert_logic(m_in, s_in)
        
        # 反向傳播與參數更新
        optimizer.zero_grad()
        output = actor(m_in, s_in)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # 顯示進度
        if (i + 1) % 5000 == 0:
            print(f"Progress: {(i+1)/EPOCHS*100:3.0f}% | Loss: {loss.item():.6f}")

    # 存檔
    torch.save({
        'actor': actor.state_dict(),
        'critic': critic.state_dict()
    }, BRAIN_PATH)
    print("--- Pretrain Complete. Weights properly saved! ---")

if __name__ == "__main__":
    run_pretrain()
