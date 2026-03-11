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
        3: 掠食者 Cos, 4: 掠食者 Sin, 5: 掠食者壓迫感 (0~10)
    s_in 結構:
        0: 速度, 1: 0 (佔位), 2: 能量比例 (0~1)
    """
    batch_size = m_in.shape[0]
    targets = torch.zeros(batch_size, 2).to(DEVICE)

    # --- 1. 計算掠食者的排斥力 (Repulsive Force) ---
    # 取得所有掠食者的 Cos (Ch 3), Sin (Ch 4) 與 壓迫感 (Ch 5)
    p_cos = m_in[:, 3, :] 
    p_sin = m_in[:, 4, :]
    p_pressure = m_in[:, 5, :] # 壓迫感越高，排斥力越強
    
    # 掠食者向量：方向是從掠食者指向自己 (所以 Cos/Sin 要取反)
    # 總排斥力向量 (X, Y)
    repel_x = torch.sum(-p_cos * p_pressure, dim=1)
    repel_y = torch.sum(-p_sin * p_pressure, dim=1)
    
    # --- 2. 計算食物的吸引力 (Attractive Force) ---
    # 取得食物 Cos (Ch 0), Sin (Ch 1) 與 距離 (Ch 2)
    f_cos = m_in[:, 0, :]
    f_sin = m_in[:, 1, :]
    f_dist = m_in[:, 2, :]
    
    # 吸引力隨距離衰減 (1 / (dist + eps))
    f_weight = 1.0 / (f_dist + 0.1)
    
    # 總吸引力向量 (X, Y)
    attract_x = torch.sum(f_cos * f_weight, dim=1)
    attract_y = torch.sum(f_sin * f_weight, dim=1)
    
    # --- 3. 綜合決策 ---
    energy_ratio = s_in[:, 2]
    # 根據能量調整權重：能量低時，對食物的吸引力權重增加
    food_importance = 1.2 - energy_ratio 
    # 根據總壓迫感決定是否進入「緊急逃跑模式」
    total_pressure = torch.sum(p_pressure, dim=1)
    
    # 合力向量 (Resultant Vector)
    res_x = repel_x + attract_x * food_importance
    res_y = repel_y + attract_y * food_importance
    
    # --- 4. 轉換為動作 ---
    # Steer: 使用 atan2 算出合力方向的角度，並對應到 [-1, 1]
    # 因為 Sin 在你的模型中代表 Y 軸偏角，通常 Steer 與 Sin 呈正相關
    target_steer = torch.atan2(res_y, res_x) / torch.pi # 映射到約 [-1, 1]
    
    # Throttle: 
    # 如果總壓迫感高，油門全開
    # 如果安全，保持穩定巡航
    target_throttle = torch.where(total_pressure > 1.0, 
                                  torch.tensor(1.0, device=DEVICE), 
                                  torch.tensor(-0.2, device=DEVICE))
    
    targets = torch.stack([target_steer, target_throttle], dim=1)
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
    
    for epoch in range(EPOCHS):
        # --- 修正: 產生符合真實物理法則的假資料 ---
        m_in = torch.zeros(BATCH_SIZE, 6, 5).to(DEVICE)
        
        # 生成真實的隨機角度 (-π 到 π)
        f_angles = (torch.rand(BATCH_SIZE, 5).to(DEVICE) * 2 * torch.pi) - torch.pi
        p_angles = (torch.rand(BATCH_SIZE, 5).to(DEVICE) * 2 * torch.pi) - torch.pi
        
        # 填入正確的 Cos/Sin 與距離/壓迫感
        m_in[:, 0, :] = torch.cos(f_angles)
        m_in[:, 1, :] = torch.sin(f_angles)
        m_in[:, 2, :] = torch.rand(BATCH_SIZE, 5).to(DEVICE)         # 食物距離 0~1
        
        m_in[:, 3, :] = torch.cos(p_angles)
        m_in[:, 4, :] = torch.sin(p_angles)
        m_in[:, 5, :] = torch.rand(BATCH_SIZE, 5).to(DEVICE) * 10.0   # 壓迫感 0~10
        
        # s_in
        s_in = torch.rand(BATCH_SIZE, 3).to(DEVICE)
        s_in[:, 1] = 0.0
        
        # 取得專家目標動作
        target = expert_logic(m_in, s_in)
        
        # 反向傳播與參數更新
        optimizer.zero_grad()
        output = actor(m_in, s_in)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # 顯示進度
        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{EPOCHS}, Pretrain Loss: {loss.item():.4f}")

    # 存檔
    torch.save({
        'actor': actor.state_dict(),
        'critic': critic.state_dict()
    }, BRAIN_PATH)
    print("--- Pretrain Complete. Weights properly saved! ---")

if __name__ == "__main__":
    run_pretrain()
