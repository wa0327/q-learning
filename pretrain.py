import torch
import torch.nn as nn
import torch.optim as optim

# 必須與原程式參數完全一致
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = 11
HIDDEN_SIZE = 16
OUTPUT_SIZE = 2
PRETRAIN_SAVE_PATH = "pretrain.pt"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(INPUT_SIZE, HIDDEN_SIZE).to(DEVICE))
        self.w2 = nn.Parameter(torch.randn(HIDDEN_SIZE, HIDDEN_SIZE).to(DEVICE))
        self.w3 = nn.Parameter(torch.randn(HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE))

    def forward(self, x):
        h1 = torch.tanh(x @ self.w1)
        h2 = torch.tanh(h1 @ self.w2)
        return torch.tanh(h2 @ self.w3)

def expert_logic(sensors):
    """
    專家規則 (在 GPU 上進行張量運算)
    sensors[9]: pressure, sensors[10]: energy_ratio
    """
    # 這裡傳入的是 batch 化的 sensors [BATCH_SIZE, INPUT_SIZE]
    food_sin = sensors[:, 1]
    pred_sin = sensors[:, 8]
    pressure = sensors[:, 9]
    energy_ratio = sensors[:, 10]

    # 飢餓修正邏輯
    risk_tolerance = 0.4 + (1.0 - energy_ratio) * 0.4
    
    # 建立輸出 Tensor
    batch_size = sensors.shape[0]
    targets = torch.zeros(batch_size, 2).to(DEVICE)

    # 判斷掩碼：壓力大於耐受度
    evade_mask = pressure > risk_tolerance
    
    # 逃跑模式 (Thrust=1.0, Turn=-pred_sin)
    targets[evade_mask, 0] = 1.0
    targets[evade_mask, 1] = -pred_sin[evade_mask] * 1.5
    
    # 覓食模式 (Thrust=0.6, Turn=food_sin)
    targets[~evade_mask, 0] = 0.6
    targets[~evade_mask, 1] = food_sin[~evade_mask] * 1.0
    
    return targets

def run_pretrain(epochs=100000):
    model = Net().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.002) # 稍微提高學習率
    criterion = nn.MSELoss()
    
    # 批次訓練效率更高
    BATCH_SIZE = 256 
    print_interval = max(1, epochs // 100)

    print(f"[Pretrain] Device: {DEVICE.type.upper()} | Epochs: {epochs}")
    print(f"[Pretrain] Training expert logic...")

    for i in range(epochs):
        # 1. 在 GPU 上直接產生隨機感測資料
        raw_input = torch.rand(BATCH_SIZE, INPUT_SIZE).to(DEVICE) * 2 - 1 
        raw_input[:, 2] = torch.rand(BATCH_SIZE).to(DEVICE)     # food_dist
        raw_input[:, 9] = torch.rand(BATCH_SIZE).to(DEVICE) * 2 # pressure
        raw_input[:, 10] = torch.rand(BATCH_SIZE).to(DEVICE)    # energy_ratio
        
        # 2. 取得專家目標值
        target = expert_logic(raw_input)
        
        # 3. 訓練步驟
        optimizer.zero_grad()
        output = model(raw_input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # 每 1% 列印一次進度
        if (i + 1) % print_interval == 0:
            percentage = (i + 1) // print_interval
            print(f"Progress: {percentage:>3}% | Step: {i+1:>6} | Loss: {loss.item():.6f}")

    # 儲存權重 (僅儲存單個專家的權重)
    torch.save({
        'w1': model.w1.data,
        'w2': model.w2.data,
        'w3': model.w3.data
    }, PRETRAIN_SAVE_PATH)
    print(f"Expert weights saved to {PRETRAIN_SAVE_PATH}")

if __name__ == "__main__":
    run_pretrain()