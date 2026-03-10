import torch
import torch.nn as nn
import torch.optim as optim

# 必須與原程式參數完全一致
INPUT_SIZE = 10
HIDDEN_SIZE = 16
OUTPUT_SIZE = 2
PRETRAIN_SAVE_PATH = "expert_seed.pt"

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(INPUT_SIZE, HIDDEN_SIZE))
        self.w2 = nn.Parameter(torch.randn(HIDDEN_SIZE, OUTPUT_SIZE))

    def forward(self, x):
        h = torch.tanh(x @ self.w1)
        return torch.tanh(h @ self.w2)

def expert_logic(sensors):
    """
    專家規則：
    sensors: [food_cos, food_sin, food_dist, X, Y, vX, vY, pred_cos, pred_sin, pressure]
    """
    food_sin = sensors[1]
    pred_sin = sensors[8]
    pressure = sensors[9]

    if pressure > 0.4:  # 威脅大於門檻
        # 轉向 = 掠食者正對向 (取反)
        turn = -pred_sin * 1.5 
        thrust = 1.0  # 全速逃跑
    else:
        # 趨向食物
        turn = food_sin * 0.8
        thrust = 0.4  # 穩定巡航
    
    return torch.tensor([thrust, turn])

def run_pretrain(epochs=200000):
    model = Net()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Starting Expert Supervision...")
    for i in range(epochs):
        # 隨機產生模擬感測資料 (Uniform Distribution)
        # 讓 NN 看過各種可能的食物與掠食者角度
        raw_input = torch.rand(INPUT_SIZE) * 2 - 1 
        raw_input[2] = torch.rand(1) # dist
        raw_input[9] = torch.rand(1) * 2 # pressure [0, 2]
        
        target = expert_logic(raw_input)
        
        # Training step
        optimizer.zero_grad()
        output = model(raw_input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if i % 5000 == 0:
            print(f"Step {i}, Loss: {loss.item():.6f}")

    # 儲存權重 (僅儲存單個專家的權重)
    torch.save({'w1': model.w1.data, 'w2': model.w2.data}, PRETRAIN_SAVE_PATH)
    print(f"Expert weights saved to {PRETRAIN_SAVE_PATH}")

if __name__ == "__main__":
    run_pretrain()