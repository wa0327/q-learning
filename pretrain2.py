import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEARBY_FOOD_COUNT = 5
NEARBY_PRED_COUNT = 3
OBJ_FEATURES = 3   
SELF_FEATURES = 3  
PRETRAIN_SAVE_PATH = "pretrain2.pt"

class HybridBrain(nn.Module):
    def __init__(self):
        super(HybridBrain, self).__init__()
        self.food_conv = nn.Conv1d(OBJ_FEATURES, 16, kernel_size=1)
        self.pred_conv = nn.Conv1d(OBJ_FEATURES, 16, kernel_size=1)
        self.fc = nn.Sequential(
            nn.Linear(35, 24),
            nn.ReLU(),
            nn.Linear(24, 2),
            nn.Tanh()
        )

    def forward(self, food_list, pred_list, self_state):
        f = F.relu(self.food_conv(food_list))
        f = torch.max(f, dim=2)[0]
        p = F.relu(self.pred_conv(pred_list))
        p = torch.max(p, dim=2)[0]
        combined = torch.cat([f, p, self_state], dim=1)
        return self.fc(combined)

def expert_logic(food_in, pred_in, self_in):
    batch_size = food_in.shape[0]
    targets = torch.zeros(batch_size, 2).to(DEVICE)
    best_food_sin = food_in[:, 1, 0]
    worst_pred_sin = pred_in[:, 1, 0]
    worst_pred_pressure = pred_in[:, 2, 0]
    energy_ratio = self_in[:, 2]

    risk_tolerance = 0.5 + (1.0 - energy_ratio) * 0.4
    evade_mask = worst_pred_pressure > risk_tolerance
    
    targets[evade_mask, 0] = 1.0 
    targets[evade_mask, 1] = -worst_pred_sin[evade_mask] * 1.5
    targets[~evade_mask, 0] = 0.6
    targets[~evade_mask, 1] = best_food_sin[~evade_mask] * 1.0
    return targets.clamp(-1, 1)

def run_pretrain(epochs=100000):
    model = HybridBrain().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    print(f"[Pretrain] Training expert brain to {PRETRAIN_SAVE_PATH}...")

    for i in range(epochs):
        f_in = torch.randn(128, 3, NEARBY_FOOD_COUNT).to(DEVICE)
        p_in = torch.randn(128, 3, NEARBY_PRED_COUNT).to(DEVICE)
        s_in = torch.rand(128, 3).to(DEVICE)
        
        target = expert_logic(f_in, p_in, s_in)
        optimizer.zero_grad()
        output = model(f_in, p_in, s_in)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 3000 == 0:
            print(f"Progress: {(i+1)/epochs*100:.0f}% | Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), PRETRAIN_SAVE_PATH)
    print("Pretrain Complete.")

if __name__ == "__main__":
    run_pretrain()