import torch
import pygame
import sys
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

# --- 環境參數 ---
SCREEN_SIZE, AGENT_SIZE, GOAL_SIZE, OBSTACLE_SIZE = 600, 10, 15, 40
MAX_STEPS, NUM_ENVS = 3000, 64

class SimpleVecEnv:
    def __init__(self, num_envs, device='cuda'):
        self.num_envs, self.device = num_envs, device
        self.num_actions = 2
        self.max_episode_length = MAX_STEPS
        self.cfg = {} # 官方測試中環境要有 cfg 屬性
        self.agent_pos = torch.zeros((num_envs, 2), device=device)
        self.goal_pos = torch.zeros((num_envs, 2), device=device)
        self.rew_buf = torch.zeros(num_envs, device=device)
        self.reset_buf = torch.ones(num_envs, device=device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.obstacles = torch.rand((8, 2), device=device) * (SCREEN_SIZE - OBSTACLE_SIZE)
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))

    def get_observations(self) -> TensorDict:
        # 官方測試中，Key 必須與 obs_groups 對應，這裡用 "policy"
        raw_obs = torch.cat([self.agent_pos / SCREEN_SIZE, self.goal_pos / SCREEN_SIZE], dim=-1)
        return TensorDict({"policy": raw_obs}, batch_size=[self.num_envs], device=self.device)

    def reset(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.agent_pos[env_ids] = torch.rand((len(env_ids), 2), device=self.device) * SCREEN_SIZE
            self.goal_pos[env_ids] = torch.rand((len(env_ids), 2), device=self.device) * SCREEN_SIZE
            self.episode_length_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0
        return self.get_observations(), None

    def step(self, actions):
        self.episode_length_buf += 1
        self.agent_pos += actions.to(self.device).clamp(-1, 1) * 7.0
        
        dist = torch.norm(self.agent_pos - self.goal_pos, dim=1)
        out = (self.agent_pos < 0).any(1) | (self.agent_pos > SCREEN_SIZE).any(1)
        hit = (torch.cdist(self.agent_pos, self.obstacles + OBSTACLE_SIZE/2) < OBSTACLE_SIZE/2 + AGENT_SIZE).any(dim=1)
        win = dist < (GOAL_SIZE + AGENT_SIZE)
        
        self.rew_buf[:] = -0.01 + 0.1 * (1.0 - dist / SCREEN_SIZE)
        self.rew_buf[win], self.rew_buf[hit | out] = 20.0, -10.0
        self.reset_buf[:] = (win | hit | out | (self.episode_length_buf >= self.max_episode_length)).long()
        
        self.reset()
        self.render()
        # 官方要求回傳: obs, rewards, dones, extras
        return self.get_observations(), self.rew_buf, self.reset_buf, {"time_outs": self.reset_buf}

    def render(self):
        self.screen.fill((30, 30, 30))
        for o in self.obstacles: pygame.draw.rect(self.screen, (80, 80, 80), (o[0], o[1], OBSTACLE_SIZE, OBSTACLE_SIZE))
        pygame.draw.circle(self.screen, (0, 255, 100), self.goal_pos[0].cpu().numpy().astype(int), GOAL_SIZE)
        pygame.draw.circle(self.screen, (255, 50, 50), self.agent_pos[0].cpu().numpy().astype(int), AGENT_SIZE)
        pygame.display.flip()
        for e in pygame.event.get(): 
            if e.type == pygame.QUIT: pygame.quit(); sys.exit()

# --- 訓練主程式 ---
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = SimpleVecEnv(num_envs=NUM_ENVS, device=device)
    
    # 完全比照你提供的官方測試案例配置
    train_cfg = {
        "num_steps_per_env": 32,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 5e-4,
            "gamma": 0.99,
            "lam": 0.95,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
            "distribution_cfg": { # 關鍵：這是 5.x 版正確啟用隨機分佈的方式
                "class_name": "GaussianDistribution",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
        },
    }

    # 初始化 Runner
    runner = OnPolicyRunner(env, train_cfg, log_dir='logs', device=device)
    
    print("Training Started with Official Test Standard...")
    runner.learn(num_learning_iterations=2000, init_at_random_ep_len=True)

if __name__ == "__main__":
    train()