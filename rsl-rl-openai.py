import torch
import pygame
import sys
from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner

# =========================
# ⚙️ 參數
# =========================
SCREEN_SIZE = 600
AGENT_SIZE = 8
GOAL_SIZE = 10
OBSTACLE_SIZE = 40
MAX_STEPS = 3000

NUM_ENVS = 128


# =========================
# 🎮 Vectorized Env
# =========================
class SimpleVecEnv:
    def __init__(self, num_envs, device='cuda'):
        self.num_envs = num_envs
        self.device = device

        self.num_obs = 6
        self.num_actions = 2
        self.num_privileged_obs = 0

        self.cfg = {"env": {"num_envs": num_envs}}
        self.max_episode_length = MAX_STEPS

        # state
        self.agent_pos = torch.zeros((num_envs, 2), device=device)
        self.goal_pos = torch.zeros((num_envs, 2), device=device)
        self.obstacles = torch.zeros((num_envs, 8, 2), device=device)

        self.rew_buf = torch.zeros(num_envs, device=device)
        self.reset_buf = torch.ones(num_envs, device=device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.time_out_buf = torch.zeros(num_envs, device=device, dtype=torch.bool)

        # render（只畫 env0）
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
        self.clock = pygame.time.Clock()

    # =========================
    def _get_obs(self):
        dist = torch.cdist(self.agent_pos.unsqueeze(1), self.obstacles).squeeze(1)
        idx = torch.argmin(dist, dim=1)
        closest_obs = self.obstacles[torch.arange(self.num_envs), idx]

        obs = torch.cat([
            self.agent_pos / SCREEN_SIZE,
            self.goal_pos / SCREEN_SIZE,
            (closest_obs - self.agent_pos) / SCREEN_SIZE
        ], dim=-1)

        return TensorDict({
            "actor": obs,
            "critic": obs
        }, batch_size=[self.num_envs], device=self.device)

    # =========================
    def reset(self):
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        if len(env_ids) > 0:
            self.agent_pos[env_ids] = torch.rand((len(env_ids), 2), device=self.device) * SCREEN_SIZE
            self.goal_pos[env_ids] = torch.rand((len(env_ids), 2), device=self.device) * SCREEN_SIZE
            self.obstacles[env_ids] = torch.rand((len(env_ids), 8, 2), device=self.device) * (SCREEN_SIZE - OBSTACLE_SIZE)

            self.episode_length_buf[env_ids] = 0
            self.reset_buf[env_ids] = 0

        return self._get_obs(), None

    # =========================
    def step(self, actions):
        actions = actions.to(self.device).clamp(-1, 1)

        self.episode_length_buf += 1
        self.agent_pos += actions * 6.0

        # 邊界
        out_of_bounds = (self.agent_pos < 0).any(dim=1) | (self.agent_pos > SCREEN_SIZE).any(dim=1)

        # 障礙物
        dist_to_obs = torch.cdist(
            self.agent_pos.unsqueeze(1),
            self.obstacles + OBSTACLE_SIZE / 2
        ).squeeze(1)
        hit_obstacle = (dist_to_obs < OBSTACLE_SIZE / 2 + AGENT_SIZE).any(dim=1)

        # 目標
        dist_to_goal = torch.norm(self.agent_pos - self.goal_pos, dim=1)
        reached_goal = dist_to_goal < (GOAL_SIZE + AGENT_SIZE)

        # =========================
        # reward
        # =========================
        self.rew_buf[:] = -0.01
        self.rew_buf += -dist_to_goal * 0.002

        self.rew_buf[reached_goal] = 5.0
        self.rew_buf[out_of_bounds | hit_obstacle] = -5.0

        done = out_of_bounds | hit_obstacle | reached_goal
        timeout = self.episode_length_buf >= self.max_episode_length

        self.reset_buf[:] = (done | timeout).long()
        self.time_out_buf = timeout

        # ⚠️ 先存再 reset
        obs = self._get_obs()
        rew = self.rew_buf.clone()
        done_out = self.reset_buf.clone()

        self.reset()

        # render（低頻）
        if torch.rand(1).item() < 0.01:
            self.render()

        return obs, rew, done_out, {"time_outs": self.time_out_buf}

    # =========================
    def render(self):
        self.screen.fill((30, 30, 30))

        for o in self.obstacles[0].cpu().numpy():
            pygame.draw.rect(self.screen, (80, 80, 80),
                             (o[0], o[1], OBSTACLE_SIZE, OBSTACLE_SIZE))

        pygame.draw.circle(self.screen, (0, 255, 100),
                           self.goal_pos[0].cpu().numpy().astype(int), GOAL_SIZE)

        pygame.draw.circle(self.screen, (255, 80, 80),
                           self.agent_pos[0].cpu().numpy().astype(int), AGENT_SIZE)

        pygame.display.flip()
        self.clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

    # =========================
    def get_observations(self):
        return self._get_obs()

    def get_privileged_observations(self):
        return None


# =========================
# 🧠 Training
# =========================
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = SimpleVecEnv(NUM_ENVS, device=device)

    train_cfg = {
        "num_steps_per_env": 32,
        "save_interval": 0,

        "algorithm": {
            "class_name": "PPO",
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 3e-4,
            "clip_param": 0.2,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "entropy_coef": 0.01,
            "value_loss_coef": 1.0,
            "max_grad_norm": 1.0,
        },

        # 🔥 正確 stochastic 寫法
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
            },
        },

        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [128, 128],
            "activation": "elu",
        },

        "runner": {
            "num_steps_per_env": 32,
            "max_iterations": 2000,
            "save_interval": 100,
            "experiment_name": "ppo_nav",
            "run_name": "final",
        },

        "obs_groups": {
            "actor": ["actor"],
            "critic": ["critic"]
        },

        "multi_gpu": {
            "enabled": False
        }
    }

    runner = OnPolicyRunner(env, train_cfg, log_dir="logs", device=device)

    runner.learn(
        num_learning_iterations=2000,
        init_at_random_ep_len=True
    )


# =========================
if __name__ == "__main__":
    train()
