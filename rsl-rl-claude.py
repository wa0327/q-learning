"""
PPO Explorer — rsl-rl-lib v5 + pygame
======================================
Agent 任務：在螢幕範圍內找到獎勵點（金色星星）
- 碰到障礙物或邊界 → terminated  (reward = -1)
- 3000 步內未找到獎勵 → truncated (reward = -0.5, time_outs=True)
- 找到目標 → reward = +10

設計依據：官方 tests/test_on_policy_runner.py 的 DummyEnv / _make_train_cfg 模式

執行方式（需在 isaaclab conda 環境下）：
    conda activate isaaclab
    pip install pygame
    python ppo_explorer.py

鍵盤控制：
    SPACE  — 暫停 / 繼續
    R      — 重置環境
    Q/ESC  — 離開
"""

import math
import random
import time
import threading

import numpy as np
import torch
import pygame
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner

# ═══════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════
SCREEN_W, SCREEN_H = 800, 600
CELL          = 20
MAX_STEPS     = 3000
NUM_OBSTACLES = 15
SPEED         = CELL
FPS_RENDER    = 30

C_BG       = (18, 18, 28)
C_AGENT    = (80, 200, 255)
C_GOAL     = (255, 215, 0)
C_OBSTACLE = (180, 60, 60)
C_TRAIL    = (40, 60, 100)
C_TEXT     = (220, 220, 220)
C_PANEL    = (30, 30, 45)


# ═══════════════════════════════════════════════════════════════════════════
#  VecEnv — 完全比照官方 DummyEnv 模式
# ═══════════════════════════════════════════════════════════════════════════
class ExplorerVecEnv(VecEnv):
    """
    Observation TensorDict:
        "policy" → (num_envs, 8)
            [ax/W, ay/H, gx/W, gy/H, dx/W, dy/H, dist/diag, steps/MAX]

    注意：dones 必須是 float tensor（官方測試：dones = ...float()）
    """

    def __init__(self, num_envs: int = 1, device: str = "cpu"):
        # ── VecEnv 必要屬性（完全比照官方 DummyEnv）──
        self.num_envs            = num_envs
        self.num_actions         = 4
        self.max_episode_length  = MAX_STEPS
        self.episode_length_buf  = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.device              = device
        self.cfg                 = {}

        self._W    = float(SCREEN_W)
        self._H    = float(SCREEN_H)
        self._diag = math.hypot(self._W, self._H)

        self._agent_x   = np.zeros(num_envs, np.float32)
        self._agent_y   = np.zeros(num_envs, np.float32)
        self._goal_x    = np.zeros(num_envs, np.float32)
        self._goal_y    = np.zeros(num_envs, np.float32)
        self._obstacles = [[] for _ in range(num_envs)]
        self._prev_dist = np.zeros(num_envs, np.float32)

        # 統計（供 UI 顯示用）
        self.total_episodes  = 0
        self.total_successes = 0
        self.reward_buf: list[float] = []
        self._ep_reward = np.zeros(num_envs, np.float32)

        self._reset_all()

    # ── helpers ──────────────────────────────────────────────────────────
    def _rand_pos(self, margin=CELL * 2):
        x = random.randint(margin, int(self._W) - margin)
        y = random.randint(margin, int(self._H) - margin)
        return float(x), float(y)

    def _reset_env(self, i: int):
        ax, ay = self._rand_pos()
        gx, gy = self._rand_pos()
        while math.hypot(gx - ax, gy - ay) < 150:
            gx, gy = self._rand_pos()

        obs_list = []
        for _ in range(NUM_OBSTACLES):
            ox, oy = self._rand_pos(CELL * 3)
            while (math.hypot(ox - ax, oy - ay) < CELL * 4 or
                   math.hypot(ox - gx, oy - gy) < CELL * 4):
                ox, oy = self._rand_pos(CELL * 3)
            obs_list.append((ox, oy))

        self._agent_x[i]   = ax
        self._agent_y[i]   = ay
        self._goal_x[i]    = gx
        self._goal_y[i]    = gy
        self._obstacles[i] = obs_list
        self._prev_dist[i] = math.hypot(gx - ax, gy - ay)
        self.episode_length_buf[i] = 0

    def _reset_all(self):
        for i in range(self.num_envs):
            self._reset_env(i)

    def _collision(self, i: int, x: float, y: float) -> bool:
        h = CELL // 2
        if x - h < 0 or x + h > self._W or y - h < 0 or y + h > self._H:
            return True
        for ox, oy in self._obstacles[i]:
            if abs(x - ox) < CELL and abs(y - oy) < CELL:
                return True
        return False

    def _build_obs(self) -> TensorDict:
        obs = np.zeros((self.num_envs, 8), np.float32)
        for i in range(self.num_envs):
            ax, ay = self._agent_x[i], self._agent_y[i]
            gx, gy = self._goal_x[i],  self._goal_y[i]
            dist   = math.hypot(gx - ax, gy - ay)
            steps  = float(self.episode_length_buf[i].item())
            obs[i] = [
                ax / self._W, ay / self._H,
                gx / self._W, gy / self._H,
                (gx - ax) / self._W, (gy - ay) / self._H,
                dist / self._diag,
                steps / MAX_STEPS,
            ]
        t = torch.tensor(obs, device=self.device)
        return TensorDict({"policy": t}, batch_size=[self.num_envs], device=self.device)

    # ── VecEnv interface ──────────────────────────────────────────────────
    def get_observations(self) -> TensorDict:
        return self._build_obs()

    def step(self, actions: torch.Tensor):
        """
        actions: (num_envs, num_actions) — 連續 Gaussian 輸出，argmax → 離散方向
        Returns: (TensorDict, rewards, dones_float, extras)
        """
        if actions.dim() == 2:
            acts = actions.argmax(dim=-1).cpu().numpy().astype(int)
        else:
            acts = actions.cpu().numpy().flatten().astype(int)

        rews      = np.zeros(self.num_envs, np.float32)
        dones     = np.zeros(self.num_envs, np.float32)   # float，與官方一致
        time_outs = torch.zeros(self.num_envs, device=self.device)

        for i in range(self.num_envs):
            ax, ay = self._agent_x[i], self._agent_y[i]
            a = int(acts[i])

            if   a == 0: ay -= SPEED
            elif a == 1: ay += SPEED
            elif a == 2: ax -= SPEED
            elif a == 3: ax += SPEED

            self.episode_length_buf[i] += 1
            gx, gy = self._goal_x[i], self._goal_y[i]
            dist   = math.hypot(gx - ax, gy - ay)

            if self._collision(i, ax, ay):
                rews[i]  = -1.0
                dones[i] = 1.0
            elif dist < CELL:
                rews[i]  = 10.0
                dones[i] = 1.0
                self.total_successes += 1
            elif self.episode_length_buf[i] >= MAX_STEPS:
                rews[i]      = -0.5
                dones[i]     = 1.0
                time_outs[i] = 1.0
            else:
                rews[i]            = (self._prev_dist[i] - dist) / 100.0 - 0.001
                self._agent_x[i]   = ax
                self._agent_y[i]   = ay
                self._prev_dist[i] = dist

            self._ep_reward[i] += rews[i]

            if dones[i]:
                self.total_episodes += 1
                self.reward_buf.append(float(self._ep_reward[i]))
                if len(self.reward_buf) > 100:
                    self.reward_buf.pop(0)
                self._ep_reward[i] = 0.0
                self._reset_env(i)

        obs_td  = self._build_obs()
        rews_t  = torch.tensor(rews,  device=self.device)
        dones_t = torch.tensor(dones, device=self.device)
        extras  = {"time_outs": time_outs}

        return obs_td, rews_t, dones_t, extras


# ═══════════════════════════════════════════════════════════════════════════
#  Config — 完全比照官方測試的 _make_train_cfg (model_type="mlp")
# ═══════════════════════════════════════════════════════════════════════════
def make_train_cfg():
    return {
        "seed": 42,
        "num_steps_per_env": 64,
        "save_interval": 500,

        "algorithm": {
            "class_name":             "PPO",
            "gamma":                  0.99,
            "lam":                    0.95,
            "value_loss_coef":        1.0,
            "use_clipped_value_loss": True,
            "clip_param":             0.2,
            "entropy_coef":           0.01,
            "num_learning_epochs":    4,
            "num_mini_batches":       4,
            "learning_rate":          3e-4,
            "schedule":               "adaptive",
            "desired_kl":             0.01,
            "max_grad_norm":          1.0,
        },

        "actor": {
            "class_name":  "MLPModel",
            "hidden_dims": [128, 128, 64],
            "activation":  "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
            },
        },
        "critic": {
            "class_name":  "MLPModel",
            "hidden_dims": [128, 128, 64],
            "activation":  "elu",
        },

        "obs_groups": {
            "actor":  ["policy"],
            "critic": ["policy"],
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Pygame Renderer
# ═══════════════════════════════════════════════════════════════════════════
class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H + 120), pygame.SCALED)
        pygame.display.set_caption("PPO Explorer — rsl-rl-lib v5")
        self.font  = pygame.font.SysFont("monospace", 14)
        self.big   = pygame.font.SysFont("monospace", 18, bold=True)
        self._clk  = pygame.time.Clock()
        self.trail: list[tuple[int, int]] = []

    def handle_events(self) -> dict:
        ev = {"quit": False, "pause": False, "reset": False}
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                ev["quit"] = True
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_q, pygame.K_ESCAPE): ev["quit"]  = True
                elif e.key == pygame.K_SPACE:               ev["pause"] = True
                elif e.key == pygame.K_r:                   ev["reset"] = True
        return ev

    def draw(self, env: ExplorerVecEnv, stats: dict):
        self.screen.fill(C_BG)
        pygame.draw.rect(self.screen, (60, 60, 90), (0, 0, SCREEN_W, SCREEN_H), 2)

        i  = 0
        ax = int(env._agent_x[i])
        ay = int(env._agent_y[i])
        self.trail.append((ax, ay))
        if len(self.trail) > 200:
            self.trail.pop(0)

        for j, (tx, ty) in enumerate(self.trail):
            alpha = int(255 * j / max(len(self.trail), 1))
            col   = (C_TRAIL[0], C_TRAIL[1], min(255, C_TRAIL[2] + alpha // 2))
            pygame.draw.circle(self.screen, col, (tx, ty), 2)

        for ox, oy in env._obstacles[i]:
            r = pygame.Rect(int(ox) - CELL//2, int(oy) - CELL//2, CELL, CELL)
            pygame.draw.rect(self.screen, C_OBSTACLE, r, border_radius=4)
            pygame.draw.rect(self.screen, (220, 80, 80), r, 1, border_radius=4)

        gx = int(env._goal_x[i])
        gy = int(env._goal_y[i])
        pulse = int(4 + 3 * math.sin(time.time() * 4))
        pygame.draw.circle(self.screen, C_GOAL, (gx, gy), CELL // 2 + pulse)
        pygame.draw.circle(self.screen, (255, 255, 180), (gx, gy), CELL // 2)

        pygame.draw.circle(self.screen, C_AGENT, (ax, ay), CELL // 2)
        pygame.draw.circle(self.screen, (180, 240, 255), (ax, ay), CELL // 2, 2)

        # 底部面板
        pygame.draw.rect(self.screen, C_PANEL, (0, SCREEN_H, SCREEN_W, 120))
        pygame.draw.line(self.screen, (80, 80, 120), (0, SCREEN_H), (SCREEN_W, SCREEN_H), 2)

        def txt(s, x, y, color=C_TEXT, big=False):
            f = self.big if big else self.font
            self.screen.blit(f.render(s, True, color), (x, y))

        it   = stats.get("iteration", 0)
        ep   = env.total_episodes
        rew  = float(np.mean(env.reward_buf)) if env.reward_buf else 0.0
        suc  = env.total_successes / max(ep, 1)
        st   = int(env.episode_length_buf[0].item())
        dist = math.hypot(env._goal_x[i] - env._agent_x[i],
                          env._goal_y[i] - env._agent_y[i])
        paused = stats.get("paused", False)

        txt("PPO Explorer  [rsl-rl-lib v5]", 10, SCREEN_H + 8, (180, 200, 255), big=True)
        txt(f"Iter: {it:>5}   Episodes: {ep:>6}   Step: {st:>5}/{MAX_STEPS}", 10, SCREEN_H + 32)
        txt(f"Mean Reward: {rew:>7.3f}   Success: {suc:.1%}   Dist: {dist:>6.1f}", 10, SCREEN_H + 52)
        txt(f"SPACE=Pause  R=Reset  Q=Quit{'  [PAUSED]' if paused else ''}",
            10, SCREEN_H + 78, (150, 150, 180))

        bar_w = int((1.0 - min(dist, 600) / 600) * (SCREEN_W - 20))
        pygame.draw.rect(self.screen, (50, 80, 50),
                         (10, SCREEN_H + 100, SCREEN_W - 20, 10), border_radius=5)
        pygame.draw.rect(self.screen, (80, 200, 100),
                         (10, SCREEN_H + 100, max(bar_w, 0), 10), border_radius=5)

        pygame.display.flip()
        self._clk.tick(FPS_RENDER)

    def clear_trail(self):
        self.trail.clear()

    def quit(self):
        pygame.quit()


# ═══════════════════════════════════════════════════════════════════════════
#  Training thread（讓 pygame 在 main thread 執行）
# ═══════════════════════════════════════════════════════════════════════════
class TrainState:
    """Main thread / training thread 的共享狀態。"""
    def __init__(self):
        self.iteration  = 0
        self.paused     = False
        self.stop       = False
        self.reset_req  = False
        self.lock       = threading.Lock()


def train_loop(runner: OnPolicyRunner, env: ExplorerVecEnv,
               state: TrainState, max_iter: int):
    """在背景 thread 執行 PPO 訓練。"""
    alg       = runner.alg
    steps_per = runner.cfg["num_steps_per_env"]
    gamma     = runner.cfg["algorithm"]["gamma"]
    lam       = runner.cfg["algorithm"]["lam"]

    alg.train_mode()
    obs_td = env.get_observations()

    for it in range(max_iter):
        # 停止 / 暫停 / 重置 檢查
        while True:
            with state.lock:
                if state.stop:
                    return
                if state.reset_req:
                    env._reset_all()
                    obs_td = env.get_observations()
                    state.reset_req = False
                if not state.paused:
                    break
            time.sleep(0.05)

        # Rollout
        for _ in range(steps_per):
            with torch.no_grad():
                actions = alg.act(obs_td)
            next_obs_td, rewards, dones, extras = env.step(actions)
            alg.process_env_step(obs=obs_td, rewards=rewards, dones=dones, extras=extras)
            obs_td = next_obs_td

        # Update
        with torch.no_grad():
            alg.compute_returns(obs_td)
        train_results = alg.update()# 從 dict 中提取 loss (Key 名稱取決於 PPO 類別定義)
        mean_val_loss = train_results.get("value_loss", 0.0)
        mean_surr_loss = train_results.get("surrogate_loss", 0.0)

        with state.lock:
            state.iteration = it + 1

        if (it + 1) % 50 == 0:
            ep  = env.total_episodes
            rew = float(np.mean(env.reward_buf)) if env.reward_buf else 0.0
            suc = env.total_successes / max(ep, 1)
            print(f"[Iter {it+1:>5}]  ep={ep:>5}  rew={rew:>7.3f}  "
                  f"success={suc:.1%}  val={mean_val_loss:.4f}  surr={mean_surr_loss:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    device    = "cuda"
    env       = ExplorerVecEnv(num_envs=1, device=device)
    renderer  = Renderer()
    train_cfg = make_train_cfg()
    log_dir   = "logs/ppo_explorer"

    print("=" * 60)
    print("  PPO Explorer — rsl-rl-lib v5")
    print("  建立 OnPolicyRunner…")
    print("=" * 60)

    runner = OnPolicyRunner(env, train_cfg, log_dir=log_dir, device=device)

    print("Runner 建立完成，開始訓練！")
    print("  SPACE=暫停  R=重置  Q/ESC=離開")

    state    = TrainState()
    max_iter = 2000

    t = threading.Thread(
        target=train_loop,
        args=(runner, env, state, max_iter),
        daemon=True,
    )
    t.start()

    # ── pygame 主迴圈（main thread）──────────────────────────────────
    stats: dict = {}
    while t.is_alive():
        ev = renderer.handle_events()
        if ev["quit"]:
            with state.lock:
                state.stop = True
            break
        if ev["pause"]:
            with state.lock:
                state.paused = not state.paused
        if ev["reset"]:
            with state.lock:
                state.reset_req = True
            renderer.clear_trail()

        with state.lock:
            stats["iteration"] = state.iteration
            stats["paused"]    = state.paused

        renderer.draw(env, stats)

    t.join(timeout=2.0)
    print("\n訓練結束，關閉視窗。")
    renderer.quit()


if __name__ == "__main__":
    main()