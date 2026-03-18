import pygame
import moderngl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import os
import subprocess
from pathlib import Path
from datetime import datetime
import math
import shutil
from pathlib import Path
import threading
import queue
import argparse
import shlex

script_name = Path(__file__).stem
CAPTION = "Vectra: Apex Protocol"
# 參數設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = f"weights/{script_name}"
SAVE_PATH = f"{BASE_PATH}/{script_name}.pt"

# 環境參數
SCREEN_W, SCREEN_H = 1280, 720 # 邏輯尺寸 (AI 看到的尺寸)
SCALE = 1.33 # 顯示倍率 (你的 133% 縮放)
WINDOW_W, WINDOW_H = int(SCREEN_W * SCALE), int(SCREEN_H * SCALE)
EST_STEPS = 300 # 評估模型在此步數內應獲得最大獎勵
# 這個 0.715 代表生存者需要在限制步數內，有能力跑完畫面 71.5% 的對角線距離
POP_MAX_SPEED = math.sqrt(SCREEN_W**2 + SCREEN_H**2) * 0.715 / EST_STEPS # 在 1280x720 時約為 3.5，僅適合無任何障礙物環境。
POP_SIZE = 16
POP_RADIUS = 4                      # 生存者體積半徑
POP_DAMPING_FACTOR = 0.85           # 阻力系數，越高越需要維持高油門
POP_BACKWARD_FACTOR = 0.33
POP_MAX_STEER = math.radians(15)    # 最大轉向角度
POP_PERCEPTION_RADIUS = 200         # 視野感知半徑
FOOD_SIZE = POP_SIZE * 2
FOOD_RADIUS = 3         # 食物觸碰半徑
MAX_ENERGY = 100.0      # 能量最大總值
FOOD_ENERGY = 50.0      # 食物補充能量
ENERGY_DECAY = FOOD_ENERGY / EST_STEPS # 每步消耗能量
PREDATOR_SIZE = 8
PREDATOR_RADIUS = 20.0  # 掠食者觸碰半徑
PREDATOR_MIN_SPEED = 1.5
PREDATOR_MAX_SPEED = 2.5
POP_ALERT_RADIUS = max(POP_MAX_SPEED, (PREDATOR_MAX_SPEED / POP_MAX_SPEED) ** 2 * 20.58) # 危險警戒半徑，按速度比例呈線性增減
RND_POS_PADDING = 50.0  # 隨機取位邊距
WALL_SIZE = 5

# 獎懲設定
FOOD_REWARD = 15.0    # 吃到食物
KILLED_REWARD = -25   # 被殺
COLLIDED_REWARD = -20 # 撞死
STARVED_REWARD = -25  # 餓死
MOVE_REWARD_FACTOR = 0.25 # [移動總獎勵]與[最大獎勵]的佔比，數值越低對模型越驅策
MOVE_REWARD = FOOD_REWARD * MOVE_REWARD_FACTOR / EST_STEPS # 移動基礎獎勵
TIME_PENALTY_FACTOR = 0.5 # [餓死前的總懲罰]與[餓死懲罰]的佔比，數值越低對模型來說越划算
STEP_REWARD = STARVED_REWARD * TIME_PENALTY_FACTOR / EST_STEPS # 每步時間獎懲
WALL_NEARBY_REWARD = COLLIDED_REWARD * 0.25     # 近牆懲罰, 提早預警危險
PREDATOR_NEARBY_REWARD = KILLED_REWARD * 0.25   # 近敵懲罰, 提早預警危險

# 模型核心參數
GAMMA = 0.99
TAU = 0.005     # 軟更新係數
LR_ACTOR = 0.0003
LR_CRITIC = 0.0003
MEMORY_SIZE = 100000
REPLAY_BATCH_SIZE = 256
FEAT_IN_DIM = 7         # 每個物件特微 [cos, sin, dist, is_wall, is_food, is_team, is_pred]
STATE_IN_DIM = 3        # 自身狀態 [速度, 轉向, 能量]
ACTOR_OUT_DIM = 2       # Actor 輸出層數，輸出動作 [轉向, 油門]
HIDDEN_FEAT_DIM = 64    # 特徵提取層 (Conv1d) 的輸出維度(環境特徵)
HIDDEN_ATTN_DIM = 32    # Attention 內部的隱藏層維度
HIDDEN_FC_DIM = 256     # 後段全連接層 (MLP) 的主要維度(決策輸出)
CRITIC_OUT_DIM = 64     # Critic 輸出前的降維層
TARGET_ENTROPY = -ACTOR_OUT_DIM
INIT_ALPHA = 1.0
MIN_ALPHA = 0.0001
MAX_OBJ = 100    # 最大環境物件數量，實際運行不會超過 POP_SIZE-1 + FOOD_SIZE + PREDATOR_SIZE + 四面牆(最多)

# --- SAC 網路架構 ---
class Actor(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. 特徵提取與廣義感知 (處理周遭物件)
        self.conv = nn.Sequential(
            nn.Conv1d(FEAT_IN_DIM, HIDDEN_FEAT_DIM, 1),
            nn.ReLU(),
            nn.Conv1d(HIDDEN_FEAT_DIM, HIDDEN_FEAT_DIM, 1)
        )

        # 2. Attention 機制 (決定關注哪個物件)
        self.attn_weights = nn.Sequential(
            nn.Linear(HIDDEN_FEAT_DIM, HIDDEN_ATTN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_ATTN_DIM, 1)
        )
        
        # 3. 決策層 (結合感官與自身狀態)
        self.fc_common = nn.Sequential(
            nn.Linear(HIDDEN_FEAT_DIM + STATE_IN_DIM, HIDDEN_FC_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_FC_DIM, HIDDEN_FC_DIM),
            nn.ReLU()
        )
        
        # 4. 輸出層 (SAC 需要 Mean 與 Log_Std)
        self.mu = nn.Linear(HIDDEN_FC_DIM, ACTOR_OUT_DIM)
        self.log_std = nn.Linear(HIDDEN_FC_DIM, ACTOR_OUT_DIM)
        
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, m_in, s_in, deterministic=False, with_logprob=True):
        # 提取環境特徵 [Batch, HIDDEN_FEAT_DIM, MAX_OBJ] -> [Batch, MAX_OBJ, HIDDEN_FEAT_DIM]
        feat = F.relu(self.conv(m_in)).transpose(1, 2)

        # 計算權重並加權平均
        weights = F.softmax(self.attn_weights(feat), dim=1)
        x_attn = torch.sum(feat * weights, dim=1)
        
        # 結合自身狀態 [Batch, HIDDEN_FEAT_DIM + STATE_IN_DIM]
        combined = torch.cat([x_attn, s_in], dim=1)
        x = self.fc_common(combined)
        
        # 計算分佈參數
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        
        # 採樣動作 (Reparameterization Trick)
        if not deterministic or with_logprob:
            dist = Normal(mu, std)
        if deterministic:
            z = mu
        else:
            z = dist.rsample()
            
        # 將動作映射到 [-1, 1]
        action = torch.tanh(z)
        
        log_prob = None
        if with_logprob:
            # 修正 Tanh 造成的機率分佈偏離
            log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            
        return action, log_prob

class Critic(nn.Module):
    def __init__(self):
        super().__init__()

        # 共享的感官層 (Backbone)
        self.conv = nn.Sequential(
            nn.Conv1d(FEAT_IN_DIM, HIDDEN_FEAT_DIM, 1),
            nn.ReLU(),
            nn.Conv1d(HIDDEN_FEAT_DIM, HIDDEN_FEAT_DIM, 1),
            nn.ReLU()
        )
        self.attn = nn.Sequential(
            nn.Linear(HIDDEN_FEAT_DIM, HIDDEN_ATTN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_ATTN_DIM, 1)
        )

        # 獨立的 Q 評價層 (Heads)
        def build_q_head():
            return nn.Sequential(
                nn.Linear(HIDDEN_FEAT_DIM + STATE_IN_DIM + ACTOR_OUT_DIM, HIDDEN_FC_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_FC_DIM, CRITIC_OUT_DIM),
                nn.ReLU(),
                nn.Linear(CRITIC_OUT_DIM, 1)
            )
        
        self.q1_head = build_q_head()
        self.q2_head = build_q_head()

    def forward(self, m_in, s_in, action):
        # 1. 統一提取特徵 (只算一次)
        feat = self.conv(m_in).transpose(1, 2)
        weights = F.softmax(self.attn(feat), dim=1)
        x_attn = torch.sum(feat * weights, dim=1)
        
        # 2. 拼裝輸入
        combined = torch.cat([x_attn, s_in, action], dim=1)

        # 3. 分別進入獨立的評估分支
        q1 = self.q1_head(combined)
        q2 = self.q2_head(combined)
        return q1, q2

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        
        self.m_states = torch.zeros((capacity, FEAT_IN_DIM, MAX_OBJ), device=DEVICE)
        self.s_states = torch.zeros((capacity, STATE_IN_DIM), device=DEVICE)
        self.next_m_states = torch.zeros((capacity, FEAT_IN_DIM, MAX_OBJ), device=DEVICE)
        self.next_s_states = torch.zeros((capacity, STATE_IN_DIM), device=DEVICE)
        self.actions = torch.zeros((capacity, ACTOR_OUT_DIM), device=DEVICE)
        self.rewards = torch.zeros(capacity, device=DEVICE)
        self.dones = torch.zeros(capacity, device=DEVICE)
        
        self.idx = 0
        self.size = 0

    def push(self, m_s, s_s, action, reward, n_m_s, n_s_s, done):
        self.m_states[self.idx].copy_(m_s.detach())
        self.s_states[self.idx].copy_(s_s.detach())
        self.next_m_states[self.idx].copy_(n_m_s.detach())
        self.next_s_states[self.idx].copy_(n_s_s.detach())
        self.actions[self.idx].copy_(action.detach())
        self.rewards[self.idx] = float(reward)
        self.dones[self.idx] = float(done)
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=DEVICE)
        return (
            self.m_states[indices],
            self.s_states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_m_states[indices],
            self.next_s_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size

class GLRenderer:
    def __init__(self, ctx, fbo_texture, w, h):
        self.ctx = ctx
        self.fbo_texture = fbo_texture
        self.w, self.h = w, h
        
        # --- 圓形 Shader (Instanced) ---
        self.circle_prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert; in vec4 in_pos_rad; in vec3 in_color;
                out vec3 v_color;
                out vec2 v_dist; // 傳遞距離資訊給 Fragment Shader
                uniform vec2 res;
                void main() {
                    vec2 p = (in_pos_rad.xy / res) * 2.0 - 1.0;
                    p.y *= -1.0;
                    vec2 scale = (in_pos_rad.z / res) * 2.0;
                    gl_Position = vec4(p + in_vert * scale, 0.0, 1.0);
                    v_color = in_color;
                    v_dist = in_vert; // 這會是圓心出發的向量 (-1 到 1)
                }
            """,
            fragment_shader="""
                #version 330
                in vec3 v_color;
                in vec2 v_dist;
                out vec4 f;
                uniform bool is_hollow;
                uniform float thickness;
                void main() {
                    float d = length(v_dist);
                    if (d > 1.0)
                        discard; // 超過圓半徑的不畫
                    
                    if (is_hollow) {
                        // 如果是空心模式，只畫邊緣 0.9 到 1.0 的部分
                        if (d < (1.0 - thickness))
                        discard; 
                    }
                    f = vec4(v_color, 1.0);
                }
            """
        )
        self.circle_prog['res'].value = (w, h)
        
        # 圓形模板 (32節點)
        ang = np.linspace(0, 2*np.pi, 32, endpoint=False)
        verts = np.stack([np.cos(ang), np.sin(ang)], axis=1).astype('f4')
        self.vbo_circle_temp = ctx.buffer(verts)
        self.vbo_circle_inst = ctx.buffer(reserve=10000 * 7 * 4) # 預留空間
        self.vao_circle = ctx.vertex_array(self.circle_prog, [
            (self.vbo_circle_temp, '2f', 'in_vert'),
            (self.vbo_circle_inst, '4f 3f /i', 'in_pos_rad', 'in_color')
        ])

        # --- 線段 Shader ---
        self.line_prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert; in vec3 in_color;
                out vec3 v_color;
                uniform vec2 res;
                void main() {
                    vec2 pixel_pos = in_vert + vec2(0.5, 0.5);
                    vec2 p = (pixel_pos / res) * 2.0 - 1.0;
                    p.y *= -1.0;
                    gl_Position = vec4(p, 0.0, 1.0);
                    v_color = in_color;
                }
            """,
            fragment_shader="#version 330\nin vec3 v_color; out vec4 f; void main() { f = vec4(v_color, 1.0); }"
        )
        self.line_prog['res'].value = (w, h)
        self.vbo_line = ctx.buffer(reserve=20000 * 5 * 4)
        self.vao_line = ctx.vertex_array(self.line_prog, [(self.vbo_line, '2f 3f', 'in_vert', 'in_color')])

        # --- Text Overlay ---
        self.text_texture = ctx.texture((w, h), 4)
        self.text_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.text_prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert; 
                in vec2 in_texcoord;
                out vec2 v_texcoord;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    v_texcoord = in_texcoord;
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D tex;
                in vec2 v_texcoord;
                out vec4 f_color;
                void main() {
                    f_color = texture(tex, v_texcoord);
                }
            """
        )
        quad = np.array([
            -1, 1, 0, 0,  -1, -1, 0, 1,   1, 1, 1, 0,
            -1, -1, 0, 1,  1, -1, 1, 1,   1, 1, 1, 0,
        ], dtype='f4')
        self.vbo_text = ctx.buffer(quad)
        self.vao_text = ctx.vertex_array(
            self.text_prog,
            [(self.vbo_text, '2f 2f', 'in_vert', 'in_texcoord')]
        )

        # --- Screen blit shader ---
        self.screen_prog = ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_vert;
                out vec2 uv;
                void main() {
                    uv = (in_vert + 1.0) * 0.5;
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                }
            """,
            fragment_shader="""
                #version 330
                uniform sampler2D tex;
                in vec2 uv;
                out vec4 fragColor;
                void main() {
                    fragColor = texture(tex, uv);
                }
            """,
        )
        quad = np.array([
            -1, -1,
            1, -1,
            -1,  1,
            1,  1,
        ], dtype='f4')
        self.vbo_screen = ctx.buffer(quad)
        self.vao_screen = ctx.simple_vertex_array(
            self.screen_prog,
            self.vbo_screen,
            'in_vert'
        )

    def draw_circles(self, pos, rad, color):
        """ pos:[N,2], rad:[N,1], color:[N,3] (0~1) """
        if pos.shape[0] == 0:
            return
        data = torch.cat([pos, rad, rad, color], dim=1).cpu().numpy().astype('f4')
        self.vbo_circle_inst.write(data)
        self.vao_circle.render(moderngl.TRIANGLE_FAN, instances=pos.shape[0])

    def draw_lines(self, start, end, color):
        """ start:[N,2], end:[N,2], color:[N,3] """
        if start.shape[0] == 0:
            return
        # 將線段轉為頂點流 [P1, Color1, P2, Color2, ...]
        N = start.shape[0]
        data = torch.empty((N, 2, 5), device=start.device)
        data[:, 0, :2] = start
        data[:, 0, 2:] = color
        data[:, 1, :2] = end
        data[:, 1, 2:] = color
        self.vbo_line.write(data.cpu().numpy().astype('f4'))
        self.vao_line.render(moderngl.LINES, vertices=N*2)

    def draw_text(self, surface):
        """ surface: pygame.Surface """

        # 更新 texture
        rgba_data = pygame.image.tostring(surface, 'RGBA')
        self.text_texture.write(rgba_data)

        # 開啟 alpha blending（文字透明必須）
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # 畫 fullscreen quad
        self.text_texture.use(0)
        self.vao_text.render()
 
    def blit_to_screen(self, win_w, win_h):
        """把 FBO texture 縮放畫到螢幕"""

        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, win_w, win_h)

        # 關閉深度測試（避免干擾）
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

        # 不需要 blending（純覆蓋）
        self.ctx.disable(moderngl.BLEND)

        self.fbo_texture.use(0)
        self.vao_screen.render(moderngl.TRIANGLE_STRIP)

# --- 模擬環境 ---
class RLSimulation:
    def __init__(self, args):
        pygame.init()
        if not args.headless:
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
            self.ctx = moderngl.create_context()
            fbo_texture = self.ctx.texture((SCREEN_W, SCREEN_H), 3)
            fbo_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
            self.fbo = self.ctx.framebuffer(color_attachments=[fbo_texture])
            self.renderer = GLRenderer(self.ctx, fbo_texture, SCREEN_W, SCREEN_H)
            self.ui_surface = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            self.pbo = self.ctx.buffer(reserve=SCREEN_W * SCREEN_H * 3)
            self.font = pygame.font.SysFont("Consolas", 14)
            self.big_font = pygame.font.SysFont("Consolas", 18, bold=True)
        self.clock = pygame.time.Clock()
        self.fps = 300
        self.update_caption()
        
        # 初始化神經網路
        self.init_network(Actor, Critic)
        self.brain_path = f"{BASE_PATH}/{script_name}_{self.actor.__class__.__name__}.pt"
        self.throttle_factor = POP_MAX_SPEED * POP_DAMPING_FACTOR
        self.reset_env()
        path = Path(BASE_PATH)
        path.mkdir(parents=True, exist_ok=True)
        self.load_state()

        # 四類環境物件的 one-hot 編碼
        self.l_wall = torch.tensor([1, 0, 0, 0], device=DEVICE).view(1, 4, 1)
        self.l_food = torch.tensor([0, 1, 0, 0], device=DEVICE).view(1, 4, 1)
        self.l_team = torch.tensor([0, 0, 1, 0], device=DEVICE).view(1, 4, 1)
        self.l_pred = torch.tensor([0, 0, 0, 1], device=DEVICE).view(1, 4, 1)

        # 定義牆面
        wall_A, wall_B = [], []
        self.add_wall_group(wall_A, wall_B, [
            [0, 0], [SCREEN_W-1, 0], [SCREEN_W-1, SCREEN_H-1], [0, SCREEN_H-1]
        ], True)
        if WALL_SIZE > 0:
            self.add_wall_group(wall_A, wall_B, [
                [SCREEN_W/4, SCREEN_H/4], [SCREEN_W*3/4, SCREEN_H/4]
            ], False)
        if WALL_SIZE > 1:
            self.add_wall_group(wall_A, wall_B, [
                [SCREEN_W*3/4, SCREEN_H*3/4], [SCREEN_W/4, SCREEN_H*3/4]
            ], False)
        if WALL_SIZE > 2:
            self.add_wall_group(wall_A, wall_B, [
                [SCREEN_W/2, SCREEN_H*3/8], [SCREEN_W/2, SCREEN_H*5/8]
            ], False)
        if WALL_SIZE > 3:
            self.add_wall_group(wall_A, wall_B, [
                [SCREEN_W*1/8, SCREEN_H*1/8], [SCREEN_W*1/8, SCREEN_H*7/8]
            ], False)
        if WALL_SIZE > 4:
            self.add_wall_group(wall_A, wall_B, [
                [SCREEN_W*7/8, SCREEN_H*1/8], [SCREEN_W*7/8, SCREEN_H*7/8]
            ], False)
        self.wall_A = torch.cat(wall_A, dim=0)
        self.wall_B = torch.cat(wall_B, dim=0)
        self.wall_v = torch.stack([self.wall_A, self.wall_B], dim=1)

    def add_wall_group(self, wall_A, wall_B, points, closed):
        points = torch.tensor(points, dtype=torch.float32, device=DEVICE)
        
        if closed:
            # 閉合：N 個點產生 N 條線 (1->2, 2->3, 3->4, 4->1)
            wall_A.append(points)
            wall_B.append(torch.roll(points, -1, 0))
        else:
            # 開放：N 個點產生 N-1 條線 (1->2, 2->3, 3->4)
            wall_A.append(points[:-1])
            wall_B.append(points[1:])

    def update_caption(self):
        pygame.display.set_caption(f"{CAPTION} | FPS:{self.fps}")

    def reset_env(self):
        self.frames = 0
        self.rewards_avg = 0.0
        self.killed = 0
        self.collided = 0
        self.starved = 0
        self.eaten = 0
        self.last_actions = torch.zeros((POP_SIZE, ACTOR_OUT_DIM), device=DEVICE)
        self.vel = torch.zeros((POP_SIZE, 2), device=DEVICE)
        self.angle = torch.rand(POP_SIZE, device=DEVICE) * (2 * np.pi)
        self.forward_speed = torch.zeros(POP_SIZE, device=DEVICE)
        self.energy = torch.full((POP_SIZE,), MAX_ENERGY, device=DEVICE, dtype=torch.float)
        self.alive = torch.ones(POP_SIZE, dtype=torch.bool, device=DEVICE)
        self.respawn_timer = torch.zeros(POP_SIZE, dtype=torch.long, device=DEVICE) 
        self.screen_size = torch.tensor([SCREEN_W, SCREEN_H], device=DEVICE, dtype=torch.float)
        self.bounds = self.screen_size - 1.0
        self.pos = torch.rand(POP_SIZE, 2, device=DEVICE) * self.screen_size
        self.food_pos = torch.rand(FOOD_SIZE, 2, device=DEVICE) * self.screen_size
        self.food_vel = (torch.rand(FOOD_SIZE, 2, device=DEVICE) - 0.5) * 3.5
        self.pred_pos = torch.rand(PREDATOR_SIZE, 2, device=DEVICE) * self.screen_size
        self.pred_vel = (torch.rand(PREDATOR_SIZE, 2, device=DEVICE) - 0.5) * 3.5
    
    def init_network(self, actor=None, critic=None):
        if actor is None:
            actor = self.actor.__class__
        if critic is None:
            critic = self.critic.__class__

        self.steps = 0
        
        self.actor = actor().to(DEVICE)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = critic().to(DEVICE)
        self.critic_target = critic().to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=LR_CRITIC)
        
        # 熵係數 (Temperature Alpha)
        self.target_entropy = -ACTOR_OUT_DIM
        self.log_alpha = torch.tensor([math.log(INIT_ALPHA)], requires_grad=True, device=DEVICE)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=LR_ACTOR)

        self.memory = ReplayMemory(MEMORY_SIZE)

        self.last_info = {
            "alpha": 0.0,
            "q_val": 0.0,
            "entropy": 0.0,
            "a_loss": 0.0,
            "c_loss": 0.0
        }

        print(f"[{self.actor.__class__.__name__}] Network & Optimizer init complete.")

    def optimize_model(self):
        if len(self.memory) < REPLAY_BATCH_SIZE:
            return False
        
        m_b, s_b, a_b, r_b, nm_b, ns_b, d_b = self.memory.sample(REPLAY_BATCH_SIZE)
        r_b = r_b.unsqueeze(1) 
        d_b = d_b.unsqueeze(1)

        # 1. 取得當前 alpha 數值用於 Actor/Critic (不帶梯度)
        alpha = self.log_alpha.exp().detach()
        alpha = torch.clamp(alpha, min=MIN_ALPHA)

        # --- 1. 更新 Critic ---
        with torch.no_grad():
            # 使用 Target Critic 計算，避免過度樂觀估計
            next_actions, next_log_probs = self.actor(nm_b, ns_b)
            q1_t, q2_t = self.critic_target(nm_b, ns_b, next_actions)
            
            # SAC 核心公式：Q - alpha * log_prob
            min_q_target = torch.min(q1_t, q2_t) - alpha * next_log_probs
            target_q = r_b + (GAMMA * (1.0 - d_b) * min_q_target)

        q1_curr, q2_curr = self.critic(m_b, s_b, a_b)
        # 計算 MSE 並結合兩個 Q 網路的損失
        critic_loss = F.mse_loss(q1_curr, target_q) + F.mse_loss(q2_curr, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # --- 2. 更新 Actor ---
        # 凍結 Critic 的參數以節省計算資源 (選擇性，但較嚴謹)
        for param in self.critic.parameters():
            param.requires_grad = False

        new_actions, log_probs = self.actor(m_b, s_b)
        q1_new, q2_new = self.critic(m_b, s_b, new_actions)
        min_q_new = torch.min(q1_new, q2_new)

        # 目標是最小化 (alpha * log_prob - Q)，等同於最大化 (Q - alpha * log_prob)
        actor_loss = (alpha * log_probs - min_q_new).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 解凍 Critic
        for param in self.critic.parameters():
            param.requires_grad = True

        # --- 3. 更新 Alpha (自動調整熵) ---
        alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # --- 4. 軟更新 Target 網路 ---
        # SAC 通常只更新 Critic Target
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(tp.data * (1.0 - TAU) + p.data * TAU)

        self.last_info = {
            "alpha": alpha.item(),
            "entropy": -log_probs.mean().item(),
            "q_val": torch.min(q1_curr, q2_curr).mean().item(),
            "c_loss": critic_loss.item(),
            "a_loss": actor_loss.item()
        }
        self.steps += 1
        return True

    def get_states(self):
        angel = self.angle.view(POP_SIZE, 1)
        mixed_in = torch.zeros((POP_SIZE, FEAT_IN_DIM, MAX_OBJ), device=DEVICE, dtype=torch.float32)
        current_idx = 0

        def fill_buffer(features):
            nonlocal current_idx
            num_new = features.shape[2]
            space_left = MAX_OBJ - current_idx
            if num_new > 0 and space_left > 0:
                actual_add = min(num_new, space_left)
                mixed_in[:, :, current_idx : current_idx + actual_add] = features[:, :, :actual_add]
                current_idx += actual_add

        def process_obj(obj_pos, obj_radius, one_hot_label, is_self=False):
            nonlocal current_idx
            if obj_pos.shape[0] == 0:
                return None
            
            # 計算相對位移 (POP_SIZE, OBJ_SIZE, 2)
            diff = obj_pos.unsqueeze(0) - self.pos.unsqueeze(1)
            dist = torch.norm(diff, dim=2)
            if is_self: # 處理隊友時排除自己
                dist.fill_diagonal_(1e6)
                
            # 計算相對角度
            abs_ang = torch.atan2(diff[..., 1], diff[..., 0])
            rel_ang = abs_ang - angel
            
            # 計算距離
            dist_norm = ((dist - POP_RADIUS - obj_radius) / POP_PERCEPTION_RADIUS).clamp(0, 1)
            # 排除視野遮罩
            mask = ((dist - POP_RADIUS - obj_radius) < POP_PERCEPTION_RADIUS).float()
            
            phys = torch.stack([
                torch.cos(rel_ang),
                torch.sin(rel_ang),
                dist_norm
            ], dim=1) * mask.unsqueeze(1)
            
            onehot = one_hot_label.expand(POP_SIZE, 4, phys.shape[2])
            return torch.cat([phys, onehot], dim=1)

        # --- 1. 處理牆壁 (Wall) ---
        p = self.pos.unsqueeze(1)    # (N, 1, 2)
        a = self.wall_A.unsqueeze(0) # (1, W, 2)
        b = self.wall_B.unsqueeze(0) # (1, W, 2)

        # 計算投影點 (同之前邏輯)
        ab = b - a
        ap = p - a
        t = (torch.sum(ap * ab, dim=-1) / (torch.sum(ab * ab, dim=-1) + 1e-6)).clamp(0, 1)
        closest_points = a + t.unsqueeze(-1) * ab # (N, W, 2)

        # 計算相對位移與距離
        diff_wall = closest_points - p               # (N, W, 2)
        wall_dist = torch.norm(diff_wall, dim=-1)         # (N, W)

        # 計算相對角度 (相對於 Agent 目前的朝向 angle)
        abs_ang = torch.atan2(diff_wall[..., 1], diff_wall[..., 0])
        rel_ang = abs_ang - self.angle.view(POP_SIZE, 1)

        # 正規化距離與遮罩 (只看感知範圍內)
        wall_dist_norm = ((wall_dist - POP_RADIUS) / POP_PERCEPTION_RADIUS).clamp(0, 1)
        wall_mask = ((wall_dist - POP_RADIUS) < POP_PERCEPTION_RADIUS).float()

        # 構建物理特徵 (Cos, Sin, Dist)
        # wall_phys 維度: (POP_SIZE, 3, W)
        wall_phys = torch.stack([torch.cos(rel_ang), torch.sin(rel_ang), wall_dist_norm], dim=1) * wall_mask.unsqueeze(1)

        # 構建 One-hot Label (假設牆的標籤是 self.l_wall)
        # wall_in 維度: (POP_SIZE, 7, W)
        wall_in = torch.cat([
            wall_phys, 
            self.l_wall.expand(POP_SIZE, 4, self.wall_A.shape[0])
        ], dim=1)
        fill_buffer(wall_in)

        # --- 2. 處理食物 (Food) ---
        if FOOD_SIZE > 0:
            fill_buffer(process_obj(self.food_pos, FOOD_RADIUS, self.l_food))

        # --- 3. 處理隊友 (Team) ---
        if POP_SIZE > 1:
            fill_buffer(process_obj(self.pos, POP_RADIUS, self.l_team, is_self=True))

        # --- 4. 處理敵人 (Predator) ---
        if PREDATOR_SIZE > 0:
            fill_buffer(process_obj(self.pred_pos, PREDATOR_RADIUS, self.l_pred))

        speed = self.forward_speed / POP_MAX_SPEED
        last_steer = self.last_actions[:, 0]
        self_in = torch.stack([speed, last_steer, self.energy / MAX_ENERGY], dim=1)

        return (mixed_in, self_in)
    
    def update(self, training, move_food, move_predator):
        was_alive = self.alive.clone()
        current_states = self.get_states()
        
        # 取得連續動作並加入探索噪音
        with torch.no_grad():
            m, s = self.last_states
            actions, _ = self.actor(m, s, deterministic=not training, with_logprob=False)

        rewards = torch.full((POP_SIZE,), STEP_REWARD, device=DEVICE)
            
        # 物理運算 (載具動力學)
        # 0. 取得存活者
        alive_mask = self.alive.float().unsqueeze(1) 
        alive_mask_flat = self.alive.float()

        # 1. 取得控制值
        steer_vals = actions[:, 0]      # [-1, 1] 舵向控制值 [左, 右]
        throttle_vals = actions[:, 1]   # [-1, 1] 油門控制值 [退, 進]

        # 2. 速度與舵效計算
        speed_vals = torch.norm(self.vel, dim=1)
        
        # 基礎舵效：(0 -> 0.2 速度區間)
        sensitivity = torch.clamp(speed_vals / 0.2, min=0.0, max=1.0).pow(2.0)
        # 高速衰減：(MAX_SPEED * 0.875 -> MAX_SPEED 速度區間)
        high_speed_damping = torch.clamp(1.875 - (speed_vals / POP_MAX_SPEED), min=0.7, max=1.0)
        
        # 計算轉角增量並更新
        steer_delta = steer_vals * POP_MAX_STEER * sensitivity * high_speed_damping
        self.angle += steer_delta * alive_mask_flat

        # 3. 換算油門與推力
        throttle_factors = torch.where(throttle_vals > 0, self.throttle_factor, self.throttle_factor * POP_BACKWARD_FACTOR)
        throttles = (throttle_vals * throttle_factors).unsqueeze(1)

        # 計算車頭向量
        pop_vecs = torch.stack([torch.cos(self.angle), torch.sin(self.angle)], dim=1)
        
        # 推力 = 向量 * 油門
        thrust = pop_vecs * throttles

        # 4. 更新物理狀態
        # 速度更新：V = V * (1 - Damping) + Thrust
        self.vel = (self.vel * (1 - POP_DAMPING_FACTOR) + thrust) * alive_mask
        # 位置更新：P = P + V
        self.pos += self.vel * alive_mask

        # 5. 能量消耗
        static_cost = 0.2 * ENERGY_DECAY
        dynamic_cost = 0.8 * ENERGY_DECAY * torch.pow(throttle_vals, 2)
        self.energy -= (static_cost + dynamic_cost) * alive_mask_flat

        # --- 移動獎勵計算 ---
        # 1. 有效前進速度 (Dot product)
        self.forward_speed = torch.sum(self.vel * pop_vecs, dim=1)
        
        # 2. 基礎移動獎勵
        move_reward = self.forward_speed * MOVE_REWARD

        # 4. 控制油門效率懲罰
        throttle_threshold = 0.75
        diff = torch.abs(torch.abs(throttle_vals) - throttle_threshold)
        throttle_penalty = MOVE_REWARD * 0.5 * (torch.exp(diff * 3.0) - 1.0)

        # 5. 動作變動量懲罰 (動作平滑度)
        action_diff = actions - self.last_actions
        smooth_penalty = MOVE_REWARD * 0.5 * torch.sum(action_diff.pow(2), dim=1)

        # 6. 整合獎勵 (僅針對活著的個體加分)
        total_step_reward = move_reward - throttle_penalty - smooth_penalty
        rewards += total_step_reward * alive_mask_flat

        # 掠食者 (Predators)
        if PREDATOR_SIZE > 0:
            if move_predator:
                self.pred_pos, self.pred_vel = self.update_entities(
                    self.pred_pos, self.pred_vel, PREDATOR_RADIUS, min_speed=PREDATOR_MIN_SPEED, max_speed=PREDATOR_MAX_SPEED
                )
            # 被吃了
            dist_pred = torch.cdist(self.pos, self.pred_pos)
            killed = (dist_pred < POP_RADIUS + PREDATOR_RADIUS).any(dim=1) & self.alive
            rewards[killed] += KILLED_REWARD
            self.kill(killed)
            # 靠近告警
            pred_mask = (dist_pred - POP_RADIUS - PREDATOR_RADIUS < POP_ALERT_RADIUS).any(dim=1) & self.alive
            rewards[pred_mask] += PREDATOR_NEARBY_REWARD
        else:
            killed = torch.zeros(POP_SIZE, dtype=torch.bool, device=DEVICE)

        # 撞牆處理
        p = self.pos.unsqueeze(1)
        a = self.wall_A.unsqueeze(0)
        b = self.wall_B.unsqueeze(0)
        ab = b - a
        ap = p - a
        t = (torch.sum(ap * ab, dim=-1) / (torch.sum(ab * ab, dim=-1) + 1e-6)).clamp(0, 1)
        wall_closest_points = a + t.unsqueeze(-1) * ab
        dist_to_walls = torch.norm(self.pos.unsqueeze(1) - wall_closest_points, dim=-1)
        hit_walls = dist_to_walls < POP_RADIUS
        collided = torch.any(hit_walls, dim=1) & self.alive
        rewards[collided] += COLLIDED_REWARD
        self.kill(collided)
        
        # 移動食物 (Food)
        if FOOD_SIZE > 0:
            if move_food:
                self.food_pos, self.food_vel = self.update_entities(
                    self.food_pos, self.food_vel, FOOD_RADIUS, min_speed=0.5, max_speed=1.0
                )
            # 食物碰撞
            dist_food = torch.cdist(self.pos, self.food_pos)
            hits_food = (dist_food < POP_RADIUS + FOOD_RADIUS) & self.alive.unsqueeze(1)
            if hits_food.any():
                # 找出每個食物最近的捕食者
                masked_dist = torch.where(hits_food, dist_food, torch.tensor(float('inf'), device=DEVICE))
                min_dists, closest_a_idx = torch.min(masked_dist, dim=0)
                valid_eaten_mask = min_dists != float('inf')
                
                f_idx = torch.where(valid_eaten_mask)[0] 
                a_idx = closest_a_idx[valid_eaten_mask]
                
                # 結算碰撞獎勵與能量
                rewards.index_add_(0, a_idx, torch.full((len(a_idx),), FOOD_REWARD, device=DEVICE))
                self.energy.index_add_(0, a_idx, torch.full((len(a_idx),), FOOD_ENERGY, device=DEVICE))
                self.energy = torch.clamp(self.energy, max=MAX_ENERGY)
                
                # 更新食物座標
                self.food_pos[f_idx] = self.get_risky_pos(len(f_idx), 0.0)
                self.eaten += len(f_idx)

        # 能量耗盡
        starved = (self.energy <= 0) & self.alive
        rewards[starved] += STARVED_REWARD
        self.kill(starved)

        # 近牆痛覺
        wall_min_dist, closest_wall_idx = torch.min(dist_to_walls, dim=1)
        wall_dist_ratio = (1.0 - (wall_min_dist - POP_RADIUS) / POP_ALERT_RADIUS).clamp(0.0, 1.0)
        wall_mask = (wall_dist_ratio > 0) & self.alive
        if wall_mask.any():
            wall_contacts = wall_closest_points[torch.arange(self.pos.size(0), device=DEVICE), closest_wall_idx]
            starts = self.pos[wall_mask]
            ends = wall_contacts[wall_mask]
            self.wall_lines = torch.stack([starts, ends], dim=1)
            rewards[wall_mask] += WALL_NEARBY_REWARD
        else:
            self.wall_lines = None

        dead_mask = killed | collided | starved

        self.last_actions = actions.detach().clone()
        next_states = self.last_states = self.get_states()

        # 把經驗推入 Replay Buffer
        for i in range(POP_SIZE):
            if not was_alive[i]: 
                continue # 忽略死屍的經驗
            self.memory.push(
                current_states[0][i],
                current_states[1][i],
                actions[i], 
                rewards[i], 
                next_states[0][i], 
                next_states[1][i], 
                dead_mask[i]
            )

        ready_to_respawn = ~self.alive & (self.respawn_timer <= 0)
        if ready_to_respawn.any():
            indices = torch.where(ready_to_respawn)[0]
            self.pos[indices] = self.get_saftest_pos(len(indices))
            self.alive[indices] = True
            self.energy[indices] = MAX_ENERGY
            self.vel[indices] = 0.0

        self.respawn_timer[~self.alive] -= 1
        self.rewards_avg = self.rewards_avg * 0.99 + (rewards.sum().item() / POP_SIZE) * 0.01
        self.killed += killed.sum().item()
        self.collided += collided.sum().item()
        self.starved += starved.sum().item()
        self.rewards = rewards
        self.frames += 1

    def kill(self, killed):
        self.alive[killed] = False
        self.vel[killed] = 0.0 # 死掉後速度歸零
        self.respawn_timer[killed] = 1 if POP_SIZE == 1 else torch.randint(60, 360, (killed.sum(),), device=DEVICE)

    def get_saftest_pos(self, n):
        """
        一次為 n 個實體尋找安全位置。
        n: 需要生成的點數量。
        """
        # 候選樣本數
        num_samples = max(n * 10, max(POP_SIZE, PREDATOR_SIZE))
        samples = torch.empty((num_samples, 2), device=DEVICE)
        samples[:, 0].uniform_(RND_POS_PADDING, SCREEN_W - RND_POS_PADDING)
        samples[:, 1].uniform_(RND_POS_PADDING, SCREEN_H - RND_POS_PADDING)

        obstacles = torch.cat([self.pos[self.alive], self.pred_pos], dim=0)

        # 如果有掠食者，進行距離篩選
        if PREDATOR_SIZE > 0:
            # 計算矩陣：(候選點數, 掠食者數)
            dists = torch.cdist(samples, obstacles)
            
            # 找到每個候選點離「最近掠食者」的距離
            min_dists = dists.min(dim=1).values
            
            # 從中選出最遠（最安全）的前 n 個點
            _, top_indices = torch.topk(min_dists, k=n)
            
            # 如果需要的 n 比 top_indices 多（雖然機率極低），用 repeat 補齊
            return samples[top_indices]
        else:
            # 無掠食者時，直接隨機回傳 n 個點
            return samples[:n]

    def get_risky_pos(self, n, min_dist=50.0):
        """
        找尋離掠食者最近，但距離至少大於 min_threshold 的位置。
        """
        if n <= 0: return torch.empty((0, 2), device=DEVICE)

        # 增加樣本數
        num_samples = max(n * 20, PREDATOR_SIZE) 
        samples = torch.empty((num_samples, 2), device=DEVICE)
        samples[:, 0].uniform_(RND_POS_PADDING, SCREEN_W - RND_POS_PADDING)
        samples[:, 1].uniform_(RND_POS_PADDING, SCREEN_H - RND_POS_PADDING)

        if PREDATOR_SIZE:
            # 1. 計算所有樣本到掠食者的最近距離
            dists = torch.cdist(samples, self.pred_pos)
            min_dists = dists.min(dim=1).values
            
            # 2. 建立遮罩：找出符合「距離 > 50」條件的點
            valid_mask = min_dists >= min_dist

            # 3. 處理符合條件的樣本
            if valid_mask.any():
                valid_min_dists = min_dists[valid_mask]
                valid_samples = samples[valid_mask]
                
                # 4. 反向找「最近」: largest=False
                # 這會回傳 valid_min_dists 中數值最小的前 k 個
                _, top_indices = torch.topk(valid_min_dists, k=n, largest=False)
                return valid_samples[top_indices]
            else:
                # 如果全部點都離掠食者太近 (極端情況)，退而求其次回傳最遠的點
                _, top_indices = torch.topk(min_dists, k=n, largest=True)
                return samples[top_indices]
        else:
            # 無掠食者時，隨機回傳
            return samples[:n]
       
    def update_entities(self, pos, vel, radius, min_speed, max_speed, jitter_chance=0.05):
        # 1. 向量化隨機擾動
        change_mask = torch.rand(pos.shape[0], 1, device=DEVICE) < jitter_chance
        vel.add_((torch.rand_like(vel) - 0.5) * (1.8 * change_mask))

        # 2. 速度約束
        speeds = torch.norm(vel, dim=1, keepdim=True)
        new_speeds = torch.clamp(speeds, min_speed, max_speed)
        vel.mul_(new_speeds / (speeds + 1e-6))

        # 3. 更新位置
        pos.add_(vel)

        # 4. 高效邊界處理 (考慮半徑)
        min_bound = torch.full_like(self.bounds, radius)
        max_bound = self.bounds - radius
        out_of_bounds = (pos < min_bound) | (pos > max_bound)
        vel[out_of_bounds] *= -1.0
        
        # 修正位置：確保物體不會卡在牆外
        # 使用 clamp_ 將座標限制在 [radius, bounds - radius] 之間
        pos.clamp_(min_bound, max_bound)

        return pos, vel
    
    def respawn_food(self, f_idx, a_idx):
        num_to_respawn = len(f_idx)
        pad = RND_POS_PADDING  # 避免生在牆縫裡，Agent 很難吃到
        
        # 1. 為每個需要重生的食物生成一組候選點 (假設每個食物給 10 個候選)
        num_samples = 10
        candidates = torch.empty((num_to_respawn, num_samples, 2), device=DEVICE)
        candidates[..., 0].uniform_(pad, SCREEN_W - pad)
        candidates[..., 1].uniform_(pad, SCREEN_H - pad)

        # 2. 獲取對應 Agent 的位置 (假設每個食物索引 f_idx 對應一個 Agent)
        # 這裡需要確保你的 f_idx 邏輯能對應到正確的 Agent 位置
        # 假設是一對一，或者我們直接取 Agent 的平均位置作為避開點
        target_agent_pos = self.pos[a_idx].unsqueeze(1) # (num_res, 1, 2)

        # 3. 計算每個候選點到 Agent 的距離
        dists = torch.norm(candidates - target_agent_pos, dim=2) # (num_res, num_samples)

        # 4. 篩選策略：找距離在 [PERCEPTION_RADIUS, Max] 之間的
        # 我們先找出每個食物最遠的候選點作為保底
        farthest_idx = torch.argmax(dists, dim=1)
        
        # 建立一個 Mask，過濾掉太近的 (小於 200)
        valid_mask = dists > POP_PERCEPTION_RADIUS
        
        final_pos = torch.empty((num_to_respawn, 2), device=DEVICE)
        
        for i in range(num_to_respawn):
            valid_indices = torch.where(valid_mask[i])[0]
            if len(valid_indices) > 0:
                # 在合格的點中隨機選一個 (增加多樣性)
                sel_idx = valid_indices[torch.randint(0, len(valid_indices), (1,))]
                final_pos[i] = candidates[i, sel_idx]
            else:
                # 如果都太近，就取最遠的那個
                final_pos[i] = candidates[i, farthest_idx[i]]

        return final_pos
        
    def save_state(self):
        torch.save({
            'pos': self.pos,
            'energy': self.energy,
            'alive': self.alive,
            'respawn_timer': self.respawn_timer,
            'food_pos': self.food_pos,
            'pred_pos': self.pred_pos,
            'rewards_avg': self.rewards_avg,
            'eaten': self.eaten,
            'killed': self.killed,
            'collided': self.collided,
            'starved': self.starved
        }, SAVE_PATH)

        torch.save({
            'steps': self.steps,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'alpha_opt': self.alpha_opt.state_dict(),
        }, self.brain_path)
        shutil.copy2(self.brain_path, f"{BASE_PATH}/{script_name}_{self.actor.__class__.__name__}_{self.steps}.pt")

        self.print_info(True)

    def load_state(self):
        if os.path.exists(SAVE_PATH):
            try:
                state = torch.load(SAVE_PATH, map_location=DEVICE)
                self.pos = state['pos']
                self.energy = state['energy']
                self.alive = state['alive']
                self.respawn_timer = state['respawn_timer']
                self.food_pos = state['food_pos']
                self.pred_pos = state['pred_pos']
                self.rewards_avg = state['rewards_avg']
                self.eaten = state['eaten']
                self.killed = state['killed']
                self.collided = state['collided']
                self.starved = state['starved']
                print(f"--- [Loaded] Load completed ---")
            except Exception as e:
                print(f"--- [Error] Loading failed: {e} ---")

        if os.path.exists(self.brain_path):
            try:
                brain_state = torch.load(self.brain_path, map_location=DEVICE, weights_only=False)
                self.steps = brain_state['steps']
                self.actor.load_state_dict(brain_state['actor'])
                self.critic.load_state_dict(brain_state['critic'])
                self.critic_target.load_state_dict(brain_state['critic_target'])
                with torch.no_grad():
                    self.log_alpha.copy_(brain_state['log_alpha'])
                self.actor_opt.load_state_dict(brain_state['actor_opt'])
                self.critic_opt.load_state_dict(brain_state['critic_opt'])
                self.alpha_opt.load_state_dict(brain_state['alpha_opt'])
                print(f"--- [Loaded] brain weights {self.brain_path}, steps {self.steps:,} ---")
                return True
            except Exception as e:
                print(f"--- [Error] brain weights loading failed: {e} ---")

    def draw(self, draw_label, draw_units, draw_perception, draw_alert, verbose):
        self.fbo.use()
        self.ctx.clear(0.08, 0.08, 0.08)
        self.ui_surface.fill((0, 0, 0, 0)) 
        
        if draw_units:
            # 繪製牆面
            w_starts = self.wall_v[:, 0, :]
            w_ends = self.wall_v[:, 1, :]
            num_walls = w_starts.size(0)            
            wall_color = torch.tensor([[0.5, 0.0, 0.0]], device=DEVICE).expand(num_walls, -1)            
            self.renderer.draw_lines(w_starts, w_ends, wall_color)

            # 繪製食物 ---
            if len(self.food_pos) > 0:
                food_color = torch.tensor([[0.0, 1.0, 0.47]], device=DEVICE).expand(len(self.food_pos), -1)
                food_rad = torch.full((len(self.food_pos), 1), FOOD_RADIUS, device=DEVICE)
                self.renderer.draw_circles(self.food_pos, food_rad, food_color)

            # 繪製掠食者 (Predators) ---
            if len(self.pred_pos) > 0:
                num_pred = len(self.pred_pos)
                p_color = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE).expand(num_pred, -1)                
                # 外圈 (空心)
                p_rad_outer = torch.full((num_pred, 1), PREDATOR_RADIUS, device=DEVICE)
                self.renderer.circle_prog['is_hollow'].value = True
                self.renderer.circle_prog['thickness'].value = 0.05
                self.renderer.draw_circles(self.pred_pos, p_rad_outer, p_color * 0.5)
                # 中心實心小圓
                p_rad_inner = torch.full((num_pred, 1), PREDATOR_RADIUS * 0.25, device=DEVICE)
                self.renderer.circle_prog['is_hollow'].value = False
                self.renderer.draw_circles(self.pred_pos, p_rad_inner, p_color)

            # 繪製智能體 (Agents) ---
            alive_mask = self.alive > 0
            dead_mask = ~alive_mask
            
            # 處理死亡 Agents
            if dead_mask.any():
                d_pos = self.pos[dead_mask]
                d_color = torch.where((self.energy[dead_mask] <= 0).unsqueeze(1), 
                                      torch.tensor([0.23, 0.23, 0.23], device=DEVICE), 
                                      torch.tensor([0.47, 0.0, 0.0], device=DEVICE))
                d_rad = torch.full((d_pos.shape[0], 1), 3.0, device=DEVICE)
                self.renderer.draw_circles(d_pos, d_rad, d_color)

            # 處理存活 Agents
            if alive_mask.any():
                a_pos = self.pos[alive_mask]
                en_ratio = (self.energy[alive_mask] / MAX_ENERGY).unsqueeze(1)
                a_color = torch.cat([en_ratio, 0.5 * en_ratio, 1.0 - en_ratio], dim=1)
                a_rad = (POP_RADIUS + 4 * en_ratio)
                
                # --- 智能體主體 ---
                self.renderer.circle_prog['is_hollow'].value = False
                self.renderer.draw_circles(a_pos, a_rad, a_color)
                
                # --- 輔助範圍 ---
                if draw_perception:
                    per_rad = torch.full_like(a_rad, POP_PERCEPTION_RADIUS)
                    self.renderer.circle_prog['is_hollow'].value = True
                    self.renderer.circle_prog['thickness'].value = 0.01
                    self.renderer.draw_circles(a_pos, per_rad, a_color)
                if draw_alert:
                    alt_rad = torch.full_like(a_rad, POP_ALERT_RADIUS)
                    self.renderer.circle_prog['is_hollow'].value = True
                    self.renderer.circle_prog['thickness'].value = 0.01
                    self.renderer.draw_circles(a_pos, alt_rad, a_color)

                # --- 向量與指示線 ---
                # 速度向量
                spd = self.forward_speed[alive_mask]
                vel = self.vel[alive_mask]
                mask_spd = torch.abs(spd) > 0.1
                if mask_spd.any():
                    l_start = a_pos[mask_spd]
                    l_end = l_start + vel[mask_spd] * (torch.abs(spd[mask_spd]) * 2.5).unsqueeze(1)
                    l_color = torch.tensor([[0.0, 0.75, 1.0]], device=DEVICE).expand(l_start.shape[0], -1)
                    self.renderer.draw_lines(l_start, l_end, l_color)

                # 油門指示線
                throttle = self.last_actions[alive_mask, 1]
                mask_t = throttle != 0
                if mask_t.any():
                    t_start = a_pos[mask_t]
                    t_val = throttle[mask_t]
                    t_abs = torch.abs(t_val)
                    t_ang = self.angle[alive_mask][mask_t]
                    
                    # --- A. 計算長度與角度偏移 ---
                    # if throttle < 0: angle += pi, power /= 3
                    is_forward = t_val > 0
                    t_final_ang = torch.where(is_forward, t_ang, t_ang + np.pi)
                    t_final_len = torch.where(is_forward, 20.0 * t_abs, (20.0 * t_abs) / 3.0)
                    t_dir = torch.stack([torch.cos(t_final_ang), torch.sin(t_final_ang)], dim=1)
                    t_end = t_start + t_dir * t_final_len.unsqueeze(1)
                    
                    # --- B. 分段顏色邏輯 ---
                    num_t = t_start.size(0)
                    t_color = torch.zeros((num_t, 3), device=DEVICE)                    
                    # 條件 1: throttle < 0 -> 黃色 (1.0, 1.0, 0.0)
                    mask_back = ~is_forward
                    t_color[mask_back] = torch.tensor([1.0, 1.0, 0.0], device=DEVICE)
                    # 條件 2: throttle > 0 且 power <= 0.5 -> 綠色 (0.0, 1.0, 0.0)
                    mask_low = is_forward & (t_abs <= 0.5)
                    t_color[mask_low] = torch.tensor([0.0, 1.0, 0.0], device=DEVICE)
                    # 條件 3: throttle > 0 且 0.5 < power <= 0.875 -> 白色 (1.0, 1.0, 1.0)
                    mask_mid = is_forward & (t_abs > 0.5) & (t_abs <= 0.875)
                    t_color[mask_mid] = torch.tensor([1.0, 1.0, 1.0], device=DEVICE)
                    # 條件 4: throttle > 0 且 power > 0.875 -> 紅色 (1.0, 0.0, 0.0)
                    mask_high = is_forward & (t_abs > 0.875)
                    t_color[mask_high] = torch.tensor([1.0, 0.0, 0.0], device=DEVICE)
                    
                    # --- C. 執行渲染 ---
                    self.renderer.draw_lines(t_start, t_end, t_color)

                if verbose >= 1:
                    a_pos_np = a_pos.cpu().numpy()
                    a_en_np = self.energy[alive_mask].cpu().numpy()
                    a_rad_np = a_rad.cpu().numpy().flatten()
                    
                    if verbose >= 2:
                        a_thr_np = self.last_actions[alive_mask, 1].cpu().numpy()
                        a_spd_np = self.forward_speed[alive_mask].cpu().numpy()
                        a_rew_np = self.rewards[alive_mask].cpu().numpy()

                    for i in range(len(a_pos_np)):
                        px, py = a_pos_np[i]
                        radius = a_rad_np[i]
                        
                        # 顯示能量數值 (Agent 上方)
                        en_text = f"{a_en_np[i]:.0f}"
                        en_surf = self.font.render(en_text, True, (255, 255, 255))
                        en_rect = en_surf.get_rect(center=(int(px), int(py - radius - 8)))
                        self.ui_surface.blit(en_surf, en_rect)

                        # 顯示除錯訊息 (Agent 下方)
                        if verbose >= 2:
                            dbg_text = f"{a_thr_np[i]:.2f} {a_spd_np[i]:.2f} {a_rew_np[i]:.4f}"
                            dbg_surf = self.font.render(dbg_text, True, (255, 255, 255))
                            dbg_rect = dbg_surf.get_rect(center=(int(px), int(py + radius + 8)))
                            self.ui_surface.blit(dbg_surf, dbg_rect)
                            
            # --- 4. 除錯資訊 (牆壁感應) ---
            if verbose >= 2 and self.wall_lines is not None:
                w_starts = self.wall_lines[:, 0, :]
                w_ends = self.wall_lines[:, 1, :]
                num_w = w_starts.size(0)

                # 感應紅線
                w_line_color = torch.tensor([[1.0, 0.2, 0.2]], device=DEVICE).expand(num_w, -1)
                self.renderer.draw_lines(w_starts, w_ends, w_line_color)
                
                # 接觸黃點
                w_dot_color = torch.tensor([[1.0, 1.0, 0.0]], device=DEVICE).expand(num_w, -1)
                w_dot_rad = torch.full((num_w, 1), 3.0, device=DEVICE)
                self.renderer.draw_circles(w_ends, w_dot_rad, w_dot_color)

        THEME = {
            "label": (180, 180, 180),    # 淺灰色 (標籤專用)
            "perf": (100, 220, 180),     # 薄荷綠 (性能指標)
            "param": (255, 200, 100),    # 亮橘黃 (關鍵參數)
            "loss": (255, 120, 120),     # 柔和紅 (損失/負面指標)
            "success": (120, 255, 120)   # 翠綠色 (獎勵/存活)
        }
        right_labels = [
            ("FPS:", f"{self.fps_avg:.0f}", THEME["perf"], False)
        ]
        def render_label_column(labels, label_x, value_anchor_x, start_y=10, padding=4):
            current_y = start_y
            for text, val, val_color, bold in labels:
                font = self.big_font if bold else self.font
                lbl_surf = font.render(text, True, THEME["label"])
                lbl_rect = lbl_surf.get_rect(topleft=(label_x, current_y))                    
                val_surf = font.render(val, True, val_color)
                val_rect = val_surf.get_rect(topright=(value_anchor_x, current_y))
                self.ui_surface.blit(lbl_surf, lbl_rect)
                self.ui_surface.blit(val_surf, val_rect)
                line_height = max(lbl_rect.height, val_rect.height)
                current_y += line_height + padding

        render_label_column(right_labels, label_x=SCREEN_W - 70, value_anchor_x=SCREEN_W - 10)

        if draw_label:
            info = self.last_info
            left_labels = [
                ("Steps:", f"{self.steps:,}", THEME["perf"], True),
                ("Init-Alpha:", f"{INIT_ALPHA:.4f}", THEME["param"], False),
                ("Alpha:", f"{info['alpha']:.4f}", THEME["perf"], False),
                ("Entropy:", f"{info['entropy']:.4f}", THEME["perf"], False),
                ("Q-Val:", f"{info['q_val']:.4f}", THEME["perf"], False),
                ("C-Loss:", f"{info['c_loss']:.4f}", THEME["perf"], False),
                ("A-Loss:", f"{info['a_loss']:.4f}", THEME["perf"], False),
                ("Rewards:", f"{self.rewards_avg:.2f}", THEME["success"] if self.rewards_avg > 0 else (255, 0, 0), False),
                ("Eaten:", f"{self.eaten:,}", THEME["success"], False),
                ("Killed:", f"{self.killed:,}", THEME["loss"], False),
                ("Collided:", f"{self.collided:,}", THEME["loss"], False),
                ("Starved:", f"{self.starved:,}", THEME["loss"], False),
                ("Alive:", f"{int(self.alive.sum())}/{POP_SIZE}", THEME["success"], False)
            ]
            render_label_column(left_labels, label_x=10, value_anchor_x=200)

        self.renderer.draw_text(self.ui_surface)

        win_w, win_h = pygame.display.get_window_size()
        self.renderer.blit_to_screen(win_w, win_h)
        pygame.display.flip()
        self.clock.tick(self.fps)

    def run(self, args):
        running = True
        training = False if args.demo else True
        is_paused = False
        draw_label = True
        draw_units = True
        draw_alert = False
        draw_perception = False
        move_food = True
        move_predator = True
        verbose = 0
        video_thread = None
        frame_queue = None

        self.fps_avg = self.clock.get_fps()
        self.last_states = self.get_states()

        self.print_info(False)

        def start_record():
            nonlocal video_thread, frame_queue
            frame_queue = queue.Queue()
            filename = f"{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.mp4"
            video_thread = threading.Thread(
                target=self.record_proc,
                args=(frame_queue, filename)
            )
            video_thread.start()
            print(f"開始錄影...")
        def stop_record():
            nonlocal video_thread, frame_queue
            frame_queue.put(None)
            video_thread = None

        if args.record:
            start_record()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_UP:
                        self.fps += 5
                        self.update_caption()
                    elif event.key == pygame.K_DOWN:
                        self.fps -= 5
                        self.update_caption()
                    elif event.key == pygame.K_EQUALS:
                        self.fps = 300
                        self.update_caption()
                    elif event.key == pygame.K_MINUS:
                        self.fps = 60
                        self.update_caption()
                    elif event.key == pygame.K_z:
                        self.reset_env()
                    if event.key == pygame.K_z and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
                        self.init_network()
                    elif event.key == pygame.K_SPACE:
                        is_paused = not is_paused
                    elif event.key == pygame.K_t:
                        training = not training
                    elif event.key == pygame.K_l:
                        draw_label = not draw_label
                    elif event.key == pygame.K_u:
                        draw_units = not draw_units
                    elif event.key == pygame.K_a:
                        draw_alert = not draw_alert
                    elif event.key == pygame.K_p:
                        draw_perception = not draw_perception
                    elif event.key == pygame.K_f:
                        move_food = not move_food
                    elif event.key == pygame.K_m:
                        move_predator = not move_predator
                    elif event.key == pygame.K_v:
                        verbose = (verbose + 1) % 3
                    elif event.key == pygame.K_r:
                        if video_thread and video_thread.is_alive():
                            stop_record()
                        else:
                            start_record()

            if not is_paused:
                self.update(training, move_food, move_predator)
                if training:
                    updated = self.optimize_model()
                else:
                    updated = False

                if self.steps >= args.steps:
                    running = False
                elif updated and self.steps % 5000 == 0:
                    self.save_state()
                elif not training:
                    if self.frames % 1000 == 0:
                        self.print_info(False)
                    if self.frames >= args.frames:
                        running = False

            if args.headless:
                self.fps_avg = self.fps_avg * 0.99 + self.clock.get_fps() * 0.01
                self.clock.tick(self.fps)
            else:
                self.fps_avg = self.fps_avg * 0.99 + self.clock.get_fps() * 0.01
                self.draw(draw_label, draw_units, draw_perception, draw_alert, verbose)

                if video_thread and video_thread.is_alive():
                    self.fbo.read_into(self.pbo, components=3)
                    frame_queue.put(self.pbo.read())

        if video_thread and video_thread.is_alive():
            stop_record()

        if training:
            self.save_state()

        if not args.headless:
            pygame.quit()

    def record_proc(self, frame_queue, filename):
        cmd = (
            f'ffmpeg -hide_banner -loglevel error -y '
            f'-f rawvideo -pix_fmt rgb24 -s {SCREEN_W}x{SCREEN_H} -r 60 -i - '
            f'-vf vflip -c:v hevc_nvenc -preset p4 -tune hq -b:v 20M -pix_fmt yuv444p '
            f'-color_range pc -colorspace bt709 -color_primaries bt709 -color_trc bt709 '
            f'{filename}'
        )
        proc = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, bufsize=SCREEN_W*SCREEN_H*3)
        
        print(f"錄影已啟動: {filename}")
        while True:
            try:
                data = frame_queue.get(timeout=1.0)
                if data is None:
                    break
                proc.stdin.write(data)
            except queue.Empty:
                continue
        
        proc.stdin.close()
        proc.wait()
        print(f"錄影已關閉：{filename}")

    def print_info(self, training):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if training:
            info = self.last_info
            print(f"[{now}][Info] FPS:{self.fps_avg:.2f} Steps:{self.steps:,} Alpha:{info['alpha']:.4f} Entropy:{info['entropy']:.4f} Q-Val:{info['q_val']:.4f} C-Loss:{info['c_loss']:.4f} A-Loss:{info['a_loss']:.4f} Rewards:{self.rewards_avg:.4f} Eaten:{self.eaten:,} Killed:{self.killed:,} Collided:{self.collided:,} Starved:{self.starved:,}")
        else:
            print(f"[{now}][Info] FPS:{self.fps_avg:.2f} Frames:{self.frames:,} Rewards:{self.rewards_avg:.4f} Eaten:{self.eaten:,} Killed:{self.killed:,} Collided:{self.collided:,} Starved:{self.starved:,}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=CAPTION)
    parser.add_argument("-s", "--steps", type=int, default=float('inf'), help="達到此步數時退出")
    parser.add_argument("-r", "--record", action="store_true", default=False, help="啟動即開始錄影")
    parser.add_argument("--demo", action="store_true", default=False, help="模型性能展示")
    parser.add_argument("--frames", type=int, default=float('inf'), help="模型性能展示幀數")
    parser.add_argument("--headless", action="store_true", default=False, help="無頭模式")
    args = parser.parse_args()
    
    print(f'環境大小：{SCREEN_W} x {SCREEN_H}')
    print(f'任務步數：{EST_STEPS}')
    print(f'最大速度：{POP_MAX_SPEED:.2f}')
    print(f'預警半徑：{POP_ALERT_RADIUS:.2f}')
    sim = RLSimulation(args)
    sim.run(args)
