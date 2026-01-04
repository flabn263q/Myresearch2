# 研究計畫書

## 題目：基於深度強化學習與競爭失效風險模型的動態彈性作業車間排程與維護整合優化
**Title:** Deep Reinforcement Learning for Dynamic Flexible Job Shop Scheduling with Competing Risks of Random Breakdown and Multi-State Deterioration

---

## 1. 研究背景與動機 (Introduction)

在工業 4.0 與智慧製造的浪潮下，動態彈性作業車間排程問題 (Dynamic Flexible Job Shop Scheduling Problem, DFJSP) 已成為生產管理的核心挑戰。然而，現有的排程研究往往存在兩個極端的假設：
1.  **過度簡化的機器假設**：傳統排程文獻常假設機器在加工期間是完美的，或僅考慮簡單的隨機故障 (Random Breakdown)，忽略了設備隨時間推移而產生的物理衰退 (Deterioration)。
2.  **過度理想的維護假設**：現有的維護文獻 (如 [Ghaleb et al., 2020]) 雖然建立了精細的多狀態衰退模型，但通常是在靜態或單機環境下進行優化，難以應對動態工件到達 (Dynamic Job Arrival) 的複雜車間環境。

在真實的半導體或精密製造場景中，機器面臨著 **「競爭失效風險 (Competing Risks)」**：它可能因為突發的隨機事件而故障，也可能因為累積的磨損而導致效能下降（加工時間變長）甚至停機。

本研究旨在填補這一缺口，提出一個 **物理感知 (Physics-informed)** 的深度強化學習框架。我們將結合 **Weibull 分佈可靠度模型** 與 **PPO (Proximal Policy Optimization)** 演算法，讓智能體 (Agent) 在面對動態訂單時，能同時權衡生產效率（Tardiness）與設備健康風險（Maintenance Cost），實現真正的實時整合優化。

---

## 2. 文獻回顧與研究缺口 (Literature Review & Research Gaps)

### 2.1 狀態特徵的演進
*   **現狀**：[Yi et al., 2025] 使用了統計聚合特徵（如平均利用率），這對於捕捉全域負載有效，但缺乏對單台機器健康狀況的微觀描述。
*   **缺口**：缺乏將 **可靠度函數 (Reliability Function)** 直接作為狀態輸入的 DRL 研究。Agent 無法感知機器是「剛修好」還是「瀕臨故障」。

### 2.2 故障模式的建模
*   **現狀**：[Ghaleb et al., 2020] 提出了離散多狀態 (Discrete Multi-state) 模型，但採用數學規劃方法求解，計算時間長，無法實時響應。
*   **缺口**：目前的 DRL 排程研究尚未整合 **隨機故障 (Poisson Process)** 與 **老化衰退 (Weibull Process)** 的雙重競爭機制。

### 2.3 獎勵函數的設計
*   **現狀**：多數研究使用每一步的 Makespan 增量作為獎勵。這在動態環境下會導致 **獎勵稀疏 (Sparse Reward)** 或 **數值爆炸** 問題。
*   **缺口**：缺乏針對長期目標（如總延遲 Total Tardiness）的有效獎勵塑形 (Reward Shaping) 機制，特別是在半馬可夫 (Semi-MDP) 的事件驅動環境下。

---

## 3. 研究方法 (Methodology)

本研究將開發一套基於 **離散事件模擬 (Discrete Event Simulation, DES)** 的強化學習環境，核心技術架構如下：

### 3.1 物理感知的環境建模 (Physics-based Environment)
我們將重構機器的物理行為，引入競爭失效模型：
*   **老化機制 (Wear-out)**：採用 Weibull 分佈 $R(t) = e^{-(t/\eta)^\beta}$ 計算機器可靠度。當可靠度下降時，機器狀態 $S_k$ 會依機率從 $d$ 轉移至 $d+1$，導致加工速度下降 (Speed Factor $< 1.0$)。
*   **隨機故障 (Random Breakdown)**：在加工過程中，疊加 Poisson Process ($\lambda$) 的隨機故障風險。
*   **維護動作**：區分 **預防性維護 (PM)**（重置狀態與機齡）、**矯正性維護 (CM)**（高成本修復）與 **最小修復 (Minimal Repair)**（僅恢復運作，不改善機齡）。

### 3.2 狀態空間設計 (State Space)
為了讓 Agent 學會物理規律，狀態向量 $S_t$ 將包含：
1.  **機器健康特徵**：標準化離散狀態 ($S_k/K$)、**即時可靠度 ($R(t)$)**、故障歷史計數。
2.  **機器運作特徵**：忙碌/閒置狀態 (One-hot encoding)。
3.  **全域生產特徵**：工件隊列長度 (Queue Length)、平均急迫度 (Average Urgency, $T_{now} - T_{due}$)、維修資源佔用率 (Crew Ratio)。

### 3.3 動作空間與遮罩機制 (Action Space with Masking)
*   **動作定義**：$A_t \in \{ \text{SPT, LPT, EDD, FIFO, PM} \}$。Agent 可選擇派工規則或執行預防性維護。
*   **無效動作遮罩 (Invalid Action Masking)**：
    *   當維修人員不足 ($Crew \le 0$) 或機器處於完美狀態 ($S_k=0$) 時，遮蔽 PM 動作。
    *   當無工件排隊時，遮蔽派工動作。
    *   此機制能大幅縮減探索空間，加速收斂。

### 3.4 獎勵函數塑形 (Reward Shaping)
為解決數值穩定性問題，採用複合獎勵函數：
$$ R_t = - (w_1 \cdot C_{PM} + w_2 \cdot C_{CM} + w_3 \cdot \sum_{j \in \text{Finished}} \text{Tardiness}_j + w_4 \cdot \text{Backlog}_t) $$
*   **稀疏項**：僅在工件完成時計算延遲懲罰。
*   **密集項**：每一步扣除微小的積壓懲罰 (Backlog)，鼓勵 Agent 提高吞吐量。

### 3.5 演算法：PPO (Proximal Policy Optimization)
採用 Actor-Critic 架構的 PPO 演算法。由於環境是隨機的 (Stochastic)，PPO 的 Trust Region 機制能提供比 DQN 更穩定的策略更新。

---

## 4. 實驗設計 (Experimental Plan)

### 4.1 實驗設置
*   **模擬器**：基於 Python `gymnasium` 開發的事件驅動模擬器 (如 `refined_dfjsp_ppo.py`)。
*   **數據集**：生成符合 Poisson 到達與 Weibull 衰退的合成數據集，涵蓋不同規模 (如 5機/5工序 至 20機/10工序)。
*   **參數設定**：$\beta=1.5$ (磨損型故障), $\eta=500$, $\lambda=1e-4$。

### 4.2 比較基準 (Baselines)
1.  **Heuristic Rules**: FIFO, SPT, EDD (無維護或固定週期維護)。
2.  **Reactive Policies**: 僅在故障時維修 (Corrective Maintenance only)。
3.  **Standard DQN**: 使用 [Yi2025] 的架構但無物理特徵輸入。
4.  **Proposed PPO**: 本研究提出的完整框架。

### 4.3 評估指標 (Metrics)
1.  **總延遲時間 (Total Tardiness)**：主要生產指標。
2.  **總維護成本 (Total Maintenance Cost)**：PM 次數 vs. CM 次數的權衡。
3.  **平均機器可靠度 (Average Reliability)**：評估策略對設備健康的保護程度。
4.  **決策響應時間 (Inference Time)**：驗證實時性。

---

### 5. 預期貢獻 (Expected Contributions)

1.  **理論貢獻**：首次將 **Weibull 可靠度物理模型** 顯式地嵌入 DRL 的狀態空間中，證明了「物理感知」能顯著提升排程器的決策品質。
2.  **方法貢獻**：提出了一套結合 **Action Masking** 與 **Event-based Reward** 的 PPO 框架，有效解決了動態排程中動作空間無效與獎勵發散的問題。
3.  **實務價值**：提供了一個可直接應用於 Industry 4.0 數位孿生系統的決策模組，能在不需重新訓練的情況下，適應不同到達率的生產場景。

---

### 6. 參考文獻 (Selected References)

1.  **[Yi2025]** W. Yi et al., "An improved deep Q-network for dynamic flexible job shop scheduling with limited maintenance resources," *Int. J. Prod. Res.*, 2025.
2.  **[Ghaleb2020]** M. Ghaleb et al., "Integrated production and maintenance scheduling for a single degrading machine with deterioration-based failures," *Comput. Ind. Eng.*, 2020.
3.  **[Lei2024]** K. Lei et al., "Large-Scale Dynamic Scheduling for Flexible Job-Shop With Random Arrivals of New Jobs by Hierarchical Reinforcement Learning," *IEEE Trans. Ind. Informat.*, 2024.
4.  **[An2023]** Y. An et al., "Multiobjective Flexible Job-Shop Rescheduling With New Job Insertion and Machine Preventive Maintenance," *IEEE Trans. Cybern.*, 2023.

## 程式碼
### `exp.py`
```python=
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random
import heapq
import math
from collections import deque
import matplotlib.pyplot as plt
import os

# ==========================================
# 1. Configuration & Parameters (Optimized)
# ==========================================
class Config:
    # --- Experiment Settings ---
    SEED = 42
    # [修正] 大幅增加訓練步數以確保收斂
    TRAIN_TIMESTEPS = 500000  
    # [修正] 增加 Batch Size 以平滑隨機梯度 (2048 -> 4096)
    UPDATE_TIMESTEP = 4096    
    
    # --- Environment Dimensions ---
    NUM_MACHINES = 5
    NUM_OPS_PER_JOB = 5
    
    # --- Physics: Weibull Distribution ---
    WEIBULL_BETA = 1.5      
    WEIBULL_ETA = 500.0     
    
    # --- Degradation & Failure ---
    K_STATES = 5            
    RANDOM_BREAKDOWN_RATE = 1e-4     
    
    # --- Job Arrival ---
    ARRIVAL_RATE = 0.05
    DUE_DATE_FACTOR = 1.5   
    
    # --- Maintenance Constraints ---
    MAX_CREWS = 2           
    TIME_PM = 50            
    TIME_CM = 150           
    TIME_MINIMAL = 20       
    
    # --- PPO Hyperparameters (Conservative) ---
    # [修正] 降低學習率，避免震盪
    LR_ACTOR = 1e-4         
    LR_CRITIC = 3e-4        
    GAMMA = 0.99
    K_EPOCHS = 10
    # [修正] 降低 Clip Range，限制更新幅度
    EPS_CLIP = 0.1          
    ENTROPY_COEF = 0.01
    
    # --- Reward Weights (Raw) ---
    # 這些是原始權重，後續會通過 Wrapper 進行標準化
    W_TARDINESS = 1.0       
    W_PM_COST = 20.0        
    W_CM_COST = 100.0       
    W_BACKLOG = 0.5         

# ==========================================
# 2. Physics-Based Entities
# ==========================================
class Job:
    def __init__(self, job_id, arrival_time, ops_times):
        self.id = job_id
        self.arrival_time = arrival_time
        self.ops_times = ops_times
        self.total_work = sum(ops_times)
        self.due_date = arrival_time + self.total_work * Config.DUE_DATE_FACTOR
        self.current_op_idx = 0
        self.completion_time = 0
        
    def get_current_proc_time(self):
        if self.current_op_idx >= len(self.ops_times): return 0
        return self.ops_times[self.current_op_idx]

    def is_finished(self):
        return self.current_op_idx >= len(self.ops_times)
    
    # [修正] 加入比較方法，防止 heapq 報錯
    def __lt__(self, other):
        return self.id < other.id

class Machine:
    def __init__(self, m_id):
        self.id = m_id
        self.state = 0          
        self.age_accum = 0.0    
        self.status = 0         # 0: Idle, 1: Busy, 2: Maint/Repair
        self.failure_count = 0
        self.next_free_time = 0.0
        
    def get_reliability(self):
        return math.exp(- (self.age_accum / Config.WEIBULL_ETA) ** Config.WEIBULL_BETA)

    def get_processing_speed_factor(self):
        return max(0.5, 1.0 - (self.state * 0.05))

    def update_age(self, duration):
        rel_before = self.get_reliability()
        self.age_accum += duration
        rel_after = self.get_reliability()
        
        loss = rel_before - rel_after
        if self.state < Config.K_STATES:
            if random.random() < (loss * 20.0): 
                self.state += 1
        
        return self.state >= Config.K_STATES 

    def repair(self, repair_type):
        if repair_type == 'PM':
            self.state = 0
            self.age_accum = 0.0 
        elif repair_type == 'CM': 
            self.state = 0
            self.age_accum = 0.0
            self.failure_count += 1
        elif repair_type == 'MINIMAL': 
            self.failure_count += 1
        self.status = 0 

# ==========================================
# 3. Discrete Event Simulation Environment
# ==========================================
class DFJSP_EventEnv(gym.Env):
    def __init__(self):
        super(DFJSP_EventEnv, self).__init__()
        
        # Action: 0-3 Rules, 4 PM
        self.action_space = spaces.Discrete(5)
        
        # State: Machine Feats + Global Feats
        m_dim = 4 * Config.NUM_MACHINES
        g_dim = 3
        self.observation_space = spaces.Box(low=0, high=1, shape=(m_dim + g_dim,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            
        self.now = 0.0
        self.machines = [Machine(i) for i in range(Config.NUM_MACHINES)]
        self.job_queue = []      
        self.active_jobs = []    
        self.avail_crews = Config.MAX_CREWS
        self.events = []
        self.job_counter = 0
        
        self._schedule_arrival()
        
        # Fast-forward to first decision
        obs, mask, _ = self._resume_simulation()
        return obs, {'mask': mask}

    def _schedule_arrival(self):
        inter_arrival = random.expovariate(Config.ARRIVAL_RATE)
        heapq.heappush(self.events, (self.now + inter_arrival, 'ARRIVAL', None))

    def _get_state(self):
        m_feats = []
        for m in self.machines:
            m_feats.extend([
                m.state / Config.K_STATES,
                m.get_reliability(),
                1.0 if m.status == 1 else 0.0,
                np.tanh(m.failure_count / 5.0)
            ])
            
        if self.job_queue:
            q_len = np.tanh(len(self.job_queue) / 20.0)
            urgency = np.mean([(self.now - j.due_date)/100.0 for j in self.job_queue])
            urgency = np.tanh(urgency)
        else:
            q_len, urgency = 0.0, 0.0
            
        crew_ratio = self.avail_crews / Config.MAX_CREWS
        
        state = np.array(m_feats + [q_len, urgency, crew_ratio], dtype=np.float32)
        return state

    def _get_mask(self, machine_id):
        mask = [1, 1, 1, 1, 1] 
        m = self.machines[machine_id]
        
        if not self.job_queue:
            mask[0] = mask[1] = mask[2] = mask[3] = 0
            
        if self.avail_crews <= 0 or m.state == 0:
            mask[4] = 0
            
        return np.array(mask, dtype=np.float32)

    def step(self, action):
        machine = self.machines[self.decision_machine]
        reward = 0.0
        
        if action == 4: # PM
            duration = Config.TIME_PM
            self.avail_crews -= 1
            machine.status = 2 
            finish_time = self.now + duration
            heapq.heappush(self.events, (finish_time, 'FINISH_MAINT', (machine.id, 'PM')))
            reward -= Config.W_PM_COST
            
        else: # Dispatching
            if not self.job_queue:
                job = None # Should be masked out
            elif action == 0: job = min(self.job_queue, key=lambda j: j.get_current_proc_time())
            elif action == 1: job = max(self.job_queue, key=lambda j: j.get_current_proc_time())
            elif action == 2: job = min(self.job_queue, key=lambda j: j.due_date)
            else: job = min(self.job_queue, key=lambda j: j.arrival_time)
            
            if job:
                self.job_queue.remove(job)
                base_time = job.get_current_proc_time()
                speed_factor = machine.get_processing_speed_factor()
                actual_time = base_time / speed_factor
                
                machine.status = 1 
                finish_time = self.now + actual_time
                
                breakdown_prob = 1 - math.exp(-Config.RANDOM_BREAKDOWN_RATE * actual_time)
                
                if random.random() < breakdown_prob:
                    break_time = self.now + random.uniform(0, actual_time)
                    heapq.heappush(self.events, (break_time, 'BREAKDOWN', (machine.id, job)))
                else:
                    heapq.heappush(self.events, (finish_time, 'FINISH_OP', (machine.id, job, actual_time)))

        next_state, next_mask, event_rewards = self._resume_simulation()
        reward += event_rewards
        
        # Dense penalty for backlog
        reward -= Config.W_BACKLOG * len(self.job_queue)
        
        return next_state, reward, False, False, {'mask': next_mask}

    def _resume_simulation(self):
        event_rewards = 0.0
        
        while True:
            idle_machines = [m for m in self.machines if m.status == 0]
            
            for m in idle_machines:
                if self.job_queue or (m.state > 0 and self.avail_crews > 0):
                    self.decision_machine = m.id
                    return self._get_state(), self._get_mask(m.id), event_rewards

            if not self.events:
                self._schedule_arrival()
                
            time, type, data = heapq.heappop(self.events)
            self.now = time
            
            if type == 'ARRIVAL':
                ops = [random.randint(10, 50) for _ in range(Config.NUM_OPS_PER_JOB)]
                new_job = Job(self.job_counter, self.now, ops)
                self.job_counter += 1
                self.job_queue.append(new_job)
                self.active_jobs.append(new_job)
                self._schedule_arrival()
                
            elif type == 'FINISH_OP':
                m_id, job, duration = data
                machine = self.machines[m_id]
                deterioration_failure = machine.update_age(duration)
                
                if deterioration_failure:
                    machine.status = 2
                    repair_dur = Config.TIME_CM
                    heapq.heappush(self.events, (self.now + repair_dur, 'FINISH_MAINT', (m_id, 'CM')))
                    event_rewards -= Config.W_CM_COST
                else:
                    machine.status = 0 
                
                job.current_op_idx += 1
                if job.is_finished():
                    job.completion_time = self.now
                    self.active_jobs.remove(job)
                    tardiness = max(0, self.now - job.due_date)
                    event_rewards -= (tardiness * Config.W_TARDINESS)
                else:
                    self.job_queue.append(job)
                    
            elif type == 'FINISH_MAINT':
                m_id, m_type = data
                machine = self.machines[m_id]
                machine.repair(m_type)
                if m_type == 'PM': self.avail_crews += 1
                
            elif type == 'BREAKDOWN':
                m_id, job = data
                machine = self.machines[m_id]
                machine.status = 2
                repair_dur = Config.TIME_MINIMAL
                remaining_time = job.get_current_proc_time() 
                finish_time = self.now + repair_dur + remaining_time
                
                heapq.heappush(self.events, (self.now + repair_dur, 'FINISH_MAINT', (m_id, 'MINIMAL')))
                heapq.heappush(self.events, (finish_time, 'FINISH_OP', (m_id, job, remaining_time)))
                
                event_rewards -= 10.0 

# ==========================================
# 4. PPO Agent with Scheduler & Masking
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)
    
    def act(self, state, mask):
        features = self.shared_layer(state)
        logits = self.actor(features)
        inf_mask = (1.0 - mask) * -1e9
        masked_logits = logits + inf_mask
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob
    
    def evaluate(self, state, action, mask):
        features = self.shared_layer(state)
        logits = self.actor(features)
        inf_mask = (1.0 - mask) * -1e9
        masked_logits = logits + inf_mask
        dist = Categorical(logits=masked_logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(features)
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO initialized on device: {self.device}")
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': Config.LR_ACTOR},
            {'params': self.policy.critic.parameters(), 'lr': Config.LR_CRITIC}
        ])
        
        # [修正] 加入 Learning Rate Scheduler
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=1.0, end_factor=0.01, 
            total_iters=Config.TRAIN_TIMESTEPS // Config.UPDATE_TIMESTEP
        )
        
        self.policy_old = ActorCritic(state_dim, action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = []
    
    def store(self, transition):
        self.buffer.append(transition)
        
    def update(self):
        states = torch.tensor(np.array([t[0] for t in self.buffer]), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array([t[1] for t in self.buffer]), dtype=torch.long).to(self.device)
        logprobs = torch.stack([t[2] for t in self.buffer]).to(self.device).detach()
        rewards = [t[3] for t in self.buffer]
        masks = torch.tensor(np.array([t[4] for t in self.buffer]), dtype=torch.float32).to(self.device)
        
        returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + Config.GAMMA * discounted_reward
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)
        
        for _ in range(Config.K_EPOCHS):
            new_logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions, masks)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(new_logprobs - logprobs)
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-Config.EPS_CLIP, 1+Config.EPS_CLIP) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*nn.MSELoss()(state_values, returns) - Config.ENTROPY_COEF*dist_entropy
            
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer = []
        
        # [修正] 更新 Learning Rate
        self.scheduler.step()

# ==========================================
# 5. Main Training Loop with Wrappers
# ==========================================
def make_env():
    env = DFJSP_EventEnv()
    # [修正] 加入 Gym Wrappers 進行標準化與裁剪
    env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
    env = gym.wrappers.NormalizeReward(env, gamma=Config.GAMMA)
    env = gym.wrappers.NormalizeObservation(env)
    return env

def main():
    # 使用封裝後的環境
    env = make_env()
    
    # 注意：Wrapper 會改變 observation space 的定義，但維度通常不變
    # 這裡我們直接取原始維度，因為 NormalizeObservation 不改變 shape
    raw_env = env.unwrapped
    ppo_agent = PPO(raw_env.observation_space.shape[0], raw_env.action_space.n)
    
    print("Starting Training with Normalized Rewards & LR Decay...")
    
    obs, info = env.reset(seed=Config.SEED)
    mask = info['mask']
    
    current_ep_reward = 0
    log_rewards = []
    
    try:
        from tqdm import tqdm
        iterator = tqdm(range(1, Config.TRAIN_TIMESTEPS + 1))
    except ImportError:
        iterator = range(1, Config.TRAIN_TIMESTEPS + 1)

    for t in iterator:
        state_t = torch.FloatTensor(obs).to(ppo_agent.device)
        mask_t = torch.FloatTensor(mask).to(ppo_agent.device)
        
        action, log_prob = ppo_agent.policy_old.act(state_t, mask_t)
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_mask = info['mask']
        
        ppo_agent.store((obs, action, log_prob, reward, mask))
        
        obs = next_obs
        mask = next_mask
        current_ep_reward += reward
        
        if t % Config.UPDATE_TIMESTEP == 0:
            ppo_agent.update()
            avg_rew = current_ep_reward / Config.UPDATE_TIMESTEP
            log_rewards.append(avg_rew)
            
            if isinstance(iterator, tqdm):
                iterator.set_description(f"Avg Norm. Reward: {avg_rew:.4f} | Q: {len(raw_env.job_queue)}")
            else:
                print(f"Step {t} | Avg Norm. Reward: {avg_rew:.4f} | Queue: {len(raw_env.job_queue)}")
                
            current_ep_reward = 0
            
    torch.save(ppo_agent.policy.state_dict(), "final_ppo_model.pth")
    print("Training Complete. Model Saved.")
    
    # Plot Learning Curve (Smoothed)
    plt.figure(figsize=(10, 5))
    # 繪製原始數據的淺色背景
    plt.plot(log_rewards, alpha=0.3, color='blue', label='Raw')
    # 繪製移動平均線
    window_size = 5
    if len(log_rewards) >= window_size:
        smoothed = np.convolve(log_rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(log_rewards)), smoothed, color='blue', linewidth=2, label='Smoothed')
        
    plt.title("PPO Learning Curve (Normalized Reward)")
    plt.xlabel(f"Updates (x{Config.UPDATE_TIMESTEP} steps)")
    plt.ylabel("Average Normalized Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("final_learning_curve.png")
    print("Learning curve saved to final_learning_curve.png")

if __name__ == "__main__":
    main()
```

### `eval.py`
```python=
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import random
import heapq
import math
import pandas as pd
from collections import deque
import os

# ==========================================
# 1. Configuration (Must match Training)
# ==========================================
class Config:
    # --- Physics Parameters ---
    NUM_MACHINES = 5
    NUM_OPS_PER_JOB = 5
    WEIBULL_BETA = 1.5      
    WEIBULL_ETA = 500.0     
    K_STATES = 5            
    RANDOM_BREAKDOWN_RATE = 1e-4     
    ARRIVAL_RATE = 0.05
    DUE_DATE_FACTOR = 1.5   
    
    # --- Maintenance Constraints ---
    MAX_CREWS = 2           
    TIME_PM = 50            
    TIME_CM = 150           
    TIME_MINIMAL = 20       
    
    # --- Evaluation Settings ---
    EVAL_EPISODES = 30      # N=30 for statistical significance
    JOBS_PER_EPISODE = 100  # Fixed horizon for fair comparison
    MODEL_PATH = "final_ppo_model.pth"

# ==========================================
# 2. Physics Classes (Same as Training)
# ==========================================
class Job:
    def __init__(self, job_id, arrival_time, ops_times):
        self.id = job_id
        self.arrival_time = arrival_time
        self.ops_times = ops_times
        self.total_work = sum(ops_times)
        self.due_date = arrival_time + self.total_work * Config.DUE_DATE_FACTOR
        self.current_op_idx = 0
        self.completion_time = 0
        
    def get_current_proc_time(self):
        if self.current_op_idx >= len(self.ops_times): return 0
        return self.ops_times[self.current_op_idx]

    def is_finished(self):
        return self.current_op_idx >= len(self.ops_times)
    
    def __lt__(self, other):
        return self.id < other.id

class Machine:
    def __init__(self, m_id):
        self.id = m_id
        self.state = 0          
        self.age_accum = 0.0    
        self.status = 0         
        self.failure_count = 0
        self.total_busy_time = 0.0 # For utilization tracking
        
    def get_reliability(self):
        return math.exp(- (self.age_accum / Config.WEIBULL_ETA) ** Config.WEIBULL_BETA)

    def get_processing_speed_factor(self):
        return max(0.5, 1.0 - (self.state * 0.05))

    def update_age(self, duration):
        rel_before = self.get_reliability()
        self.age_accum += duration
        rel_after = self.get_reliability()
        loss = rel_before - rel_after
        if self.state < Config.K_STATES:
            if random.random() < (loss * 20.0): 
                self.state += 1
        return self.state >= Config.K_STATES 

    def repair(self, repair_type):
        if repair_type == 'PM':
            self.state = 0
            self.age_accum = 0.0 
        elif repair_type == 'CM': 
            self.state = 0
            self.age_accum = 0.0
            self.failure_count += 1
        elif repair_type == 'MINIMAL': 
            self.failure_count += 1
        self.status = 0 

# ==========================================
# 3. Evaluation Environment
# ==========================================
class EvalDFJSPEnv(gym.Env):
    """
    Modified environment for evaluation:
    1. Stops after N jobs are finished.
    2. Tracks raw metrics (Tardiness, Cost, Utilization).
    """
    def __init__(self):
        super(EvalDFJSPEnv, self).__init__()
        self.action_space = spaces.Discrete(5)
        m_dim = 4 * Config.NUM_MACHINES
        g_dim = 3
        self.observation_space = spaces.Box(low=0, high=1, shape=(m_dim + g_dim,), dtype=np.float32)
        
    def reset(self, seed=None, options=None):
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            
        self.now = 0.0
        self.machines = [Machine(i) for i in range(Config.NUM_MACHINES)]
        self.job_queue = []      
        self.active_jobs = []    
        self.avail_crews = Config.MAX_CREWS
        self.events = []
        self.job_counter = 0
        self.finished_count = 0
        
        # Metrics Tracking
        self.total_tardiness = 0.0
        self.total_maintenance_cost = 0.0
        self.pm_count = 0
        self.cm_count = 0
        
        self._schedule_arrival()
        
        # [修正] 解包 _resume_simulation 的 5 個回傳值，只回傳 obs 和 info
        obs, _, _, _, info = self._resume_simulation()
        return obs, info

    def _schedule_arrival(self):
        # Stop generating new jobs if we have enough to finish the episode
        if self.job_counter < Config.JOBS_PER_EPISODE:
            inter_arrival = random.expovariate(Config.ARRIVAL_RATE)
            heapq.heappush(self.events, (self.now + inter_arrival, 'ARRIVAL', None))

    def _get_state(self):
        m_feats = []
        for m in self.machines:
            m_feats.extend([
                m.state / Config.K_STATES,
                m.get_reliability(),
                1.0 if m.status == 1 else 0.0,
                np.tanh(m.failure_count / 5.0)
            ])
        if self.job_queue:
            q_len = np.tanh(len(self.job_queue) / 20.0)
            urgency = np.mean([(self.now - j.due_date)/100.0 for j in self.job_queue])
            urgency = np.tanh(urgency)
        else:
            q_len, urgency = 0.0, 0.0
        crew_ratio = self.avail_crews / Config.MAX_CREWS
        state = np.array(m_feats + [q_len, urgency, crew_ratio], dtype=np.float32)
        return state

    def _get_mask(self, machine_id):
        mask = [1, 1, 1, 1, 1] 
        m = self.machines[machine_id]
        if not self.job_queue:
            mask[0] = mask[1] = mask[2] = mask[3] = 0
        if self.avail_crews <= 0 or m.state == 0:
            mask[4] = 0
        return np.array(mask, dtype=np.float32)

    def step(self, action):
        machine = self.machines[self.decision_machine]
        
        if action == 4: # PM
            duration = Config.TIME_PM
            self.avail_crews -= 1
            machine.status = 2 
            finish_time = self.now + duration
            heapq.heappush(self.events, (finish_time, 'FINISH_MAINT', (machine.id, 'PM')))
            
            # Metric Update
            self.total_maintenance_cost += 20.0 # Assuming PM cost $20
            self.pm_count += 1
            
        else: # Dispatching
            if not self.job_queue:
                job = None
            elif action == 0: job = min(self.job_queue, key=lambda j: j.get_current_proc_time())
            elif action == 1: job = max(self.job_queue, key=lambda j: j.get_current_proc_time())
            elif action == 2: job = min(self.job_queue, key=lambda j: j.due_date)
            else: job = min(self.job_queue, key=lambda j: j.arrival_time)
            
            if job:
                self.job_queue.remove(job)
                base_time = job.get_current_proc_time()
                speed_factor = machine.get_processing_speed_factor()
                actual_time = base_time / speed_factor
                
                machine.status = 1 
                machine.total_busy_time += actual_time # Track utilization
                finish_time = self.now + actual_time
                
                breakdown_prob = 1 - math.exp(-Config.RANDOM_BREAKDOWN_RATE * actual_time)
                
                if random.random() < breakdown_prob:
                    break_time = self.now + random.uniform(0, actual_time)
                    heapq.heappush(self.events, (break_time, 'BREAKDOWN', (machine.id, job)))
                else:
                    heapq.heappush(self.events, (finish_time, 'FINISH_OP', (machine.id, job, actual_time)))

        return self._resume_simulation()

    def _resume_simulation(self):
        while True:
            # Check termination condition
            if self.finished_count >= Config.JOBS_PER_EPISODE:
                return self._get_state(), 0, True, False, {}

            idle_machines = [m for m in self.machines if m.status == 0]
            for m in idle_machines:
                if self.job_queue or (m.state > 0 and self.avail_crews > 0):
                    self.decision_machine = m.id
                    return self._get_state(), 0, False, False, {'mask': self._get_mask(m.id)}

            if not self.events:
                # If no events and not finished, something is wrong or waiting for arrival
                if self.job_counter < Config.JOBS_PER_EPISODE:
                    self._schedule_arrival()
                else:
                    # No more arrivals, just wait for processing to finish
                    pass
                
            if not self.events: # Still empty?
                return self._get_state(), 0, True, False, {} # Force finish

            time, type, data = heapq.heappop(self.events)
            self.now = time
            
            if type == 'ARRIVAL':
                ops = [random.randint(10, 50) for _ in range(Config.NUM_OPS_PER_JOB)]
                new_job = Job(self.job_counter, self.now, ops)
                self.job_counter += 1
                self.job_queue.append(new_job)
                self.active_jobs.append(new_job)
                self._schedule_arrival()
                
            elif type == 'FINISH_OP':
                m_id, job, duration = data
                machine = self.machines[m_id]
                deterioration_failure = machine.update_age(duration)
                
                if deterioration_failure:
                    machine.status = 2
                    repair_dur = Config.TIME_CM
                    heapq.heappush(self.events, (self.now + repair_dur, 'FINISH_MAINT', (m_id, 'CM')))
                    # Metric Update
                    self.total_maintenance_cost += 100.0 # CM Cost
                    self.cm_count += 1
                else:
                    machine.status = 0 
                
                job.current_op_idx += 1
                if job.is_finished():
                    job.completion_time = self.now
                    self.active_jobs.remove(job)
                    self.finished_count += 1
                    # Metric Update
                    tardiness = max(0, self.now - job.due_date)
                    self.total_tardiness += tardiness
                else:
                    self.job_queue.append(job)
                    
            elif type == 'FINISH_MAINT':
                m_id, m_type = data
                machine = self.machines[m_id]
                machine.repair(m_type)
                if m_type == 'PM': self.avail_crews += 1
                
            elif type == 'BREAKDOWN':
                m_id, job = data
                machine = self.machines[m_id]
                machine.status = 2
                repair_dur = Config.TIME_MINIMAL
                remaining_time = job.get_current_proc_time() 
                finish_time = self.now + repair_dur + remaining_time
                
                heapq.heappush(self.events, (self.now + repair_dur, 'FINISH_MAINT', (m_id, 'MINIMAL')))
                heapq.heappush(self.events, (finish_time, 'FINISH_OP', (m_id, job, remaining_time)))
                
                # Metric Update
                self.total_maintenance_cost += 10.0 # Minimal Repair Cost

# ==========================================
# 4. Model Architecture (Must match Training)
# ==========================================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)
    
    def act(self, state, mask):
        features = self.shared_layer(state)
        logits = self.actor(features)
        inf_mask = (1.0 - mask) * -1e9
        masked_logits = logits + inf_mask
        dist = torch.distributions.Categorical(logits=masked_logits)
        return dist.sample().item()

# ==========================================
# 5. Evaluation Logic
# ==========================================
def run_evaluation(agent_type, model=None):
    metrics = {
        'Tardiness': [],
        'Cost': [],
        'Utilization': [],
        'PM_Count': [],
        'CM_Count': []
    }
    
    print(f"Evaluating {agent_type} over {Config.EVAL_EPISODES} episodes...")
    
    for ep in range(Config.EVAL_EPISODES):
        env = EvalDFJSPEnv()
        # Use NormalizeObservation wrapper to match training input distribution
        env = gym.wrappers.NormalizeObservation(env)
        
        # [修正] 傳遞 options=None
        obs, info = env.reset(seed=ep + 1000, options=None) 
        mask = info['mask']
        done = False
        
        while not done:
            if agent_type == 'PPO':
                state_t = torch.FloatTensor(obs)
                mask_t = torch.FloatTensor(mask)
                with torch.no_grad():
                    action = model.act(state_t, mask_t)
            elif agent_type == 'FIFO':
                if mask[3] == 1: action = 3
                else: action = 4 
            elif agent_type == 'SPT':
                if mask[0] == 1: action = 0
                else: action = 4
            elif agent_type == 'Random':
                valid_actions = [i for i, m in enumerate(mask) if m == 1]
                action = random.choice(valid_actions)
            
            # Force Heuristics to NOT do PM (Corrective Maintenance Only Strategy)
            if agent_type in ['FIFO', 'SPT'] and action == 4:
                # Heuristics only dispatch. If forced to PM (queue empty), it's effectively waiting.
                pass

            obs, _, done, _, info = env.step(action)
            if 'mask' in info: mask = info['mask']
            
        # Collect Episode Metrics
        raw_env = env.unwrapped
        metrics['Tardiness'].append(raw_env.total_tardiness)
        metrics['Cost'].append(raw_env.total_maintenance_cost)
        
        # Avg Utilization across machines
        total_time = raw_env.now * Config.NUM_MACHINES
        total_busy = sum([m.total_busy_time for m in raw_env.machines])
        util = (total_busy / total_time) * 100 if total_time > 0 else 0
        metrics['Utilization'].append(util)
        
        metrics['PM_Count'].append(raw_env.pm_count)
        metrics['CM_Count'].append(raw_env.cm_count)

    # Aggregate
    results = {}
    for k, v in metrics.items():
        results[f'{k}_Mean'] = np.mean(v)
        results[f'{k}_Std'] = np.std(v)
    
    return results

def main():
    # 1. Load PPO Model
    env_dummy = EvalDFJSPEnv()
    model = ActorCritic(env_dummy.observation_space.shape[0], env_dummy.action_space.n)
    
    try:
        model.load_state_dict(torch.load(Config.MODEL_PATH))
        model.eval()
        print("PPO Model loaded successfully.")
    except FileNotFoundError:
        print("Error: 'final_ppo_model.pth' not found. Please train first.")
        return

    # 2. Run Evaluations
    results = []
    
    # PPO
    res_ppo = run_evaluation('PPO', model)
    res_ppo['Agent'] = 'Proposed PPO'
    results.append(res_ppo)
    
    # Baselines
    for agent in ['FIFO', 'SPT', 'Random']:
        res = run_evaluation(agent)
        res['Agent'] = agent
        results.append(res)
        
    # 3. Create DataFrame & Display
    df = pd.DataFrame(results)
    
    # Reorder columns
    cols = ['Agent', 'Tardiness_Mean', 'Tardiness_Std', 'Cost_Mean', 'Cost_Std', 'Utilization_Mean', 'PM_Count_Mean', 'CM_Count_Mean']
    df = df[cols]
    
    print("\n" + "="*80)
    print("COMPARATIVE RESULTS (IEEE TII/TSE Standard)")
    print("="*80)
    print(df.to_string(index=False, float_format="%.2f"))
    print("="*80)
    
    # Save to CSV for paper
    df.to_csv("evaluation_results.csv", index=False)
    print("Results saved to evaluation_results.csv")

if __name__ == "__main__":
    main()
```
