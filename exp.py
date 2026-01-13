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
    TRAIN_TIMESTEPS = 500000 #[修正] 大幅增加訓練步數以確保收斂
    UPDATE_TIMESTEP = 4096 #[修正] 增加 Batch Size 以平滑隨機梯度 (2048 -> 4096)
    
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
    LR_ACTOR = 1e-4 #[修正] 降低學習率，避免震盪
    LR_CRITIC = 3e-4        
    GAMMA = 0.99
    K_EPOCHS = 10
    EPS_CLIP = 0.1 #[修正] 降低 Clip Range，限制更新幅度          
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
    
    def __lt__(self, other): #[修正] 加入比較方法，防止 heapq 報錯
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
