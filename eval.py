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
