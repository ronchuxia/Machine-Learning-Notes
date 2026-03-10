# Q-Learning
```python
class RL_Trainer(object):
	def run_training_loop(self, ...)
		for itr in range(n_iter + 1):
			self.agent.step_env()
			
			self.train_agent()
```

```python
class DQNAgent(object):
	def __init__(self, env, agent_params):
		...
		self.env = env
		self.critic = DQNCritic(...)
		self.actor = ArgMaxPolicy(self.critic)
		self.replay_buffer = MemoryOptimizedReplayBuffer(...)
		
	def step_env(self):
		"""Step the env and store the transition"""
		...
		
	def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
		self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)
		
		if self.num_param_updates % self.target_update_freq == 0:
			self.critic.update_target_network()
```

```python
class DQNCritic(BaseCritic):
	def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
		qa_t_values = self.q_net(ob_no) # shape (N, num_actions)
		q_t_values = torch.gather(qa_t_values, 1, ac_na.unsqueeze(1)).squeeze(1)
		
		# compute the Q-values from the target network 
        qa_tp1_values = self.q_net_target(next_ob_no)

		if self.double_q:
            max_ac_idx = torch.argmax(self.q_net(next_ob_no), dim=1)
            q_tp1 = torch.gather(qa_tp1_values, 1, max_ac_idx.unsqueeze(1)).squeeze(1)
        else:
            q_tp1, _ = qa_tp1_values.max(dim=1)
	
		# compute targets for minimizing Bellman error
		target = reward_n + self.gamma * q_tp1 * (torch.bitwise_not(terminal_n))
		target = target.detach()
		
		loss = self.loss(q_t_values, target)
		...
```

# Actor Critic
```python
class ACAgent(BaseAgent):
	def __init__(self, env, agent_params):
		super(ACAgent, self).__init__()
		self.env = env
		self.actor = MLPPolicyAC(...)
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()
        
    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
	    for ...:
		    self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)

		advantage = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)

		for ...:
			self.actor.update(ob_no, ac_na, advantage)
```

```python
class BootstrappedContinuousCritic(nn.Module, BaseCritic):
	def update(self, ob_no, ac_na, next_ob_no, reward_n, terminal_n):
		# do the following (
        # self.num_grad_steps_per_target_update * self.num_target_updates)
        # times:
        # every self.num_grad_steps_per_target_update steps (which includes the
        # first step), recompute the target values by
        #     a) calculating V(s') by querying the critic with next_ob_no
        #     b) and computing the target values as r(s, a) + gamma * V(s')
        # every time, update this critic using the observations and targets
```