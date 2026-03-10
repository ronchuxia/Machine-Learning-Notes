# Trainer
```python
class RL_Trainer(object):
	def __init__(self, params):
		self.env = gym.make(...)
		self.agent = ...
	
	def run_training_loop(self, ...):
		for itr in range(n_iter):
			training_returns = self.collect_training_trajectories(..., self.params['batch_size'])
			
			self.agent.add_to_replay_buffer(paths)
			
			self.train_agent()
	
	def train_agent(self):
		for train_step in range(self.params['num_agent_train_steps_per_iter']):
			# sample some data from the data buffer
			ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch = self.agent.sample(self.params['train_batch_size'])
			
			self.agent.train(ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch)
```

Training Loop:
For every training iteration:
1. Collect training trajectories: Load expert data (1st iteration) or execute current policy.
2. Add to replay buffer,
3. Conduct the training iteration (`self.train_agent()`).

Training Iteration:
For every training step:
1. Sample a batch of steps from the replay buffer.
2. Conduct the training step (`self.agent.train()`).

# Agent
```python
class BCAgent(BaseAgent):
	def __init__(self, env, agent_params):
		super(BCAgent, self).__init__()
		self.env = env
		
		# actor/policy
		self.actor = MLPPolicySL(...)
	
		# replay buffer
		self.replay_buffer = ReplayBuffer(...)
		
	def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
		self.actor.update(ob_no, ac_na)
		
	def add_to_replay_buffer(self, paths):
		self.replay_buffer.add_rollouts(paths)
	
	def sample(self, batch_size):
		return self.replay_buffer.sample_random_data(batch_size)
```

# Policy
```python
class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
	def __init__(self, ...):
		self.optimizer = optim.Adam(...)
		
	def get_action(self, obs):
		pass
	
	def update(self, observation, actions, **kwargs):
		pass
	
	def forward(self, observation):
		pass
```


