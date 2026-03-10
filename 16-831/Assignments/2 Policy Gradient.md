# Agent
```python
class PGAgent(BaseAgent):
	def train(self, observations, actions, rewards_list, next_observations, terminals):
		# input: list of rewards, output: concatenated q values
		q_values = self.calculate_q_vals(rewards_list)

        advantages = self.estimate_advantage(observations, rewards_list, q_values, terminals)

        train_log = self.actor.update(observations, actions, advantages, q_values)
	
	def sample(self, batch_size):
		return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)
```