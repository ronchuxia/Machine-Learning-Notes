# Method
## Policy Gradient
**REINFORCE:**
$$\nabla J = \nabla \log \pi (a | s) R$$

**REINFORCE with constant baseline:**
$$\nabla J = \nabla \log \pi (a | s) (R - b)$$

**REINFORCE with state-dependent baseline:**
$$\nabla J = \nabla \log \pi (a | s) (R - V(s))$$
- MC returns.

**Actor-Critic:**
$$\nabla J = \nabla \log \pi (a | s) (r + \gamma V(s') - V(s))$$
- TD returns.

**Advantage Actor-Critic (A2C):**
$$\nabla J = \nabla \log \pi (a | s) (R^{(n)} - V(s))$$
- n-step returns.

**TRPO:**
- Constrain KL divergence.

**PPO:**
- Clipped objective.
- Importance sampling.
- Generalized advantage estimation.

## Q-Function
**Policy Iteration:**
$$y = r(s) + \gamma Q_\theta(s', a')$$

**SARSA (Generalized Policy Iteration):**
$$y = r(s) + \gamma Q_\theta(s', a')$$

**Q-Learning:**
$$y = r(s) + \gamma \max_{a'} Q_\theta(s', a')$$

**Expected SARSA:**
$$y = r(s) + \gamma \mathbb E_{a' \sim \pi(a'|s')} [Q_\theta(s', a')]$$

**Soft Actor-Critic (SAC):**
$$y = r(s) + \gamma \mathbb{E}_{a' \sim \pi}[Q_\theta(s', a') - \alpha \log \pi(a' | s')]$$

**DQN:**
$$y = r + \gamma \max_{a' \in \mathcal A}Q_{\theta^-}(s', a')$$

**Double DQN:**
$$y = r(s) + \gamma Q_{\theta^-} \left( s', \arg\max_{a' \in \mathcal{A}} Q_\theta(s', a') \right)$$

**DPG/DDPG:**
$$y = r(s) + \gamma Q_{\theta^-}(s', \pi_\phi(s'))$$

**TD3:**
$$y = r(s) + \min \left( Q_{\theta_1^-}(s', \pi_\phi(s')), Q_{\theta_2^-}(s', \pi_\phi(s')) \right)$$

# On/Off Policy
On-policy
- REINFORCE
- Actor-Critic
- A2C
- TRPO
- PPO
- SARSA

Off-policy
- Policy iteration with off-policy MC
- Q-Learning
- Expected SARSA
- SAC
- DQN
- Double DQN
- DDPG
- TD3