Input: 
- State space, action space 
- Transition model $P(s' | s, a)$ 
- No reward function $R$ 
- Set of one or more teacher’s demonstrations $(s_0, a_0, s_1, a_1, . . .)$ (actions drawn from teacher’s policy $π^∗$)

Imitation learning is useful when it is easier for the expert to demonstrate the desired behavior rather than:
- Specifying a reward that would generate such behavior（设计 dense in time 的 reward 是困难的）
- Specifying the desired policy directly

# Behavior Cloning
Behavior cloning 将强化学习问题规约为监督学习问题，根据训练数据 $(s_0, a_0), (s_1, a_1), (s_2, a_2), \cdots$ 拟合策略。

Behavior cloning 存在 compounding errors 的问题，即当测试集策略与训练集策略（即最优策略）不一样时，测试集中状态的分布与训练中状态的分布会有很大的不同，导致测试的效果较差。

DAGGER 算法是对 behavior cloning 的改进，它在由 behavior cloning 得到的策略所经过的路径上收集最优策略的动作，以补充数据集：
- Initialize $D \leftarrow \emptyset$
- Initialize $\hat\pi_1$ to any policy in $\Pi$
- for $i = 1$ to N do
	- Let $\pi_i = \beta_i \pi^* + (1 - \beta_i) \hat\pi_i$
	- Sample T-step trajectories using $\pi_i$
	- Get dataset $D_i = \{(s, \pi^*(s))\}$ of visited states by $\pi_i$ and actions given by expert
	- Aggregate datasets: $D \leftarrow D \cup D_i$
	- Train calssifier $\hat\pi_{i+1}$ on D
- endfor
- Return best $\hat\pi_i$ on validation

DAGGER 的缺点是需要不断的人为（最优策略）监督。

# Inverse RL
Inverse RL 根据最优策略学习奖励函数。然而，对于一个策略，有无数个奖励函数使这个策略是最优策略（如对于 $R(s) = 0$，任意策略都是最优策略）。

TODO:
- Maximumum Entropy Inverse Reinforcement Learning (Ziebart et al. AAAI 2008) 
- Generative adversarial imitation learning (Ho and Ermon, NeurIPS 2016)






