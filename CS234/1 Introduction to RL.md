# Sequential Decision Making
- Action: $a_t$
- Observation: $o_t$
- Reward: $r_t$
- History: $h_t = (a_1, o_1, r_1, ..., a_t, o_t, r_t)$
	- squence of past observations, actions & rewards
- State: $s_t = f(h_t)$
	- information assumed to determine what happens next
	- function of history 

# Markov assumption
$$p(s_{t+1} | s_t, a_t) = p(s_{t+1} | h_t, a_t)$$
- 假设状态转移概率和奖励函数只依赖于当前的状态和动作，而与过去的状态和动作序列无关
- 如果以历史作为状态，即 $s_t = h_t$，则任何过程都满足马尔可夫假设

# Types of Sequential Decision Processes
- MDP：观测足以描述状态，即 $s_t = o_t$
- POMPD：观测不足以描述状态
	- 扑克牌：只看观测自己的和牌
	- 医疗：只能观测部分健康数据
- Bandit：动作只影响当前的奖励，不影响将来的状态，因此没有延时奖励
	- 网页广告：对当前访问的用户投放的广告不会影响下一个访问的用户是否会点击广告