# Jensen's Inequality
如果 $f$ 是凸函数，$X$ 是随机变量，则：
$$E[f(X)] \geq f(EX)$$

特别地，如果 $f$ 是严格凸函数，则 $E[f(X)] = f(EX)$ 当且仅当随机变量 $X$ 是常数。

# The EM algorithm
当模型含有隐变量时，显式地求解使 $l(\theta)$ 最大化的参数 $\theta$ 可能是困难的。

为了最大化 $l(\theta)$，EM 算法：
1. E-step（estimation）：构建 $l(\theta)$ 的下界
2. M-step（maximization）：最大化该下界
3. 重复 1，2 直到收敛

对 $i = 1, 2, \cdots, m$，如果 $Q_i$ 是与隐变量 $z$ 有关的某个概率分布，则：
$$\begin{align}
\sum_i \log p(x^{(i)}; \theta) & = \sum_i \log \sum_{z^{(i)}} p(x^{(i)}, z^{(i)}; \theta)\\
& = \sum_i \log \sum_{z^{(i)}} Q(z^{(i)}) \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}\\
& = \sum_i \log E_{z^{(i)} \sim Q_i}[\frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}]\\
& \geq \sum_i E_{z^{(i)} \sim Q_i}[\log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}]\\
& = \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}
\end{align}$$
这给出了 $l(\theta)$ 的下界。

为了使 EM 算法中 $l(\theta)$ 单调递增，我们要求在当前 $\theta$ 处，下界与 $l(\theta)$ 的值相等。由 Jensen 不等式，为了取等号：
$$\frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})} = c$$
即：
$$\begin{align}
Q_i(z^{(i)}) & = \frac{p(x^{(i)}, z^{(i)}; \theta)}{\sum_z p(x^{(i)}, z; \theta)}\\
& = p(z^{(i)} | x^{(i)}; \theta)
\end{align}$$
据此，我们构建了 $l(\theta)$ 的一个下界。

完整的 EM 算法如下：
1. E-step：对 $i = 1, 2, \cdots, m$，计算条件概率
$$Q_i(z^{(i)}) := p(z^{(i)} | x^{(i)}; \theta)$$
2. M-step：更新参数
$$\theta := \arg\max_\theta \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}$$

因为在 EM 算法中 $l(\theta)$ 单调递增，所以 $l(\theta)$ 一定会收敛。因为 $l(\theta)$ 不是凹函数，所以 EM 算法会**收敛到局部最大值**。

定义：
$$J(Q, \theta) = \sum_i \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \theta)}{Q_i(z^{(i)})}$$
则 EM 算法可以看作是对于下界做 coordinate ascent，先固定 $\theta$ 并优化 $Q$，然后固定 $Q$ 并优化 $\theta$。

# Mixture of Gaussians Revisited
可以证明，高斯混合模型的参数更新规则可以由 EM 算法的参数更新规则得到。

注 1：$\nabla_{A^T} tr ABA^T = B^TA^T + BA^T$
注 2：$\nabla_B tr ABA^T = A^TA$
注 3：$\nabla_A |A| = |A|(A^{-1})^T$

对于 $w_j^{(i)}$：
$$w_j^{(i)} = Q_i(z^{(i)} = j) = p(z^{(i)} = j | x^{(i)}; \phi, \mu, \Sigma)$$

则优化目标 $J(\phi, \mu, \Sigma)$ 可以写成：
$$\begin{align}
& \sum_{i = 1}^m \sum_{z^{(i)}} Q_i(z^{(i)}) \log \frac{p(x^{(i)}, z^{(i)}; \phi, \mu, \Sigma)}{Q_i(z^{(i)})}\\
= & \sum_{i = 1}^m \sum_{j = 1}^k Q_i(z^{(i)} = j) \log \frac{p(x^{(i)} | z^{(i)} = j; \mu, \Sigma) p(z^{(i)} = j; \phi)}{Q_i(z^{(i)} = j)}\\
= & \sum_{i = 1}^m \sum_{j = 1}^k w_j^{(i)} \log \frac{\frac{1}{(2\pi)^{n/2} |\Sigma_j|^{1/2}} \exp(- \frac{1}{2} (x^{(i)} - \mu_j)^T \Sigma_j^{-1} (x^{(i)} - \mu_j)) \phi_j}{w_j^{(i)}}
\end{align}$$

对于 $\phi_j$，与 $\phi_j$ 有关的项为：
$$\sum_{i = 1}^m \sum_{j = 1}^k w_j^{(i)} \log \phi_j$$
约束为：
$$\sum_{j = 1}^k \phi_j = 1$$
构造拉格朗日函数：
$$\mathcal{L}(\phi) = \sum_{i = 1}^m \sum_{j = 1}^k w_j^{(i)} \log \phi_j + \beta (\sum_{j = 1}^k \phi_j - 1)$$
令其导数为 0：
$$\frac{\partial}{\partial \phi_j} \mathcal{L}(\phi) = \sum_{i = 1}^m \frac{w^{(i)}_j}{\phi_j} + \beta$$
则：
$$\phi_j = \frac{\sum_{i = 1}^m w_j^{(i)}}{- \beta}$$
为了满足约束：
$$-\beta = \sum_{j = 1}^k \sum_{i = 1}^m w_j^{(i)} = m$$
即：
$$\phi_j = \frac{1}{m} \sum_{i = 1}^m w_j^{(i)}$$

对于 $\mu_j$，与 $\mu_j$ 有关的项为：
$$\sum_{i = 1}^m \sum_{j = 1}^k w_j^{(i)} (- \frac{1}{2}(x^{(i)} - \mu_j)^T \Sigma_j^{-1} (x^{(i)} - \mu_j))$$
令其导数为 0：
$$\begin{align}
& \nabla_{\mu_j} \sum_{i = 1}^m \sum_{j = 1}^k w_j^{(i)} (- \frac{1}{2}(x^{(i)} - \mu_j)^T \Sigma_j^{-1} (x^{(i)} - \mu_j)) \\
= & \frac{1}{2} \sum_{i = 1}^m w_j^{(i)} \nabla_{\mu_j} (2 \mu_j^T \Sigma_j^{-1} x^{(i)} - \mu_j^T \Sigma_j^{-1} \mu_j)\\
= & \sum_{i = 1}^m w_j^{(i)} (\Sigma_j^{-1} x^{(i)} - \Sigma_j^{-1} \mu_j)\\
= & 0
\end{align}$$
则：
$$\mu_j = \frac{\sum_{i = 1}^m w^{(i)}_j x^{(i)}}{\sum_{i = 1}^m w^{(i)}_j}$$

对于 $\Sigma_j$，与 $\Sigma_j$ 有关的项为：
$$\sum_{i = 1}^m \sum_{j = 1}^k w_j^{(i)} (- \frac{1}{2} \log |\Sigma_j|) + w_j^{(i)} (- \frac{1}{2}(x^{(i)} - \mu_j)^T \Sigma_j^{-1} (x^{(i)} - \mu_j))$$
令其导数为 0：
$$\begin{align}
& \nabla_{\Sigma_j^{-1}} \sum_{i = 1}^m \sum_{j = 1}^k - w_j^{(i)} \log |\Sigma_j^{-1}| + w_j^{(i)} (x^{(i)} - \mu_j)^T \Sigma_j^{-1} (x^{(i)} - \mu_j)\\
= & \sum_{i = 1}^m - w_j^{(i)} \Sigma_j + w_j^{(i)} (x^{(i)} - \mu_j) (x^{(i)} - \mu_j)^T\\
= & 0
\end{align}$$
则：
$$\Sigma_j = \frac{\sum_{i = 1}^m w_j^{(i)} (x^{(i)} - \mu_j) (x^{(i)} - \mu_j)^T}{\sum_{i = 1}^m w_j^{(i)}}$$






