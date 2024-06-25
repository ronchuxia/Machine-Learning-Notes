# The K-Means Clustering Algorithm
1. 随机初始化（或随机选择 k 个数据点作为）聚类中心 $\mu_1, \mu_2, \cdots, \mu_k \in \mathbb{R}^n$
2. 对 $i = 1, 2, \cdots, m$
$$c^{(i)} := \arg\min_j ||x^{(i)} - \mu_j||$$
3. 对 $j = 1, 2, \cdots, k$
$$\mu_j := \frac{\sum_{i = 1}^m 1\{c^{(i)} = j\}x^{(i)}}{\sum_{i = 1}^m 1\{c^{(i)} = j\}}$$
4. 重复 2，3 直到收敛

Distortion function：
$$J(c, \mu) = \sum_{i = 1}^m ||x^{(i)} - \mu_{c^{(i)}}||$$
K-means 算法可以看作是对 $J(c, \mu)$ 做 coordinate descent：先固定 $\mu$ 并优化 $c$，然后固定 $c$ 并优化 $\mu$。因为 $J \geq 0$，且在 k-means 算法中 $J$ 单调递减，所以 $J$ 一定会收敛（这一般意味着 $c$ 和 $\mu$ 也会收敛。理论上 $c$ 和 $\mu$ 有可能在几个最优值之间震荡，但这种情况很少发生）。

因为 $J$ 不是凸函数，所以 k-means 算法会**收敛到局部最小值**。局部最小值的聚类效果一般已经足够。

# Mixtures of Gaussians and the EM Algorithm
高斯混合模型属于**密度估计**问题。给定一组观测数据，密度估计的任务是找到一个能够描述这些数据分布的函数。

高斯混合模型基于以下假设：
$$z \sim \mathrm{Categorical}(\phi)$$
$$x | z = j \sim \mathcal{N}(\mu_j, \Sigma_j)$$
即先生成服从分类分布的隐变量 $z$，再根据 $z$ 的值生成服从高斯分布的变量 $x$。

MLE 最大化边缘概率的对数似然函数：
$$\begin{align}
l(\phi, \mu, \Sigma) & = \sum_{i = 1}^m \log p(x^{(i)}; \phi, \mu, \Sigma)\\
& = \sum_{i = 1}^m \log \sum_{z^{(i)} = 1}^k p(x^{(i)}|z^{(i)};\mu, \Sigma) p(z^{(i)}; \phi) 
\end{align}$$
十分复杂，无法通过让导数等于 0 求得其闭形解。

**EM 算法**可以优化高斯混合模型：
1. E-step：对 $i = 1, 2, \cdots, m$，$j = 1, 2, \cdots, k$，计算条件概率
$$w^{(i)}_j := p(z^{(i)} = j | x^{(i)}; \phi, \mu, \Sigma)$$
2. M-step：更新参数
$$\phi_j := \frac{1}{m} \sum_{i = 1}^m w_j^{(i)}$$
$$\mu_j := \frac{\sum_{i = 1}^m w^{(i)}_j x^{(i)}}{\sum_{i = 1}^m w^{(i)}_j}$$
$$\Sigma_j := \frac{\sum_{i = 1}^m w^{(i)}_j (x^{(i)} - \mu_j)(x^{(i)} - \mu_j)^T}{\sum_{i = 1}^m w^{(i)}_j}$$
3. 重复 1，2 直到收敛

EM 算法的过程与 k-means 类似，但 EM 算法对类别做 soft guess（概率分布），而 k-means 对类别做 hard guess（唯一的值）。

EM 算法会收敛到局部最大值。