# The Bias-Variance Tradeoff
Bias 表示模型的假设空间的合理性。
- 如果 bias 很大，则无论在什么样的数据集上训练都无法获得较好的结果
- 如果模型过于简单，则 bias 较大，容易欠拟合

减小 bias 的方法：
- 选择更合理的假设空间 
- 扩大假设空间，但有可能导致 variance 增加

Variance 表示模型对噪声的敏感性
- 如果 variance 大，则在不同的数据集上训练得到的模型会有很大的差别，模型的泛化性差
- 如果模型过于复杂，则 variance 较大，容易过拟合

减小 variance 的方法：
- 增加训练集大小
- 添加正则化，但有可能导致 bias 增加。

传统经验认为，随着模型复杂度（参数个数）的增加，bias 会减小，variance 会增加。存在一个平衡点，使得两者的和最小。寻找这个平衡点，即 bias-variance tradeoff。也即模型选择。

假设：
1. 训练数据和测试数据都采样自相同的数据分布 $\mathcal{D}$
2. 训练数据和测试数据都独立采样

Notations：
- $g$ ：实际的最优假设
- $\mathcal{H}$ ：模型的假设空间
- $h^*$ ：模型的假设空间中的最优假设
- $S$ ：训练集，采样自数据分布 $\mathcal{D}$
- $h_S$ ：在训练集 $S$ 上训练得到的假设

**经验风险 (empirical error / training error)**：假设在训练集 $S$ 上的平均风险
**泛化风险 (generalization error / test error)**：假设在数据分布 $\mathcal{D}$ 上的期望风险

泛化风险可以分解为**近似误差**和**估计误差**：
$$\text{Generalization Error} = \text{Irreducible Error} + \text{Approximation Error} + \text{Estimation Error}$$
- Irreducible Error：$\varepsilon(g)$，实际的最优假设的差距
- Approximation Error：$\varepsilon(h^*) - \varepsilon(g)$，模型的假设空间中的最优假设与实际的最优假设的差距
- Estimation Error：$\varepsilon(h_S) - \varepsilon(h^*)$，训练得到的假设与模型的假设空间中的最优假设的差距

泛化风险的期望可以分解为 bias 和 variance：
对于回归问题，训练数据和测试数据都满足 $y = g(x) + \xi$，$\xi \sim \mathcal{N}(0, \sigma^2)$。则泛化风险的期望为：
$$\begin{align}
E_{S, x, \xi}[\varepsilon(h_S)] & = E_{S, x, \xi}[(y - h_S(x))^2]\\
& = E_{S, x, \xi}[(\xi + (h^*(x) - h_S(x)))^2]\\
& = E_{\xi}[\xi^2] + E_{S, x}[(h^*(x) - h_S(x))^2]\\
& = \sigma^2 + E_{S, x}[(h^*(x) - h_S(x))^2]\\
& = \sigma^2 + E_{S, x}[((h^*(x) - h_{avg}(x)) + (h_{avg}(x) - h_S(x)))^2]\\
& = \sigma^2 + E_x[E_S[(h^*(x) - h_{avg}(x))^2]] + E_x[E_S[(h_{avg}(x) - h_S(x))^2]]\\
& = \sigma^2 + E_x[(h^*(x) - h_{avg}(x))^2] + E_x[Var_S[h_S(x)]]
\end{align}$$
- $\sigma^2$ : irreducible error
- $E_x[(h^*(x) - h_{avg}(x))^2]$ : bias
- $E_x[Var_S[h_S(x)]]$ : variance

对于分类问题，bias 和 variance 则不太好定义。

# The Double Descent Phenomenon
随着深度学习的发展，人们发现，随着模型复杂度（参数个数）的增加，泛化损失会先减小，然后增加，然后再减小。当参数个数增加到模型刚好能拟合所有训练数据（参数个数约等于数据个数）后，继续增加模型复杂度（参数个数），模型的泛化损失会开始第二次减小，且能减小到比之前的最小值更小。

![](https://images.ctfassets.net/kftzwdyauwt9/9b1defbc-a847-4a59-131ef795a083/9e9efe8767bdb54428099a3ab543b0e2/modeldd.svg?w=1920&q=90&fm=webp)

反过来，随着数据个数的增加，泛化损失也会先减小，然后增加，然后再减小。当数据个数增加到模型刚好不能拟合所有训练数据（数据个数约等于参数个数）后，继续增加数据个数，模型的泛化损失会开始第二次减小。

这告诉我们可以尝试 over-parametrized models，通过显著增加参数量的方式降低泛化损失。

这一现象可能是由算法特性造成的。比如梯度下降算法可能包含了隐式的正则化。

另外，这一现象也可能是由模型复杂度的衡量标准选取不当造成的。

# Sample Complexity Bounds
## Preliminaries
引理 1：Union bound
$$P(A_1 \cup \cdots \cup A_k) \leq P(A_1) + \cdots + P(A_k)$$

引理 2：Hoeffding inequality / Chernoff bound
- 让 $Z_1, Z_2, \cdots, Z_m$ 为 $m$ 个独立同分布的随机变量，$Z_i \sim \mathrm{Bernoulli}(\phi)$，$i \in \{1, 2, \cdots, m\}$
- 让 $\hat \phi = (1 / m) \sum_{i = 1}^m Z_i$ 为参数 $\phi$ 的估计量
- 让 $\gamma > 0$ 为常数
$$P(|\phi − \hat \phi| > \gamma) \leq 2 \exp(−2\gamma^2m)$$
Chernoff bound 可以扩展到非伯努利分布的情况。

为了简化，下面讨论二分类问题。
二分类问题的经验风险：
$$\hat \varepsilon(h) = \frac{1}{m} \sum_{i = 1}^m 1\{h(x^{(i)}) \neq y^{(i)}\}$$
二分类问题的泛化风险：
$$\varepsilon(h) = P_{(x,y) \sim \mathcal{D}}(h(x) \neq y)$$

**ERM (empirical error minimization)** 算法最小化经验风险：
$$\hat h = \arg\min_{h \in \mathcal{H}} \hat\varepsilon(h)$$
Logistic 回归可以被视为一种一种 ERM 算法。

## The case of finite $\mathcal{H}$
如果假设空间 $\mathcal{H} = \{h_1, h_2, \cdots , h_k\}$ 由 $k$ 个假设组成。则对于其中的任意一个假设 $h_i$，让 $Z_j = 1\{h_i(x^{(j)}) \neq y^{(j)}\}$，则 $Z_j \sim \mathrm{Bernoulli}(\varepsilon(h_i))$ 为伯努利分布随机变量，且 $\hat\varepsilon(h_i) = \frac{1}{m} \sum_{j = 1}^m Z_j$ 是参数 $\varepsilon(h_i)$ 的估计量。

由 Hoeffding inequality：
$$P(|\varepsilon(h_i) - \hat\varepsilon(h_i)| < \gamma) \leq 2 \exp(-2 \gamma^2 m)$$

由 Union bound：
$$P(\exists h_i \in \mathcal{H}. |\varepsilon(h_i) − \hat\varepsilon(h_i)| > \gamma) \leq 2k \exp(-2 \gamma^2 m)$$

取反：
$$P(\forall h_i \in \mathcal{H}. |\varepsilon(h_i) − \hat\varepsilon(h_i)| \leq \gamma) \leq 1 - 2k \exp(-2 \gamma^2 m)$$

即 $\varepsilon(h)$ **一致收敛**到 $\hat\epsilon(h)$。

当：
$$m \geq \frac{1}{2\gamma^2} \log \frac{2k}{\delta}$$
时，有至少 $1 - \delta$ 的概率，$\forall h_i \in \mathcal{H}. |\varepsilon(h_i) − \hat\varepsilon(h_i)| \leq \gamma$。要让算法达到一定的性能所需的训练集大小称为算法的 **样本复杂度 (sample complexity)**。

此时：
$$\begin{align}
\varepsilon(\hat h) & \leq \hat\varepsilon(\hat h) + \gamma\\
& \leq \hat\varepsilon(h^*) + \gamma\\
& \leq \varepsilon(h^*) + 2\gamma
\end{align}$$
即 ERM 算法输出的假设 $\hat h$ 的泛化损失 $\varepsilon(\hat h)$ 最多比假设空间 $\mathcal{H}$ 所能达到的最小的泛化损失 $\varepsilon(h^*)$ 多 $2\gamma$。

## The case of infinite $\mathcal{H}$
对于集合 $S$，如果假设空间 $\mathcal{H}$ 能够实现集合 $S$ 的任意标注，则称 $\mathcal{H}$ 可以切分 $S$。

假设空间 $\mathcal{H}$ 的 **VC 维** $d = VC(\mathcal{H})$ 定义为 $\mathcal{H}$ 可以切分的最大集合的大小。

当：
$$m = O_{\gamma, \delta}(d)$$
时，有至少 $1 - \delta$ 的概率，$\forall h_i \in \mathcal{H}. |\varepsilon(h_i) − \hat\varepsilon(h_i)| \leq \gamma$。

TODO




