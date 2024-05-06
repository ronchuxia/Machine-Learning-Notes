# Logistic Regression
直接使用线性回归进行分类并不合适。可以使用 logistic 回归。
## Hypothesis
 logistic 回归中，假设 $h$ 与输入 $x$ 满足关系：
$$h_\theta(x) = g(\theta^Tx) = \frac{1}{1 + e^{- \theta^Tx}}$$
其中，$g(z)$ 称为 logistic function 或 sigmoid function：
$$g(z) = \frac{1}{1 + e^{-z}}$$
我们选择 $g(z)$ 是因为它是指数族分布中的伯努利分布的均值。

若 $h_\theta(x) > 0.5$，预测 $y = 1$；若 $h_\theta(x) \leq 0.5$，预测 $y = 0$。
**决策边界**：$\theta^T x = 0$。

## Probabilistic Interpretation
 logistic 回归基于以下假设：
$$
\begin{align}
p(y^{(i)} = 1|x^{(i)};\theta) & = h_\theta(x^{(i)})\\
p(y^{(i)} = 0|x^{(i)};\theta) & = 1 - h_\theta(x^{(i)})
\end{align}
$$
简写为：
$$p(y^{(i)}|x^{(i)};\theta) = (h_\theta(x^{(i)}))^{y^{(i)}}(1 - h_\theta(x^{(i)}))^{1 - y^{(i)}}$$
即对于给定的点 $x^{(i)}$，$y^{(i)}$ 服从伯努利分布。

对于参数 $\theta$，我们做最大似然估计，最大化参数 $\theta$ 的对数似然函数 $l(\theta)$：
$$l(\theta) = \sum_{i = 1}^m y^{(i)} \log h_\theta(x^{(i)}) + (1 - y^{(i)})\log(1 - h_\theta(x^{(i)}))$$
可以证明，$l(\theta)$ 的海森矩阵 $H$ 是一个半负定矩阵 ( $z^THz \leq 0$ )，因此 $l(\theta)$ 是一个凹函数，有唯一的局部最大值，即全局最大值。

## Gradient Ascend
使用梯度上升 (gradient ascend) 算法最大化最大化参数 $\theta$ 的对数似然函数 $l(\theta)$：
$$
\begin{align}
\theta_j & := \theta_j + \alpha \frac{\partial}{\partial \theta_j} l(\theta)\\
& := \theta_j + \alpha \sum_{i = 1}^m (y^{(i)} - h_\theta(x^{(i)}))x^{(i)}_j
\end{align}
$$
上述过程中用到了 $g(z)$ 的性质：
$$g'(z) = g(z)(1-g(z))$$
可以看到，参数的更新方法和线性回归是一样的。

# Newton's Method
可以使用牛顿法解方程。对于方程 $f(\theta) = 0$：
$$\theta := \theta - \frac{f(\theta)}{f'(\theta)}$$

最大化 $l(\theta)$ 即解方程 $l'(\theta) = 0$：
$$\theta := \theta - \frac{l'(\theta)}{l''(theta)}$$
如果 $\theta$ 是向量：
$$\theta := \theta - H^{-1} \nabla_\theta l(\theta)$$
其中，$H$ 是海森矩阵：
$$H_{ij} = \frac{\partial^2 l(\theta)}{\partial \theta_i \partial \theta_j}$$

牛顿法根据损失函数的**曲率**（海森矩阵）调整更新的步长：曲率越大，更新的步长越小；曲率越小，更新的步长越大。

牛顿法比梯度下降**收敛更快**，但**每一步所花时间更长**，因为需要计算矩阵的逆。因此，**在参数数量不多时可以使用牛顿法**。

# Perceptron
感知机中，假设 $h$ 与输入 $x$ 满足关系：
$$h_\theta(x) = g(\theta^Tx) = \begin{cases}
1 & if \ \theta^Tx \geq 0\\
0 & if \ \theta^Tx < 0
\end{cases}
$$
其中，$g(z)$ 可以看作是 a "hard" version of logistic function

感知机以一个超平面作为决策边界 (decision boundary)，参数 $\theta$ 为这个超平面的法向量。感知机根据以下算法更新参数 $\theta$：
$$\theta_j := \theta_j + \alpha \sum_{i = 1}^m (y^{(i)} - h_\theta(x^{(i)})) x^{(i)}$$
使得：
$$\begin{cases}
\theta \rightarrow x & if \ y^{(i)} = 1\\
\theta \nrightarrow x & if \ y^{(i)} = 0
\end{cases}
$$

可以解释为学习模板 $\theta$，用于匹配与模板 $\theta$ 相似的样本 $x$。
[[2 Linear Classification]]

感知机没有概率解释。