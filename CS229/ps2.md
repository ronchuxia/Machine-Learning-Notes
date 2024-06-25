# Logistic Regression: Training Stability
1. 在数据集 A 上快速收敛，在数据集 B 上无法收敛
2. $y \in \{1, -1\}$ 的 logistic 回归基于以下假设：
$$p(y^{(i)} = 1|x^{(i)};\theta) = \frac{1}{1 + e^{- \theta^T x^{(i)}}}$$
$$p(y^{(i)} = -1|x^{(i)};\theta) = \frac{1}{1 + e^{\theta^T x^{(i)}}}$$
即：
$$p(y^{(i)}|x^{(i)};\theta) = \frac{1}{1 + e^{- y^{(i)} \theta^T x^{(i)}}}$$
对数似然函数：
$$l(\theta) = \sum_{i = 1}^m \log(1 + e^{- y^{(i)} \theta^T x^{(i)}})$$
当 $\theta^T x \leq 0$ 时，预测 $y = -1$；当 $\theta^T x > 0$ 时，预测 $y = 1$。

数据集 B 线性可分，则在分割超平面附近：
$$\forall \ i \in \{1, 2, \cdots m\}, \ - y^{(i)} \theta^T x^{(i)} < 0$$
若 $\theta \leftarrow k \theta$，$k > 1$，则 $l(\theta)$ 将减小。即优化问题无解。

数据集 A 线性不可分，则在分割超平面附近：
$$\exists \ i \in \{1, 2, \cdots m\}, \ - y^{(i)} \theta^T x^{(i)} > 0$$
若 $\theta \leftarrow k \theta$，$k > 1$，则当 $k$ 足够大时，$l(\theta)$ 将增大。即优化问题有解。

可以从两个角度解释 $y \in \{1, -1\}$ 的 logistic 回归和标准的 logistic 回归的区别：
- $y \in \{1, -1\}$ 的 logistic 回归不再假设输出 $y$ 服从伯努利分布，因此也不再属于广义线性模型，因此其负对数似然函数不再是凸函数
- $y \in \{1, -1\}$ 的 logistic 回归相当于对所有点的 functional margin 进行优化。对于线性可分的情况，等比例地放大权重，functional margin 也会等比例地放大

3. 添加学习率衰减、添加正则化项、添加高斯噪声（破坏数据集的线性可分性）可以让 $y \in \{1, -1\}$ 的 logistic 回归在线性可分的数据集上收敛。

4. SVM 可以在线性可分的数据集上收敛，可以从两个角度解释：
	- SVM 最小化 geometric margin，而不是 functional margin
	- 添加了正则化的 SVM 的优化问题等价于最小化成本函数
$$J(\theta) = \sum_{i = 1}^m \max(0, 1 - y^{(i)} \theta^T x^{(i)}) + C||w||^2$$
其中第一项为 **hinge loss**：
$$L(\theta) = \sum_{i = 1}^m \max(0, 1 - y^{(i)}\theta^T x^{(i)})$$
当 functional margin < 1 时，会产生 hinge loss。

数据集 B 线性可分，则在分割超平面附近：
$$\exists \ \theta, \forall \ i \in \{1, 2, \cdots m\}, \ y^{(i)} \theta^T x^{(i)} > 1$$
若 $\theta \leftarrow k \theta$，$k > 1$，则 $J(\theta)$ 将增加。即优化问题有解。

# Model Calibration
1. Logistic regression 的最优解满足：
$$\frac{\partial}{\partial \theta} l(\theta) = \sum_{i = 1}^m (y^{(i)} - h_\theta(x^{(i)}))x^{(i)} = 0$$
即：
$$\sum_{i = 1}^m y^{(i)} x^{(i)} = \sum_{i = 1}^m h_\theta(x^{(i)})x^{(i)}$$
只考虑其中的 bias term，得：
$$\sum_{i = 1}^m y^{(i)} = \sum_{i = 1}^m h_\theta(x^{(i)})$$
这与要证的性质等价。
2. 没看懂。
3. 正则项的加入使得模型不再是 perfectly calibrated。

# Bayesian Interpretation of Regularization
1. **边缘独立与独立等价**：
$$p(A | B) = p(A) \iff p(A, B) = p(A) p(B)$$
因此：
$$\begin{align}
\theta_{\text{MAP}} & = \arg\max_\theta p(\theta|x, y)\\
& = \arg\max_{\theta} \frac{p(\theta, x, y)}{p(x, y)}\\
& = \arg\max_{\theta} p(y | x, \theta) p(\theta) \frac{p(x)}{p(x, y)}\\
& = \arg\max_{\theta} p(y | x, \theta) p(\theta)
\end{align}$$

2. 证明使用高斯先验 $\theta \sim \mathcal{N}(0, \eta^2I)$ 的 MAP 与使用 L2 正则化的 MLE 等价：
$$\begin{align}
\theta_{MAP} & = \arg\max_\theta p(y|x,\theta)p(\theta)\\
& = \arg\max_\theta p(y|x,\theta) \frac{1}{\sqrt{2\pi} (\eta \sqrt{n})} \exp(-\frac{1}{2} \theta^T \frac{I}{\eta^2} \theta)\\
& = \arg\max_\theta \log p(y|x,\theta) - \frac{1}{2\eta^2} ||\theta||^2\\
& = \arg\min_\theta - \log p(y|x,\theta) + \frac{1}{2\eta^2} ||\theta||^2
\end{align}$$

3. 使用 L2 正则化 / 高斯先验 $\theta \sim \mathcal{N}(0, \eta^2I)$ 的线性回归称为**岭回归**。对岭回归做 MAP：
$$\begin{align}
\theta_{\text{MAP}} & = \arg\min_\theta - \log p(Y|X,\theta) + \frac{1}{2\eta^2} ||\theta||^2\\
& = \arg\min_\theta - \sum_{i = 1}^m \log p(y^{(i)} | x^{(i)}, \theta) + \frac{1}{2\eta^2} ||\theta||^2\\
& = \arg\min_\theta \sum_{i = 1}^m \frac{1}{2 \sigma^2} (y^{(i)} - \theta^T x^{(i)})^2 + \frac{1}{2\eta^2} ||\theta||^2\\
& = \arg\min_\theta \sum_{i = 1}^m (y^{(i)} - \theta^T x^{(i)})^2 + \frac{\sigma^2}{\eta^2} ||\theta||^2
\end{align}$$

岭回归的成本函数：
$$J(\theta) = \sum_{i = 1}^m (y^{(i)} - \theta^T x^{(i)})^2 + \frac{\sigma^2}{\eta^2} ||\theta||^2$$

岭回归的成本函数是凸函数，有全局最小值：
$$\begin{align}
\nabla_\theta J(\theta) & = \sum_{i = 1}^m (y^{(i)} - \theta^T x^{(i)}) (- x^{(i)}) + \frac{\sigma^2}{\eta^2} \theta\\
& = - \sum_{i = 1}^m y^{(i)} x^{(i)} + \sum_{i = 1}^m \theta^T x^{(i)} \cdot x^{(i)} + \frac{\sigma^2}{\eta^2} \theta\\
& = - \sum_{i = 1}^m y^{(i)} x^{(i)} + \sum_{i = 1}^m x^{(i)} \cdot \theta^T x^{(i)} + \frac{\sigma^2}{\eta^2} \theta\\
& = - \sum_{i = 1}^m y^{(i)} x^{(i)} + \sum_{i = 1}^m x^{(i)} (x^{(i)})^T \theta + \frac{\sigma^2}{\eta^2} \theta\\
& = - X^T Y + X^T X \theta + \frac{\sigma^2}{\eta^2} \theta\\
& = 0
\end{align}$$

最优解为：
$$\theta = (\frac{\sigma^2}{\eta^2} + X^T X)^{-1} X^T Y$$

4. 证明使用 Laplace 先验 $\theta \sim \mathcal{L}(0, bI)$ 的 MAP 与使用 L1 正则化的 MLE 等价：
$$p(\theta) = \frac{1}{(2b)^n} \exp(- \frac{||\theta||_1}{b})$$
$$\begin{align}
\theta_{\text{MAP}} & = \arg\max_\theta p(y|x,\theta)p(\theta)\\
& = \arg\max_\theta p(y|x,\theta) \frac{1}{(2b)^n} \exp(- \frac{||\theta||_1}{b})\\
& = \arg\max_\theta \log p(y|x,\theta) - \frac{1}{b} ||\theta||_1\\
& = \arg\min_\theta - \log p(y|x,\theta) + \frac{1}{b} ||\theta||_1
\end{align}$$

5. 使用 L1 正则化 / Laplace 先验 $\theta \sim \mathcal{L}(0, bI)$ 的线性回归称为 Lasso 回归。对 Lasso 回归做 MAP：
$$\begin{align}
\theta_{\text{MAP}} & = \arg\min_\theta - \log p(y|x,\theta) + \frac{1}{b} ||\theta||_1\\
& = \arg\min_\theta \sum_{i = 1}^m (y^{(i)} - \theta^T x^{(i)})^2 + \frac{2\sigma^2}{b} ||\theta||_1
\end{align}$$

Lasso 回归的成本函数：
$$J(\theta) = \sum_{i = 1}^m (y^{(i)} - \theta^T x^{(i)})^2 + \frac{2\sigma^2}{b} ||\theta||_1$$

Lasso 回归没有闭形解。可以使用梯度下降法优化成本函数。

# Constructing Kernels
a, c, e, f, g, h 是 kernel。

# Kernelizing the Perceptron
将 kernel method 用于 perceptron。
1. 参数 $\theta$ 的更新规则：
$$\theta^{(t+1)} := \theta^{(t)} + \alpha \sum_{i = 1}^m (y^{(i)} - h_\theta(\phi(x^{(i)}))) \phi(x^{(i)})$$
3. 当 $\theta^{(0)} = 0$ 时，由参数更新规则，$\theta^{(t)} = \sum_{i = 1}^m \beta_i^{(t)} \phi(x^{(i)})$ 是特征向量 $\phi(x^{(i)})$ 的线性组合。因此，无论 $\phi(x^{(i)})$ 的维度是多少，都可以用一个 m 维向量 $\theta^{'(t)}$ 表示 $\theta^{(t)}$：
$$\theta^{'(t)} = \begin{bmatrix} 
\beta_1^{(t)}\\
\beta_2^{(t)}\\
\vdots\\
\beta_m^{(t)}
\end{bmatrix}$$
2. 预测时：
$$\begin{align}
h_\theta(x) & = g(\theta^T \phi(x))\\
& = g(\sum_{i = 1}^m \beta_i^{(t)} (\phi(x^{(i)}))^T \phi(x))\\
& = g(\sum_{i = 1}^m \beta_i^{(t)} K(x^{(i)}, x))
\end{align}$$
3. 参数 $\theta^{'}$ 的更新规则：
$$\beta^{(t+1)}_i := \beta^{(t)}_i + \alpha (y^{(i)} - h_\theta(\phi(x^{(i)})))$$

# Spam Classification
实现 Naive Bayes (Multinomial Event Model)。
















