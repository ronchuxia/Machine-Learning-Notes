# Supervised Learning
Notations:
- $x = (x_1, x_2, \cdots, x_n)$ : input variables / features
	- $x_i$ : ith input variable
	- $n$ : # input variables (not counting $x_0$)
- $y$ : output variable / target
- $(x, y)$ : training example
	- $(x^{(i)}, y^{(i)})$ : ith training example
- $\{(x^{(i)}, y^{(i)}): i = 1, 2, \cdots, m\}$ : training set
- $m$ : # training examples

Supervised Learning: 给定训练集 $\{(x, y)\}$ ，学习一个假设 (hypothesis) $h$，输入 $x$，预测 $y$
- $y$ 是连续的：回归
- $y$ 是离散的：分类

# Linear Regression
线性回归中，假设 $h$ 是输入 $x$ 的线性函数（在线性代数中，更严谨的说法是仿射函数 (affine function)）：
$$h_\theta(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + \cdots + \theta_nx_n$$
令 $x_0 = 1$，则可以写成：
$$h_\theta(x) = \sum^{n}_{i = 0} \theta_ix_i = \theta^Tx$$

用于衡量参数 $\theta$ 效果的成本函数 (cost function)：
$$J(\theta) = \frac{1}{2} \sum^{m}_{i = 0} (h_\theta(x^{(i)}) - y^{(i)})^2$$

## Gradient Descent
为了最小化成本函数 $J(\theta)$，可以使用梯度下降算法：
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j}J(\theta)$$
- 沿梯度的反方向（下降最快的方向）更新所有参数。
- 每次更新的步长不一样，梯度越大，更新的步长也越大。
- $\alpha$ : learning rate，用于调整更新的步长

### Batch Gradient Descent
参数的更新方向为所有 traning example 的 error 的梯度的反方向：
$$\frac{\partial}{\partial \theta_j}J(\theta) = \sum^{n}_{i = 0} (h_\theta(x^{(i)}) - y^{(i)})x^{(i)}_j$$
$$\theta_j := \theta_j - \alpha \sum^{n}_{i = 0} (h_\theta(x^{(i)}) - y^{(i)})x^{(i)}_j$$

线性回归的成本函数 $J(\theta)$ 有唯一的局部最小值，即全局最小值，因此 batch gradient descent 最终会收敛到全局最小值。

### Stochastic Gradient Descent
参数的更新方向为一个 traning example 的 error 的梯度的反方向：
$$\theta_j := \theta_j - \alpha (h_\theta(x^{(i)}) - y^{(i)})x^{(i)}_j$$

每遇到一个 traning example，就更新一次参数。由于每一步更新都只考虑了一个 training example，stochastic gradient descent 并不能保证收敛到最小值。

## Normal Equation
线性回归的成本函数 $J(\theta)$ 有唯一的局部最小值，因此参数有唯一解，可以利用线性代数的知识直接解出参数值。

Notations:
对于函数 $f: \mathbb{R}^{m \times n} \mapsto \mathbb{R}$，定义 $f$ 的导数：
$$\nabla_Af(A) = 
\begin{bmatrix}
\frac{\partial f}{\partial A_{11}} & \cdots & \frac{\partial f}{\partial A_{1n}}\\
\vdots & \ddots & \vdots\\
\frac{\partial f}{\partial A_{m1}} & \cdots & \frac{\partial f}{\partial A_{mn}}\\
\end{bmatrix}$$

导数的性质：
1. $\nabla_A trAB = \nabla_A trBA = B^T$
	- 要求 $AB$ 是方阵
	- 注：$trAB = trBA$
2. $\nabla_{A^T} f(A) = (\nabla_A f(A))^T$
3. $\nabla_A trABA^TC = CAB + C^TAB^T$
	- 要求 $B$ 是方阵
	- 注 1：$(fg)' = f'g + fg'$
	- 注 2：由性质 2 和性质 3，$\nabla_{A^T} trABA^TC = B^TA^TC^T + BA^TC$
	- 注 3：$\nabla_A tr ABA^T = AB + AB^T$，$\nabla_{A^T} tr ABA^T = B^TA^T + BA^T$
1. $\nabla_A |A| = |A|(A^{-1})^T$
	- 注 1：$A^{-1} = \frac{1}{|A|} (A')^T$
	- 注 2：$|A| = \sum_j A_{ij}A'_{ij}$

将成本函数写成向量形式：
$$J(\theta) = \frac{1}{2} (X \theta - Y)^T (X \theta - Y)$$
线性回归的成本函数 $J(\theta)$ 的最小值在其导数等于0的地方取到：
$$
\begin{align}
\nabla_\theta J(\theta) & = \nabla_\theta \frac{1}{2} (X \theta - Y)^T (X \theta - Y)\\
& = \frac{1}{2} \nabla_\theta (\theta^T X^T X \theta - Y^T X \theta - \theta^T X^T Y + Y^T Y)\\
& = \frac{1}{2} \nabla_\theta tr(\theta^T X^T X \theta - Y^T X \theta - \theta^T X^T Y + Y^T Y)\\
& = \frac{1}{2} \nabla_\theta tr(\theta^T X^T X \theta - 2Y^T X \theta)\\
& = \frac{1}{2} (\nabla_{(\theta^T)^T} tr(\theta^T X^T X \theta) - 2\nabla_\theta tr(Y^T X \theta))\\
& = \frac{1}{2} (X^TX\theta + X^TX\theta - 2X^TY)\\
& = X^TX\theta - X^TY\\
& = 0
\end{align}
$$
将下式称为正规方程 (normal equation)：
$$X^TX\theta - X^TY = 0$$
其解为：
$$\theta = (X^TX)^{-1}X^TY$$

也可以将 $J(\theta)$ 写成：
$$J(\theta) = \frac{1}{2} ||X\theta - Y||^2$$
然后用矩阵求导法求最优解 $\theta$。

# Locally Weighted Linear Regression
对于一些数据，使用线性假设并不能很好的拟合，而使用高次多项式假设又容易过拟合，此时可以使用局部加权线性回归。

局部加权线性回归的成本函数：
$$J(\theta) = \frac{1}{2} \sum_{i = 1}^m w^{(i)}(h_\theta(x^{(i)}) - y^{(i)})^2$$
其中，$w^{(i)}$ 是非负的权重，常用：
$$w^{(i)} = \exp(- \frac{(x^{(i)} - x)^2}{2 \tau^2})$$
如果 $x$ 是向量，$w^{(i)}$ 可以一般化为：
$$w^{(i)} = \exp(- \frac{(x^{(i)} - x)^T(x^{(i)} - x)}{2 \tau^2})$$
或：
$$w^{(i)} = \exp(- \frac{(x^{(i)} - x)^T \Sigma^{-1} (x^{(i)} - x)}{2})$$
可以看到，权重的值和我们想要预测其输出的那个点 $x$ 的位置有关。数据点与预测点的距离越小，其 error 的权重越大。

对于给定的点 $x$，权重函数即高斯函数，$\tau$ 称为带宽 (bandwidth) 系数，用于调节权重随距离增大而衰减的速度。

对于低维数据可以使用局部加权线性回归。

局部加权线性回归是 **non-parametric 算法**，它的参数量是不固定的。局部加权线性回归需要保存所有的训练集数据，并在预测时使用。

# Probabilistic Interpretation of Linear Regression
线性回归基于以下假设：
1. 输入与输出满足关系：$y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)}$
2. $\epsilon^{(i)}$ 独立同分布，且服从高斯分布：$\epsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$

由上述假设可得，对于给定的点 $x^{(i)}$，$y^{(i)}$ 服从高斯分布：
$$y^{(i)} | x^{(i)}; \theta \sim \mathcal{N}(
\theta^T x^{(i)}, \sigma^2
)$$
即：
$$p(y^{(i)} | x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2})$$

对于参数 $\theta$，我们做最大似然估计 (maximum likelihood estimation)，最大化参数取 $\theta$ 的似然函数 (likelihood) $L(\theta)$，即数据集中的输出取 $y^{(i)}$ 的概率：
$$L(\theta) = \prod_{i = 1}^m p(y^{(i)} | x^{(i)}; \theta)$$
对似然函数 $L(\theta)$ 取对数得对数似然函数  (log likelihood) $l(\theta)$：
$$l(\theta) = m\log \frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{\sigma^2} \frac{1}{2} \sum_{i = 1}^m (y^{(i)} - \theta^T x^{(i)})^2$$
即最小化：
$$J(\theta) = \frac{1}{2}(y^{(i)} - \theta^T x^{(i)})^2$$

基于对于数据的统计假设，最小均方误差回归就相当于对参数 $\theta$ 进行最大似然估计。
