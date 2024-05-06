Notations:
- $y \in \{-1, 1\}$
- $h_{w, b}(x) = g(w^T x + b)$
- $g(z) = \begin{cases} 1, \ z \geq 0\\ -1, z < 0 \end{cases}$

# Functional and Geometric Margins
使用 margin 描述分类的准确度和可信度。

Functional Margin :
$$\hat\gamma^{(i)} = y^{(i)} (w^Tx + b)$$
$$\hat\gamma = \min_{i = 1, 2, \cdots, m} \hat{\gamma}^{(i)}$$
对于线性分类器，functional margin 并不是一个很好的可信度度量值。因为**当 w 和 b 等比例放大时，决策边界没有变，而 functional margin 却等比例放大了**。

Geometric Margin :
$$\gamma^{(i)} = y^{(i)} ((\frac{w}{|w|})^T x^{(i)} + \frac{b}{|w|})$$
$$\gamma = \min_{i = 1, 2, \cdots, m} \gamma^{(i)}$$
Geometric margin 描述了样本点到 separating hyperplane 的距离。且**当 w 和 b 的等比例放大时，geometric margin 不变**。

# The Optimal Margin Classifier
对于线性可分的训练集，寻找使得 geometric margin 最大的 separating hyperplane 可以形式化为以下优化问题：
$$\begin{align}
\max_{\gamma, w, b} & \ \gamma\\
\mathrm{s.t.} & \ y^{(i)} (w^Tx^{(i)} + b) \geq \gamma\\
& \ ||w|| = 1
\end{align}$$
约束 $||w|| = 1$ 非凸。为了摆脱该约束，将优化问题改写为：
$$\begin{align}
\max_{\hat\gamma, w, b} & \ \frac{\hat\gamma}{||w||}\\
\mathrm{s.t.} & \ y^{(i)} (w^Tx^{(i)} + b) \geq \hat\gamma\\
\end{align}$$
目标函数 $\frac{\hat\gamma}{||w||}$ 非凸。添加约束 $\hat\gamma = 1$ （这可以通过等比例缩放 w 和 b 做到）并变最大化 $\frac{1}{||w||}$ 为最小化 $||w||^2$ 可以将目标函数变为凸函数。优化问题改写为：
$$\begin{align}
\min_{w, b} & \ \frac{1}{2} ||w||^2\\
\mathrm{s.t.} & \ y^{(i)} (w^Tx^{(i)} + b) \geq 1\\
\end{align}$$
这是一个 quadratic programming 问题。将其求解即为 optimal margin classifier。

Support Vector Machine Loss : [[2 Linear Classification]]









