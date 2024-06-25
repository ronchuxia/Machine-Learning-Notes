Notations:
- $y \in \{-1, 1\}$
- $h_{w, b}(x) = g(w^T x + b)$
- $g(z) = \begin{cases} 1, \ z \geq 0\\ -1, z < 0 \end{cases}$

# Functional and Geometric Margins
使用 margin 描述分类的准确度和可信度。

Functional Margin :
$$\hat\gamma^{(i)} = y^{(i)} (w^T x^{(i)} + b)$$
$$\hat\gamma = \min_{i = 1, 2, \cdots, m} \hat{\gamma}^{(i)}$$
对于线性分类器，functional margin 并不是一个很好的可信度度量值。因为**当 w 和 b 等比例放大时，决策边界没有变，而 functional margin 却等比例放大了**。

Geometric Margin :
$$\gamma^{(i)} = y^{(i)} ((\frac{w}{|w|})^T x^{(i)} + \frac{b}{|w|})$$
$$\gamma = \min_{i = 1, 2, \cdots, m} \gamma^{(i)}$$
Geometric margin 描述了样本点到 separating hyperplane 的距离。且**当 w 和 b 的等比例放大时，geometric margin 不变**。

# Optimal Margin Classifiers
对于**线性可分**的训练集，Optimal Margin Classifier 寻找使得 geometric margin 最大的 separating hyperplane。该问题可以形式化为以下优化问题：
$$\begin{align}
\max_{\gamma, w, b} & \ \gamma\\
\mathrm{s.t.} & \ y^{(i)} (w^Tx^{(i)} + b) \geq \gamma, \ i = 1, 2, \cdots, m\\
& \ ||w|| = 1
\end{align}$$
约束 $||w|| = 1$ 非凸。为了摆脱该约束，将优化问题改写为：
$$\begin{align}
\max_{\hat\gamma, w, b} & \ \frac{\hat\gamma}{||w||}\\
\mathrm{s.t.} & \ y^{(i)} (w^Tx^{(i)} + b) \geq \hat\gamma, \ i = 1, 2, \cdots, m\\
\end{align}$$
目标函数 $\frac{\hat\gamma}{||w||}$ 非凸。添加约束 $\hat\gamma = 1$ （这可以通过等比例缩放 w 和 b 做到）并变最大化 $\frac{1}{||w||}$ 为最小化 $||w||^2$ 可以将目标函数变为凸函数。优化问题改写为：
$$\begin{align}
\min_{w, b} & \ \frac{1}{2} ||w||^2\\
\mathrm{s.t.} & \ y^{(i)} (w^Tx^{(i)} + b) \geq 1, \ i = 1, 2, \cdots, m\\
\end{align}$$
这是一个 quadratic programming 问题。将其求解即为 optimal margin classifier。

Support Vector Machine Loss : [[2 Linear Classification]]

# Lagrange Duality
对于原始优化问题：
$$\begin{align}
\min_{w} & \ f(w) \\
\mathrm{s.t.} & \ h_i(w) = 0, \ i = 1, 2, \cdots, l
\end{align}$$
使用**拉格朗日乘数法**。定义**拉格朗日函数**：
$$\mathcal{L}(w, b) = f(w) + \sum_{i = 1}^l b_i h_i(w)$$
方程：
$$\frac{\partial \mathcal{L}}{\partial w} = 0; \ \frac{\partial \mathcal{L}}{\partial b_i} = 0$$
的解即为原始优化问题的解。

下面我们将拉格朗日乘数法扩展到含有不等式约束的优化问题。对于原始优化问题：
$$\begin{align}
\min_{w} & \ f(w) \\
\mathrm{s.t.} & \ g_i(w) \leq 0, \ i = 1, 2, \cdots, k\\
& \ h_i(w) = 0, \ i = 1, 2, \cdots, l
\end{align}$$
定义**广义拉格朗日函数**：
$$\mathcal{L}(w, \alpha, \beta) = f(w) + \sum_{i = 1}^k \alpha_i g_i(w) + \sum_{i = 1}^l \beta_i h_i(w)$$
考虑这个量：
$$\theta_\mathcal{P}(w) = \max_{\alpha, \beta: \alpha_i \geq 0} \mathcal{L}(w, \alpha, \beta)$$
可以证明：
$$\theta_\mathcal{P}(w) = \begin{cases}
f(w) & \text{if } w \text{ satisfies primal constraints}\\
\infty & \text{otherwise}
\end{cases}$$
因此，最小化问题：
$$\min_w \theta_\mathcal{P}(w) = \min_w \max_{\alpha, \beta: \alpha_i \geq 0} \mathcal{L}(w, \alpha, \beta)$$
与原始优化问题等价。

再考虑这个量：
$$\theta_\mathcal{D}(\alpha, \beta) = \min_w \mathcal{L}(w, \alpha, \beta)$$
定义原始优化问题的**对偶优化问题**：
$$\max_{\alpha, \beta: \alpha_i \geq 0} \theta_\mathcal{D}(\alpha, \beta) = \max_{\alpha, \beta: \alpha_i \geq 0} \min_w \mathcal{L}(w, \alpha, \beta)$$
由于函数的 "max min" ≤ 函数的 "min max"，原始优化问题和对偶优化问题满足以下关系：
$$d^* = \max_{\alpha, \beta: \alpha_i \geq 0} \min_w \mathcal{L}(w, \alpha, \beta) \leq \min_w \max_{\alpha, \beta: \alpha_i \geq 0} \mathcal{L}(w, \alpha, \beta) = p^*$$

当 $f$ 和 $g_i$ 为凸函数（海森矩阵为半正定矩阵），$h_i$ 为仿射变换，且约束 $g_i$ 严格可满足（存在 $w$，对于所有 $i$，满足 $g_i(w) < 0$）时，等号可以取到。此时，对偶优化问题的解就是原始优化问题的解。且解满足 **KKT 条件**：
$$\begin{align}
\frac{\partial}{\partial w} \mathcal{L}(w, \alpha, \beta) & = 0\\
\frac{\partial}{\partial \beta_i} \mathcal{L}(w, \alpha, \beta) & = 0, \ i = 1, 2, \cdots, l\\
\alpha_i g_i(w) & = 0, \ i = 1, 2, \cdots, k \tag{1}\\
g_i(w) & \leq 0, \ i = 1, 2, \cdots, k\\
\alpha_i & \geq 0, \ i = 1, 2, \cdots, k
\end{align}$$
公式 (1) 称为 **KKT dual complementary condition**。公式 (1) 表明，若 $g_i(w) > 0$，则 $\alpha_i = 0$。

# Optimal Margin Classifiers: The Dual Form
Optimal margin classifier 的优化问题：
$$\begin{align}
\min_{w, b} & \ \frac{1}{2} ||w||^2\\
\mathrm{s.t.} & \ y^{(i)} (w^Tx^{(i)} + b) \geq 1, \ i = 1, 2, \cdots, m\\
\end{align}$$
满足 $d^* \leq p^*$ 取等的条件。因此，我们可以通过求解对偶优化问题求解 optimal margin classifier。下面，我们就将 optimal margin classifier 的优化问题改写为其对偶优化问题。

首先，构造拉格朗日函数：
$$\mathcal{L}(w, b, \alpha) = \frac{1}{2} ||w||^2 - \sum_{i = 1}^m \alpha_i [y^{(i)} (w^Tx^{(i)} + b) - 1]$$
注：这里没有 $h_i(w)$ 和 $\beta$。

其次，求对偶函数：
$$\theta_\mathcal{D}(\alpha) = \min_{w, b} \mathcal{L}(w, b, \alpha)$$
对 $\mathcal{L}(w, b, \alpha)$ 求导：
$$\nabla_{w} \mathcal{L}(w, b, \alpha) = w - \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)}= 0$$
$$\nabla_b \mathcal{L}(w, b, \alpha) = - \sum_{i=1}^m \alpha_i y^{(i)} = 0$$
则：
$$w = \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)}$$
将上述两式带回到 $\mathcal{L}(w, b, \alpha)$ 中，得：
$$\theta_\mathcal{D}(\alpha) = \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i, j} h^{(i)} y^{(j)} \alpha_i \alpha_j \langle x^{(i)}, x^{(j)} \rangle$$

接着，将对偶函数与新的约束放在一起，得到对偶优化问题：
$$\begin{align}
\max_\alpha & \ \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i, j} h^{(i)} y^{(j)} \alpha_i \alpha_j \langle x^{(i)}, x^{(j)} \rangle\\
\mathrm{s.t.} & \ \ \alpha_i \geq 0, \ i = 1, 2, \cdots m\\
& \ \sum_{i=1}^m \alpha_i y^{(i)} = 0
\end{align}$$

最后，求解对偶优化问题，得到 $\alpha$。

对于原始优化问题的约束：
$$g_i(w, b) = - y^{(i)} (w^Tx^{(i)} + b) + 1 \leq 0$$
根据 KKT 条件，只有当样本 i 的 functional margin 恰好等于 1，即 $g_i(w, b) = 0$ 时，$\alpha_i \neq 0$。这些样本点被称为**支持向量**。支持向量的数量要比训练集的大小小很多。

求得 $\alpha$ 后，可以求得 $w$：
$$w = \sum_{i=1}^m \alpha_i y^{(i)} x^{(i)}$$

根据原始优化问题，对于正负样本的支持向量，有：
$$\max_{i:y^{(i)} = -1} w^T x^{(i)} + b = -1$$
$$\max_{i:y^{(i)} = 1} w^T x^{(i)} + b = 1$$
因此，可以求得 $b$：
$$b = - \frac{\max_{i:y^{(i)} = -1} w^T x^{(i)} + \max_{i:y^{(i)} = 1} w^T x^{(i)}}{2}$$

在推理时：
$$w^T x + b = \sum_{i = 1}^m \alpha_i y^{(i)} \langle x^{(i)}, x \rangle + b$$
只需要计算 $x$ 与支持向量的内积。

# Regularization and the Non-separable Case
将数据投影到高维的特征空间并不能确保投影后的数据线性可分，离群点也会对分割超平面产生较大的影响。为了让 SVM 适用于**线性不可分**的数据，并降低它对**离群点**的敏感性，我们对原始优化问题添加正则化：
$$\begin{align}
\min_{w, b, \xi} & \ \frac{1}{2} ||w||^2 + C\sum_{i = 1}^m \xi_i\\
\mathrm{s.t.} & \ y^{(i)} (w^Tx^{(i)} + b) \geq 1 - \xi_i, \ i = 1, 2, \cdots, m\\
& \ \xi_i \geq 0, \ i = 1, 2, \cdots, m\\
\end{align}$$

拉格朗日函数：
$$\mathcal{L}(w, b, \alpha, r) = \frac{1}{2}||w||^2 + C\sum_{i = 1}^m\xi_i - \sum_{i = 1}^m \alpha_i [y^{(i)} (w^Tx^{(i)} + b) - 1 + \xi_i] - \sum_{i = 1}^m r_i \xi_i$$

对偶优化问题：
$$\begin{align}
\max_\alpha & \ \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i, j} h^{(i)} y^{(j)} \alpha_i \alpha_j \langle x^{(i)}, x^{(j)} \rangle\\
\mathrm{s.t.} & \ \ 0 \leq \alpha_i \leq C, \ i = 1, 2, \cdots m\\
& \ \sum_{i=1}^m \alpha_i y^{(i)} = 0
\end{align}$$

w 和推理阶段的计算公式不变，b 的计算公式有变。KKT dual complementary condition 变为：
$$\begin{align}
\alpha_i = 0 & \Rightarrow y^{(i)} (w^Tx^{(i)} + b) \geq 1\\
\alpha_i = C & \Rightarrow y^{(i)} (w^Tx^{(i)} + b) \leq 1\\
0 < \alpha_i < C & \Rightarrow y^{(i)} (w^Tx^{(i)} + b) = 1
\end{align}$$

# The SMO Algorithm
## Coordinate ascent
对于无约束的优化问题：
$$\max_\alpha W(\alpha_1, \alpha_2, \cdots, \alpha_m)$$
Coordinate ascent 算法的每次内部循环都只针对某个变量进行优化：
```
Loop until convergence: { 
	For i = 1, . . . , m, { 
		αi := argmax_{αi} W(α1, . . . , αi−1, αi, αi+1, . . . , αm). 
	} 
}
```
上述算法中，内部循环按照 $a_1$ 到 $a_m$ 的顺序进行优化，也可以选择更为复杂的优化顺序。
## SMO (sequential minimal optimization)
对于 SVM 的对偶优化问题，我们不能直接套用 coordinate ascent 算法依次对每一个 $\alpha_i$ 进行优化，因为这样做会导致约束条件被破坏。为了确保约束条件不被破坏，必须同时优化至少两个 $\alpha_i$。基于此，SMO 算法的基本过程如下：
```
Loop till convergence: { 
	1. Select some pair αi and αj to update next (using a heuristic that tries to pick the two that will allow us to make the biggest progress towards the global maximum). 
	2. Reoptimize W(α) with respect to αi and αj , while holding all the other αk’s (k 6= i, j) fixed. 
}
```
SMO 算法是高效的。在固定了 $a_i$ 和 $a_j$ 之外的其他变量之后，我们可以将 $a_j$ 写成 $a_i$ 的函数，并带回到对偶优化问题的目标函数中。这使得目标函数变为了仅有 $a_i$ 一个自变量的二次函数。

























