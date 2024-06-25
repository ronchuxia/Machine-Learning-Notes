对于给定的特征映射 $\phi$，它的核函数为：
$$K(x, z) = \phi(x)^T \phi(z)$$

对于一些特定的 $\phi$，计算 $\phi(x)$ 或 $\phi(x)^T \phi(z)$ 非常困难，但计算 $K(x, z)$ 非常容易。这使得 SVM 可以容易地将输入属性（input attributes）投影到高维的特征空间进行学习，而不需要显式地计算输入特征（input features）。

核函数：
$$\begin{align}K(x, z) & = (x^T z)^2\\
& = \bigg(\sum_{i = 1}^n x_i z_i\bigg) \bigg(\sum_{j = 1}^n x_j z_j\bigg)\\
& = \sum_{i = 1}^n \sum_{j = 1}^n x_i z_i x_j z_j\\
& = \sum_{i, j} (x_i x_j) (z_i z_j)
\end{align}$$
对应特征映射（$n = 3$ 时）：
$$\phi(x) = \begin{bmatrix}
x_1x_1\\
x_1x_2\\
x_1x_3\\
x_2x_1\\
x_2x_2\\
x_2x_3\\
x_3x_1\\
x_3x_2\\
x_3x_3\\
\end{bmatrix}$$

核函数：
$$\begin{align}
K(x, z) & = (x^T z + c)^2\\
& = (x^T z)^2 + 2c(x^T z) + c^2\\
& = \sum_{i, j} (x_ix_j)(z_iz_j) + \sum_{i = 1}^n (\sqrt{2c}x_i)(\sqrt{2c}z_i) + c^2
\end{align}$$
对应特征映射（$n = 3$ 时）：
$$\phi(x) = \begin{bmatrix}
x_1x_1\\
x_1x_2\\
x_1x_3\\
x_2x_1\\
x_2x_2\\
x_2x_3\\
x_3x_1\\
x_3x_2\\
x_3x_3\\
\sqrt{2c}x_1\\
\sqrt{2c}x_2\\
\sqrt{2c}x_3\\
c
\end{bmatrix}$$

最常用的两个核函数：

|               $K(x, z)$                |                      $\phi(x)$                       |
| :------------------------------------: | :--------------------------------------------------: |
|            $(x^T z + c)^d$             | ${n + d \choose d}$ 个 $0 \leq \text{次数} \leq d$ 的单项式 |
| $\exp(-\frac{\|x - z\|^2}{2\sigma^2})$ |       无穷个 $0 \leq \text{次数} \leq \infty$ 的单项式        |

核函数的构造：[[ps2]]

对于训练集 $\{x^{(1)}, x^{(2)}, \cdots, x^{(m)}\}$ 和核函数 $K(x, z) = \phi(x)^T \phi(z)$，若矩阵 $K$ 满足 $K_{i, j} = K(x^{(i)}, x^{(j)})$，则称矩阵 $K$ 为核矩阵。可以证明，矩阵 $K$ 是一个对称的半正定矩阵：
$$\begin{align}
z^TKz & = \sum_i \sum_j z_i K_{ij} z_j\\
& = \sum_i \sum_j z_i K(x^{(i)}, x^{(j)}) z_j\\
& = \sum_i \sum_j z_i \phi(x^{(i)})^T \phi(x^{(j)}) z_j\\
& = \sum_i \sum_j \sum_k z_i \phi_k(x^{(i)}) \phi_k(x^{(j)}) z_j\\
& = \sum_k (\sum_i z_i \phi_k(x^{(i)}))^2\\
& \geq 0
\end{align}$$

**Mercer 定理**
$K(x, z)$ 是一个核函数（存在与 $K(x, z)$ 对应的特征映射）的**充分必要条件**为对于任意的 $\{x^{(1)}, x^{(2)}, \cdots, x^{(m)}\}$，$\ (m \leq \infty)$，核矩阵 $K$ 是一个对称的半正定矩阵。 

如果我们可以将一个算法改写为内积的形式，我们就可以使用 kernel trick。使用 kernel trick 的步骤如下：
1. 将原算法改写为内积的形式
2. 将原算法中所有的 $x$ 用 $\phi(x)$ 代替
3. 在算法运行的过程中，用核函数计算内积

Kernel trick 的使用案例：[[ps2]]
