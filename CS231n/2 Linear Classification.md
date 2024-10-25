# Linear Classifier
$$f(x_i, W, b) = Wx_i + b$$
- $W$ 是一组分类器，$W$ 的每一行是一个分类器。分类器会偏好这一类的某些特征，比如 ship 类的分类器会偏好蓝色（权重大）
- $b$ 作为偏移量，不与 $x$ 直接交互，而影响每一类的判断，比如当训练集中 ship 类比较多时，偏移量会偏好 ship 类

线性分类器是特征空间中的一个**超平面**
![](https://cs231n.github.io/assets/pixelspace.jpeg)
- 调整 $W$ 即旋转超平面
- 调整 $b$ 即平移超平面，如果没有 $b$，则所有超平面经过原点

线性分类器也可以解释成一个**模板**
- 将模板与样本做内积，计算两者的匹配程度
[[2 Logistic Regression]]

线性分类器也可以解释成 nearest neighbor
- 每一类只取一个代表，且这个代表是通过学习得到的
- 比如 ship 类的代表/模板就以蓝色为主

代表/模板
![](https://cs231n.github.io/assets/templates.jpg)
- 代表/模板融合了数据集中样本的特征，比如马的两种朝向
- 神经网络可以将这些特征做进一步的拆分和整合（神经元）

Trick：为 $x$ 添加一个维度 1，并将 $b$ 添加到 $W$ 的最后一列，则 $f = Wx$。

# Multiclass Support Vector Machine Loss
SVM classifier 的损失函数
- $x_i$：第 i 个样本
- $y_i$：第 i 个样本的类别
- $s_j$：第 i 个样本预测为第 j 类的 score
- $L_i$：第 i 个样本的损失
$$L_i = \sum_{j \neq y_i} \max(0, s_j - s_{y_i} + \Delta)$$
SVM Loss 希望错误类别的 score 比正确类别的 score 至少低 $\Delta$（一个超参数）。
- $\max(0, -)$ : hinge loss
- $\max(0, -)^2$ : squared hinge loss

添加 **L2 正则化惩罚**，抑制过大的权重：
$$R(W) = \sum\sum W_{k,l}^2$$
- 使权重**分布均匀**，防止某些特征对预测结果具有决定性的影响（**防止过拟合**）（惩罚 peaky 的权重，偏好 diffuse 的权重），从而提高泛化性
- 添加 L2 正则化惩罚后，SVM Loss 就不可能为零了

完整的 SVM Loss：
$$L = \frac{1}{N} \sum_i^N L_i + \lambda R(W)$$
- $\Delta$ 影响不大，可以通过调整 $W$ 轻易达成
- $\lambda$ 影响较大

其他 SVM 类型：
- One-VS-All SVM
- All-VS-All SVM
- Structured SVM

# Cross Entropy Loss
Softmax classifier 的损失函数
$$L_i = -\log \left( \frac{e^{s_{y_i}}}{\sum_j e^{s_j}} \right)$$
$$L = \frac{1}{N} \sum_i^N L_i + \lambda R(W)$$
信息论解释：
- $p(x)$：真实分布
- $q(x)$：预测分布
- $H(p, q)$：真实分布和预测分布的交叉熵
$$H(p, q) = − \sum_x p(x)\log(q(x))$$
$$p(x) = [0, \cdots, 1, \cdots, 0]$$
$$q(x) = \left[ \frac{e^{s_{y_0}}}{\sum_j e^{s_j}}, \cdots, \frac{e^{s_{y_i}}}{\sum_j e^{s_j}}, \cdots, \frac{e^{s_{y_N}}}{\sum_j e^{s_j}} \right]$$
因此，$L_i = H(p, q)$。又由于 $H(p, q) = H(p) + D_{KL}(p||q)$，且 $\Delta$ 函数 $p$ 的熵 $H(p) = 0$，即 Cross-Entropy Loss 最小化 KL 散度。

概率论解释：MLE。[[3 Generalized Linear Models]]

Trick：除以一个很大的数会导致  **numerically unstable**，通常先将 $s$ 中的所有元素减去 $\max(s)$ 后再计算 Cross-Entropy Loss：
$$L_i = -\log \frac{e^{s_{y_i}}}{\sum_j e^{s_j}} = - \log \frac{e^{s_{y_i} - \max(s)}}{\sum_j e^{s_j - \max(s)}}$$

$\lambda$ 越大，$W$ 越小，$s$ 越小，softmax 得到的分布越均匀。

与 Cross-Entropy Loss 相比，SVM Loss 更关注**局部**，对于已经正确分类的分类器，它不会再去更新。

Softmax 函数的梯度：
[Derivative of the Softmax Function and the Categorical Cross-Entropy Loss | by Thomas Kurbiel | Towards Data Science](https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1)

Cross-Entropy Loss 的梯度：
$$\frac{\partial L_i}{\partial s_k} = p_k - 1\{y_i = k\}$$






