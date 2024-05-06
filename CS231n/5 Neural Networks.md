# Single Neuron as a Binary Linear Classifier
- $x_i$ : the signal of the ith dendrite
- $w_i$ : the synaptic strength of the ith dendrite
- $fr$ : the firing rate of the neuron
- $f$ : the activation function that models the firing rate of the neuron
$$fr = f(\sum_i w_ix_i + b)$$

一个神经元是一个**二元分类器**
- Binary Cross-Entropy Loss：
	- Binary Softmax classifier / logistic regression 的损失函数
- Binary SVM Loss：
	- Binary SVM classifier 的损失函数

# Activation Functions
## Sigmoid
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$
- **梯度饱和**：当输入非常大或非常小的时候，sigmoid 函数的梯度接近于 0
- **非 0 对称**：sigmoid 函数的输出为正数，这会对下一层的梯度更新造成影响。由于 $f = w^Tx + b$，如果 $x$ 全为正，则 $\mathrm{d} w$ 要么全为正，要么全为负（取决于 $\mathrm{d}f$ 的正负），这会导致梯度来回摆动（不过当我们使用 BGS 时，会将 batch 中所有样本的梯度求和，这缓解了上述问题）。
由于 sigmoid 函数的不良性质，现在很少使用 sigmoid 函数。
## Tanh
$$\mathrm{tanh}(x) = 2\sigma(2x) - 1$$
- 梯度饱和
- 0 对称
- 是 sigmoid 函数的拉伸
由于 tanh 是 0 对称版本的 sigmoid，tanh 比 sigmoid 更常用。
## ReLU
Rectified Linear Unit
$$f(x) = \max(0, x)$$
- 无梯度饱和，加速收敛
- 计算方便
- **死亡神经元**：如果在某一次反向传播中，$\mathrm{d}w$ 很大，使得 $w$ 变得很小，对于所有训练样本 $w^Tx + b < 0$，则这个神经元的梯度就永远都是 0，不会再更新了。如果发生了死亡神经元问题，可以适当调小学习率（死亡神经元是梯度饱和的加强版）
## Leaky ReLU
$$f(x) = 1(x < 0)\cdot(\alpha x) + 1(x \geq 0)\cdot(x)$$
- $\alpha$ 是一个常数，但也可以作为一个参数
- 实际使用效果不一致
## Maxout
$$\max(w_1^T x + b_1, w_2^T x + b_2)$$
- ReLU 和 Leaky ReLU 都是 Maxout 的特例
- 同一个神经元需要两倍的参数量

# Neural Network Architectures
## Layer-wise organization
- 计算层数时一般不算输入层（层数 = 隐藏层 + 输出层）
- 逻辑回归和 SVM 是一层神经网络的特例
## Representational power
拥有至少一个隐藏层的神经网络可以拟合任何**连续函数**（非线性激活函数必不可少）。
[Neural networks and deep learning](http://neuralnetworksanddeeplearning.com/chap4.html)
使用多层网络是经验选择，更有利于优化。
## Setting number of layers and their sizes
使用更多的神经元可以更复杂的函数。
使用较多的神经元容易过拟合，可以用一些方法防止过拟合（比如 L2 正则化，dropout，添加输入噪声等）。
**不能通过减少神经元的数量防止过拟合**：对于神经元数量较少的神经网络，虽然它的损失函数有较少的局部最小值，但是这些局部最小值仍然较大，网络容易收敛到这些较差的局部最小值	






