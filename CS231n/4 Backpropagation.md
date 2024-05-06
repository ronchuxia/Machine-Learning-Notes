![[Computational Graph.png]]
**计算图**（computational graphs）可以帮助我们方便地计算梯度。

图中的每一个门的计算（前向传播和反向传播）都是局部的：
$$输入的梯度 = 输出的梯度 \times 输出关于输入的梯度$$

因此，我们可以从后往前**分阶段**地计算梯度。这样的好处是我们并不需要得到梯度的解析式，就能计算出梯度的值。

Tricks
- 如果遇到分叉，需要将所有分叉的梯度相加（多变量求偏导的规则）
- 为了简化运算过程，可以将一些门**合并**（比如上图中的 sigmoid 函数）
- 为了方便反向传播，可以**缓存**一些前向传播时的中间变量，

# Patterns in Backward Flow
反向传播过程具有一定的可解释性：
- add：将梯度等值地传给每一个输入
- mul
- max：将梯度传给最大的输入
$$\frac{\partial \max(x, y)}{\partial x} = 1 \ (x\geq y)$$
注：mul 的梯度的大小和输入的大小有关，因此对输入做预处理是有必要的

# Gradients for Vectorized Operations
$$D = WX$$
$$\mathrm{d}W = \mathrm{d}D \cdot X^T$$
$$\mathrm{d}X = W^T \cdot \mathrm{d}D$$
用**维度分析**辅助记忆！











