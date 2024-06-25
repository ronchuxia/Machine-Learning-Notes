[[网络结构]]
# Convolutional Layer
**Local Connectivity**
The connections are **local in 2D space (along width and height)**, but always **full along the entire depth** of the input volume.

**Parameter Sharing**
每个 kernel / filter 都识别一种特定的特征（如边缘、颜色等）。由于特征的平移不变性，图像中不同位置的相同特征可以由同一个 kernel 提取，因此不同位置的神经元可以共享 kernel。 

注：”特征的平移不变性“这个假设并不总是合理。有时我们可能想要让处于不同位置的相同特征乘以不同的权重。这种结构叫做 **Locally-Connected Layer**。

**Implementation as Matrix Multiplication**
1. 将每个卷积区域展开成一维向量 (img2col)，得到感受野矩阵
2. 将每个卷积核展开成一维向量，得到卷积核矩阵
3. 将卷积核矩阵与感受野矩阵相乘，得到输出矩阵
4. 重塑输出矩阵

**Backpropagation**
卷积的反向传播仍是卷积。

# Pooling Layer
**Getting Rid of Pooling**
Discarding pooling layers has also been found to be important in training good generative models, such as variational autoencoders (VAEs) or generative adversarial networks (GANs).

# CNN Architectures
[[经典网络]]
卷积神经网络中，显存占用主要集中在前几个卷积层，参数量主要集中在 FC 层。因此，常通过减少 FC 层来减少参数量。





