# Kernels
可视化第一个卷积层的卷积核。可以看到卷积核所匹配的特征。

# Activation Maps
可视化中间层的特征图。可以看到哪些神经元在哪些输入下被激活。

# Maximally Activating Patches
可视化输入哪些图像可以最大程度地激活某个神经元。可以看到这个神经元所匹配的特征。

# Features
可视化哪些图像的特征向量相似（KNN/PCA/t-SNE）。可以看到网络是否成果提取了图像的语义特征。

# Occlusion Experiments
可视化图像中部分区域被遮挡对预测概率的影响。可以看到哪些区域对分类起到至关重要的作用。

# Saliency Maps
可视化分类得分相对于图像像素的梯度。可以看到哪些像素对分类起到至关重要的作用。

# Imtermediate Features via Guided Backpropagation
可视化某个神经元相对于图像像素的梯度。效果类似 Maximally Activating Patches。
为了让可视化的效果更清晰，引导式反向传播中，ReLU 将正梯度反向传播，将负梯度置零。

# Gradient Ascent
可视化哪些图像可以最大程度地激活某个神经元。初始化图像为 0 或高斯噪声，然后使用梯度上升算法更新图像，最大化神经元输出。需要对生成图像进行正则化以生成有意义的图像。

# Feature Inversion
可视化哪些图像拥有与某张真实图像相似的特征值。使用梯度上升算法最小化生成图像的特征值与某张真实图像的特征值的距离。

# Texture Synthesis
可视化哪些图像拥有与某张真实图像相似的纹理特征。
- 使用神经网络提取图像特征
- 计算图像特征的 Gram Matrix，Gram Matrix 描述了图像的纹理特征
- 使用梯度上升算法最小化生成图像的 Gram Matrix 与某张真实图像的 Gram Matrix 的距离

# Style Transfer
结合 feature inversion 和 texture synthesis，在保留图像语义特征的同时修改图像的纹理特征。
缺点：需要多次前向传播和反向传播，开销大。









