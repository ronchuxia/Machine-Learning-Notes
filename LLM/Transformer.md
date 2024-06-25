[The Illustrated Transformer – Jay Alammar – Visualizing machine learning one concept at a time. (jalammar.github.io)](https://jalammar.github.io/illustrated-transformer/)
# RNN的缺点
- 无法抛弃不必要的特征
- **串行**计算，速度慢，无法堆叠多层
- 单向结构，没有考虑前文信息。解决方案：双向RNN
- 间隔远的词也有可能有很强的关系，但是RNN中只有相近的词关系最强（长期依赖问题）
注：RNN 的 decoder 中也可以有注意力机制

# Transformer中的模块
## Self Attention
输入特征向量$X$，由特征向量$X$乘以可学习的权重参数$W$得到query向量$Q$，key向量K和value向量$V$。其中，$Q$和$K$的维度为$d_k$。
![[Pasted image 20230822153638.png]]
将value向量V做线性组合得到第i个元素的新特征向量$X_i'$。线性组合时，用$Q_i$和$K_j$的内积作为$X_j$在$X_i'$中的权重。
$$X_i' = \sum_j w_{ij} \cdot V_j = \sum_j Q_i K_j^{T} \cdot V_j$$
实际使用时，需要先将权重softmax归一化。
$$X_i' = \sum_j w_{ij} \cdot V_j = \sum_j softmax(Q_i K_j^{T}) \cdot V_j$$
这种算法称为 dot-product attention。当 $d_k$ 较大时，$Q_i K_j^{T}$ 的方差较大，容易落在使 softmax 梯度消失的部分。因此，实际使用时，还要对内积除以维度开根号，将方差固定为 1：
$$X_i' = \sum_j w_{ij} \cdot V_j = \sum_j softmax(\frac{Q_i K_j^{T}}{\sqrt{d_k}}) \cdot V_j$$
这种算法称为scaled dot-product attention。
[保姆级分析self Attention为何除根号d，看不懂算我的 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/503321685)

Self attention可以堆叠多层。

TODO：Layer Normalization

## Multi Head
用多组权重参数W，获得多组Q，K和V，从而提取更多的特征。类似于多个卷积核，或 GNN 中多个空域。
![[Pasted image 20230822162427.png]]
TODO

## Positional Encoding
注意力机制只考虑了元素的特征，没有考虑元素所处的位置。Positional Encoding有多种实现方法，主要体现在以下几个方面：
- Positional Encoding的编码方式：1d, 2d, relative postion, ...
- Positional Encoding与特征向量的组合方式：直接相加，拼接，...
- Positional Encoding与模型的组合方式：在模型的哪个位置添加Positional Encoding
Transformer采用sinusoida positional encoding。
![[Pasted image 20230823190342.png]]

```python
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        """
        Construct the PositionalEncoding layer.

        Inputs:
         - embed_dim: the size of the embed dimension
         - dropout: the dropout value
         - max_len: the maximum possible length of the incoming sequence
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        assert embed_dim % 2 == 0
        # Create an array with a "batch dimension" of 1 (which will broadcast
        # across all examples in the batch).
        pe = torch.zeros(1, max_len, embed_dim)

        positions = torch.arange(0, max_len)  # shape (max_len, )
        exponents = torch.arange(0, embed_dim, 2) # shape (embed_dim // 2, )
        
        pe[:, :, 0::2] = torch.sin(torch.outer(positions, torch.pow(10000, - exponents / embed_dim)))
        pe[:, :, 1::2] = torch.cos(torch.outer(positions, torch.pow(10000, - exponents / embed_dim)))

        # Make sure the positional encodings will be saved with the model
        # parameters (mostly for completeness).
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Element-wise add positional embeddings to the input sequence.

        Inputs:
         - x: the sequence fed to the positional encoder model, of shape
              (N, S, D), where N is the batch size, S is the sequence length and
              D is embed dim
        Returns:
         - output: the input sequence + positional encodings, of shape (N, S, D)
        """
        N, S, D = x.shape
        # Create a placeholder, to be overwritten by your code below.
        output = torch.empty((N, S, D))

        pe = self.pe[:, :S, :D]
        output = x + pe
        output = self.dropout(output)

        return output
```

# Encoder
TODO

# Decoder
训练时可并行
测试时不可并行

# Vision Transformer（ViT）
![[Pasted image 20230822164531.png]]
ViT将图像按照卷积核的大小分成很多Patch，将每个Patch embed成一个多维的特征向量（这个过程称为Patch Embedding），并将这个特征向量组成的序列输入给Transformer Encoder。与卷积神经网络相比，ViT只需要一层，其感受野就能覆盖整个图像。具体实现细节如下：
- Patch Embedding相当于对输入图像做一个kernel_size和stride都等于patch_size的Conv2d。卷积时，使用多个卷积核，每个卷积核对某一Patch做卷积得到一个特征值，将所有特征值拼接在一起得到这一Patch的特征向量。
- 需要在整个序列前拼接一个Class Token，用于整合整张图像的特征。

## Positional Encoding的实现
文章做了Ablation Study，尝试了多种Positional Encoding：
- 1d Positional Encoding：将图像视作由很多块组成的1d序列，每个块有一个Positional Encoding。
- 2d Positional Encoding：将图像视作由很多块组成的2d网格，每一块的x坐标和y坐标都有一个Positional Encoding，将x坐标和y坐标拼接在一起作为这个块的Positional Encoding。
- Relative Positional Encoding
将Positional Encoding融入模型的方法也有多种：
- 在Transformer Encoder前做Positional Encoding。
- 在Transformer Encoder的每个Layer前都做Positional Encoding。
- 在Transformer Encoder的每个Layer前都做Positional Encoding（所有层共享一个Positional Encoding）。
文章最终选择了1d Positional Encoding并选择在Transformer Encoder前做Positional Encoding。Positional Encoding是一组可学习的权重参数，将特征向量加上Positional Encoding后再输入给Transformer Encoder。
	
# Swin Transformer
Vision Transformer中，迭代了L个Transformer Encoder，每个Transformer Encoder的输入序列长度是一样的。如果图像的分辨率较高，则序列的长度会很长，迭代多层encoder时训练很慢。
Swin Transformer结合了Transformer和卷积，在每个Transformer Stage之后做下采样（效果类似卷积）降低序列的长度，从而加快训练的速度。具体实现细节如下：
![[Pasted image 20230823131655.png]]
- 首先对图像做Patch Embedding，得到特征图，然后将特征图输入给Swin Transformer。
- Swin Transformer由多个Layer组成，每个Layer中有多个Swin Transformer Block。每两个Swin Transformer Block组成一组，其中，第一个做W-MSA，第二个做SW-MSA。
	- W-MSA将特征图分为多个Window，每个Window内部做MSA（一个Window包含多个Patch，对Window内的Patch做MSA）。
	- SW-MSA和W-MSA原理类似，但分Window时需要添加一个偏移量，以整合更多Patch中的特征（使得一个Patch可以获得它左上和右下两个Window中其他Patch的信息）。
- 每个Layer后做下采样，降低序列的长度。
![[Pasted image 20230823161240.png]]
- Swin Transformer使用Relative Position Bias的方法做Positional Encoding。

## SW-MSA的实现
SW-MSA在分块Window时添加了偏移量，导致图像四周的Patch被分成了较小的Window，总的Window数有所增加。为了保持总Window数不变，先对图像进行位移，然后再分Window，如下图。
![[Pasted image 20230823161147.png]]
偏移后在同一个Window内的两个Patch可能在原图中并不在同一个Window内，计算MSA时这两个Patch的attention应设为$-\infty$（或一个足够小的负数，在softmax时其权重就会被设为0）。因此，在得到每个Window内部的attention后，还要加上一个mask。
mask矩阵的计算方法如下：为偏移后的特征图中的每一个Patch编号（类似上图中的编号一样），在原图中属于同一个Window的Patch编号相同，属于不同Window的Patch编号不同。偏移后的Window中，如果Patch i和Patch j的编号相同，则mask ij设为0，否则mask ij设为$-\infty$。
也可以采用Sliding Window（综合上下左右4个Window，卷积就采用了Sliding Window），但是Shifted Window更快。

## 下采样（Patch Merging）的实现
![[Pasted image 20230823165143.png]]
把每个$2 \times 2$的块中的4个Patch拼接在一起，得到一个长度为4C的一维向量，然后用全连接层下采样到2C。（类似卷积，但不是卷积，卷积中同一个像素点的所有特征值乘的是同一个权重参数）

## Positional Encoding的实现
同一个Window中的两个Patch的相对坐标共有$(2 \cdot window\_size - 1)^2$种可能，每种相对坐标对应一个Relative Position Bias $B$。用如下公式计算attention。
![[Pasted image 20230823171630.png]]


