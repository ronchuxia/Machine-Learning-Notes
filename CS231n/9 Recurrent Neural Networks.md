[[自然语言处理]]
# Image Captioning
1. CNN 提取图像的全局特征 $v$
2. RNN 根据图像的全局特征 $v$ 生成 caption
$$h_{t} = \tanh(W_x x_{t-1} + W_h h_{t-1} + W_v v)$$
与 RNN 的区别是每一步都融入了图像信息

# Image Captioning with Attention
1. CNN 提取图像的局部特征
2. RNN 在生成 caption 的同时也生成位置的分布，根据位置的分布将每个位置的特征向量加权求和，将加权和和这一步生成的 caption 合在一起作为下一步的输入





