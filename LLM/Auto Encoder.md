# Bert
使用 **masked language model（MLM）**，生成**深度的双向语言表征**。

Bert 的结构是多层 transformer encoder（因此是双向的）的堆叠。

Bert 对输入序列的预处理：
- 序列前添加 cls token
- 每两句话之间添加 sep token

Bert 的 input embedding：
- token embedding
	- 使用 WordPiece 进行分词，将每个 subword 在词表中的 id 输入 MLP 得到 token embedding（词嵌入向量）（等同于根据 id 在词嵌入向量表中查找词嵌入向量，但词嵌入向量表也是可学习的参数）
- positional embedding
- segment embedding
	- 同一个句子的 token 拥有相同的 segment embedding

Bert 的预训练策略：
1. masked language model
	提取 token 层次的信息。
	以 15% 的概率用 mask token（\[MASK\]）随机地对每一个训练序列中的 token 进行替换，然后预测 \[MASK\] 位置原有的单词。然而，测试时序列中并没有 \[MASK\]，为了保证一致性，采用以下替换策略：
	- 80% 的概率替换为 \[MASK\]
	- 10% 的概率替换为随机的词
	- 10% 的概率保留原词不变
2. next sentence prediction
	提取 sentence 层次的信息。
	从语料库中抽取两个句子，50% 的情况下这两个句子连在一起，50% 的情况下这两个句子不连在一起。用 \[CLS\] 进行二分类，判断输入的两个句子是否连在一起。
	
