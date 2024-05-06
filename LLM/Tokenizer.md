[NLP三大Subword模型详解：BPE、WordPiece、ULM - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/191648421)
# BPE (Byte Pair Encoding)
词表构建过程：
1. 准备足够大的训练语料，并确定目标词表的大小
2. 将单词拆分为成最小单元（比如 26 个字母加上各种符号），作为初始词表
3. 在训练语料上统计相邻 subword 的频数，**选取频数最高的相邻 subword 合并**成新的 subword
4. 如果合并后的两个 subword 不再单独出现，则从词表中删去
5. 重复第 3 步直到达到第 1 步设定的目标词表的大小或下一个相邻 subword 的最高频数为 1

注：每个单词结尾添加一个终止符，表示空格。

解码过程：
1. 将词典中的所有 subword **按照长度由大到小排序**
2. 对于单词 w，依次遍历排好序的词典，找到匹配的 subword 并输出
3. 对于剩余的字符串按照 3 的方法继续进行匹配
4. 如果遍历完字典后，仍然有剩余的字符串，则将其替换为特殊符号输出，如 ”\<unk\>”

# WordPiece
WordPiece 选择能够最大限度提升语言模型概率的 subword 进行合并。
TODO



