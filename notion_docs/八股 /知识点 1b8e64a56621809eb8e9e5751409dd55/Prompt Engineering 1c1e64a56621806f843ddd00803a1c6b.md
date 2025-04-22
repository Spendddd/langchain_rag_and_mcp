# Prompt Engineering

State: 未开始

https://wjn1996.blog.csdn.net/article/details/120607050?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-120607050-blog-145784581.235%5Ev43%5Epc_blog_bottom_relevance_base5&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7EPaidSort-1-120607050-blog-145784581.235%5Ev43%5Epc_blog_bottom_relevance_base5&utm_relevant_index=1

# PLM任务

## 经典pre-trained任务

### MLM

![image.png](Prompt%20Engineering%201c1e64a56621806f843ddd00803a1c6b/image.png)

MLM是一种自监督的训练方法，其先从大规模的无监督语料上通过固定的替换策略获得自监督语料，设计预训练的目标来训练模型，具体的可以描述为:

替换策略:在所有语料中，随机抽取15%的文本。被选中的文本中，则有80%的文本中，随机挑选一个token并替换为 [mask]，10%的文本中则随机挑选一个**token**替换为其他token，10%的文本中保持不变。

训练目标:当模型遇见 [mask] token时，则根据学习得到的上下文语义去预测该位置可能的词，因此，训练的目标是对整个词表上的分类任务，可以使用交叉信息熵作为目标函数。

现如今有诸多针对MLM的改进版本，我们挑选两个经典的改进进行介绍

### wMM

Whole Word Masking(wWM):来源于ROBERTa等，其认为BERT经过分词后得到的是word piece，而**BERT的MLM则是基于word piece进行随机替换操作的，即Single-token Masking**，因此被mask的token语义并不完整。而WWM则表示**被mask的必须是个完整的单词**。

### EMR

Entity Mention Replacement(EMR):来源于ERNIE-BAIDU等、其通常是**在知识增强的预训练场景中，即给定已知的知识库(实体)，对文本中的整个实体进行mask**，而不是单一的token或字符。

![image.png](Prompt%20Engineering%201c1e64a56621806f843ddd00803a1c6b/image%201.png)

## Task-specific任务

### Single-text classification(单句分类)

常见的单句分类任务有短文本分类、长文本分类、意图识别、情感分析、关系抽取等。给定一个文本，喂入多层Transformer模型中，**获得最后一层的隐状态向量后，再输入到新添加的分类器MLP中进行分类**。在Fine-tuning阶段，则**通过交叉信息熵损失函数训练分类器**。

### Sentence-pair Classification(句子匹配/成对分类)

常见的匹配类型任务有语义推理、语义蕴含、文本匹配与检索等。给定两个文本，用于判断其是否存在匹配关系。此时**将两个文本拼接后喂入模型中，训练策略则与Single-text Classification一样;**

### Span Text Prediction(区间预测)

常见的任务类型有抽取式阅读理解、实体抽取、抽取式摘要等。给定一个passage和auery，根据query寻找passage中可靠的字序列作为预测答案。通常该类任务需要模型预测区间的起始位置，因此在Transformer头部添加两个分类器以预测两个位置。

### Single-token Classification(字符分类)

此类涵盖序列标注、完形填空、拼写检测等任务。获得给定文本的隐状态向量后，喂入MLP中，获得每个token对应的预测结果，并采用交叉熵进行训练。

### Text Generation(文本生成)

文本生成任务常用于生成式摘要、机器翻译、问答等。通常选择单向的预训练语言模型实现文本的自回归生成，当然也有部分研究探索非自回归的双向Transformer进行文本生成任务。BART等模型则结合单向和双向实现生成任务,