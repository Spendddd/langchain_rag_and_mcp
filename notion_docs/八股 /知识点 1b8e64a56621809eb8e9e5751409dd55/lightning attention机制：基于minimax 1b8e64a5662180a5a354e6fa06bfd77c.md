# lightning attention机制：基于minimax

State: 已完成

https://zhuanlan.zhihu.com/p/20473885314**MiniMax-01中Lightning Attention的由来（线性注意力进化史）**

## minimax论文证明过程

![image.png](lightning%20attention%E6%9C%BA%E5%88%B6%EF%BC%9A%E5%9F%BA%E4%BA%8Eminimax%201b8e64a5662180a5a354e6fa06bfd77c/image.png)

![image.png](lightning%20attention%E6%9C%BA%E5%88%B6%EF%BC%9A%E5%9F%BA%E4%BA%8Eminimax%201b8e64a5662180a5a354e6fa06bfd77c/image%201.png)

![image.png](lightning%20attention%E6%9C%BA%E5%88%B6%EF%BC%9A%E5%9F%BA%E4%BA%8Eminimax%201b8e64a5662180a5a354e6fa06bfd77c/image%202.png)

论文中指出作者们在实验过程中发现了线性注意力机制的检索能力存在局限，因此混合softmax注意力以提高检索能力

通过对标度定律实验、下游性能和速度对比的分析，我们得出结论:**纯线性注意力模型虽然计算效率高，但并不适合大型语言模型。这是由于他们天生无法进行检索，而检索能力对于上下文学习至关重要。**相比之下，我们的混合模型在检索和外推任务中不仅匹配而且超过了softmax注意力。