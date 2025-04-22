# RLHF：PPO、DPO、GRPO

State: 学习中

# **SFT 与 RLHF 算法特点**

| **算法** | **特点** |
| --- | --- |
| 监督微调（SFT） | ● 在标注的SFT数据上对预训练模型进行微调。 |
| 直接偏好优化（DPO） | ● DPO依赖于理论上的偏好模型，如 Bradley-Terry 模型，来测量奖励函数与经验偏好数据的对齐程度。它直接根据策略定义偏好损失，无需在训练过程中明确学习 Reward 模型。 |
| 近端策略优化（PPO） | ● PPO算法采用 Actor-Critic 架构，需要 Policy 模型、Value 模型、 Reward 模型、 Reference 模型。 
● 使用 Value 模型评估模型的预期总收益（模型回复的好坏） |
| 群体相对策略优化（GRPO） | ● GRPO算法采用 Actor-Critic 架构，需要 Reward 模型、Reference 模型，但是删掉了 Value 模型。 
● 不使用 Value 模型，而是使用一组 LLM 生成的针对同一上文输入的多次采样结果来做预期总收益的估计。 |

# PPO

PPO（Proximal Policy Optimization）是一种用于强化学习的策略优化算法，由 OpenAI 提出。它通过限制策略更新的幅度，确保训练过程的稳定性。

## 奖励模型

https://blog.csdn.net/gzroy/article/details/132630418

### 数据集的准备

在InstructGPT论文中，OpenAI介绍了如何准备数据，通过第一步的SFT模型，准备一批提示语，对于每个提示语，模型都生成多个回答，例如生成9个回答。然后人工对这9个回答的质量进行排序。之所以不是人工直接对这9个回答进行评分，是因为每个人对于回答的评分标准都不同，然而大家对于那个回答质量更高是比较容易得到统一的，因此采取排序的方式。获得了排序之后，我们就可以计算这个pair-wise的loss值，即把回答两两比较，其得分之间的差距应该足够大。例如对于回答A和回答B，A的质量比B的要高，那么我们可以用以下公式表示这两个回答之间的质量差值，其中x表示prompt, y表示对应的回答：

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image.png)

如果我们有9个回答，那么两两配对之后总共有（K2）个，k=9，总共36个配对。因此总的loss值为：

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%201.png)

对这个loss值进行最小化，即代表模型能最大限度的区分质量好和质量差的回答之间的评分。

### 建立奖励模型

按照InstructGPT论文的介绍，奖励模型最好基于SFT模型来构造，因此我这里也是采用同样的方式，基于之前训练好的SFT模型来进行训练。为了能够根据输入的文字得出一个分值，需要在原有模型的基础上，**去掉最后的(hidden_dim, vocab_size)的线性变换层，改为添加一个维度为(hidden_dim, 1)的线性变换层**，从而**将模型输出的隐变量映射为一个分值**。

**语言模型中的PPO强化学习术语：**

- **状态**：在语言模型的 PPO 训练中，状态指模型在生成下一个 token 时已经生成的 token 序列。
- **动作**：动作指模型在当前状态下选择的下一个 token。
- **关系**：动作的选择依赖于状态，模型基于之前的上下文生成下一个 token。

因为PPO中RM要对actor输出的每一个token都计算奖励值，因此在训练RM的时候需要RM本身也有生成一系列token的隐藏状态的能力，才能进行对每个token的奖励值计算；在训练RM的时候，还隐式要求奖励模型对于（指令）输入+正例输出作为输入以及（指令）输入+负例输出作为输入时，模型的输出长度需要是一致的，才能进行求每个token的奖励→求成对输入的每个token的奖励差值→求整个输出序列的奖励期望→作为损失的一部分

### 训练代码

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%202.png)

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%203.png)

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%204.png)

## 训练策略模型

https://zhuanlan.zhihu.com/p/1885405099548975288

### 训练过程

1. 生成文本（Rollout）：LLM（策略）为不同的提示生成大量文本样本。
2. 获取分数（奖励模型）：奖励模型对每个文本样本进行打分。
3. 计算优势（[GAE](https://zhida.zhihu.com/search?content_id=254829206&content_type=Article&match_order=1&q=GAE&zhida_source=entity) —— “好多少”分数）：这就是 GAE 的作用！它是一种巧妙的方法，用于计算每个单词选择的优劣，考虑奖励和价值函数的预测。（关于 GAE 的更多内容见下文！）
4. 优化 LLM（策略更新）：我们更新 LLM 的策略，以最大化一个特殊的 PPO 目标函数。这个目标函数现在有三个关键部分：
    - **鼓励更高奖励：**它推动 LLM 生成能够获得更高分数的文本。
    - **限制策略变化（剪切代理目标）**：它防止策略在一次更新中变化过大，确保稳定性。
        
        ![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%205.png)
        
    - [**KL 散度惩罚**](https://zhida.zhihu.com/search?content_id=254829206&content_type=Article&match_order=1&q=KL+%E6%95%A3%E5%BA%A6%E6%83%A9%E7%BD%9A&zhida_source=entity)：如果新策略与旧策略偏离太远，它会增加惩罚，进一步增强稳定性。
    - [**熵奖励**](https://zhida.zhihu.com/search?content_id=254829206&content_type=Article&match_order=1&q=%E7%86%B5%E5%A5%96%E5%8A%B1&zhida_source=entity)：它还包括一个熵奖励。简单来说，熵衡量 LLM 文本生成的“随机性”或“多样性”。增加熵奖励可以鼓励 LLM 更多地探索，而不是总是生成相同、可预测的响应。它有助于防止 LLM 过早变得“过于确定”，从而错过可能更好的策略。
    - 伪代码
        
        ```python
        # 这是一个高度简化的预期目标版本
        def ppo_loss_with_gae_entropy(old_policy_logprobs, new_policy_logprobs, advantages, kl_penalty_coef, clip_epsilon, entropy_bonus_coef):
            """概念性 PPO 损失函数，带有 GAE 和熵奖励（简化版）。"""
        
            ratio = np.exp(new_policy_logprobs - old_policy_logprobs)  # 概率比
        
            # 剪切代理目标（限制策略变化）
            surrogate_objective = np.minimum(ratio * advantages, np.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages)
            policy_loss = -np.mean(surrogate_objective)
        
            # KL 散度惩罚（保持接近旧策略）
            kl_divergence = np.mean(new_policy_logprobs - old_policy_logprobs)
            kl_penalty = kl_penalty_coef * kl_divergence
        
            # 熵奖励（鼓励探索）
            entropy = -np.mean(new_policy_logprobs)  # 简化版熵（概率越高 = 熵越低，取负值以最大化熵）
            entropy_bonus = entropy_bonus_coef * entropy
        
            total_loss = policy_loss + kl_penalty - entropy_bonus  # 减去熵奖励，因为我们希望*最大化*熵
            return total_loss
        ```
        
5. 更新价值函数（辅助教练更新）：训练价值函数成为一个更好的“辅助教练”——更准确地预测不同文本生成的“好坏”。

### **为什么选择 GAE？**

- 蒙特卡洛（MC）—— 高方差，低偏差：想象一下**等到整个文本生成后再获得奖励**，然后将该奖励分配给文本中的每一个单词。就像只有在小狗完成整个“坐下、待命、取回”动作序列后才给予奖励。对整个序列的奖励是准确的，但对单个动作（“坐下”与“待命”与“取回”）的信号非常嘈杂。高方差，学习速度慢。
- 时间差分（TD）—— 低方差，高偏差：想象一下**在每个单词生成后给予奖励**。“好单词！”“普通单词！”“很棒的单词！”信号不那么嘈杂，学习速度更快。但是，我们只是局部地判断单词，没有考虑整个文本的长期质量。可能会有偏差，可能会错过“大局”。
- GAE —— 平衡：广义优势估计（GAE）就像“多步 TD”。它考**虑了多个步骤（单词）上的奖励，平衡了方差（MC）与偏差（TD）之间的权衡**。就像**不仅在结束时给予奖励，还在价值函数预测的指导下，为沿途的“小步骤”给予奖励**。
- GAE计算公式
    
    ![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%206.png)
    
    即当前时刻的优势函数由当前TD误差和下一时刻的优势函数加权组成。
    
    代码实现步骤
    
    ![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%207.png)
    

从GAE的计算公式不难发现，**对于每个时间步都要用RM计算一次当前时间步的奖励值rt**，RM输出的logits形状是**(batch_size, sequence_length)**，需要经过一定的处理才能获得形状为**(batch_size, )的奖励值序列，**为了将这些logits转换成一个单一的奖励值，通常需要对时间步的奖励进行某种形式的聚合。以下是几种常见的聚合方法：求和、求平均、求最大值、求最小值、求加权和、取最后一个token对应的logits值。

### 进一步解释

1. PPO的核心是**强化“实际收益 - 预估收益”**，所以PPO损失函数的核心是它的负数；
2. **“实际收益”分为两部分，一部分约束过程，一部分奖励结果。**
    1. “约束过程”来自reference model，因为actor model其实已经具备比较好的知识和能力了，所以要避免在PPO把它训歪了。而reference model其实就是actor model最初的状态，它的输入是prompt和actor model的response，输出是【1】**作为reference的response的每一个token的概率分布；**
    2. **“**奖励结果**”来自**reward model，它是提前train好的model，输入是response的最后一个token，输出是对这个token的奖励，也是对整个response的奖励。
    3. 二者加起来就是实际收益，像是老师来评估你的考卷，要按老师课上教你的思路来解题，当然如果你最终答案是对的，那也能一定程度上宽容你的“标新立异”。
3. 我们知道对于actor model（一个LLM），输入上文，它将输出**下一个token的概率分布**，而critic model其实就是actor model的倒数第二层连接到一个新的全连接层上，它的输出是下一个token的**“预估收益”**。critic model像是老师对你的预期模型，这个模型会根据你这次的考试结果的进行更新。

### **核心思想**

PPO 的核心在于限制策略更新的幅度，避免因更新过大导致性能下降。它通过引入“裁剪”机制，控制新旧策略之间的差异。

### **公式**

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%208.png)

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%209.png)

其中，*πθ* 和 *πθ_old* 分别是当前策略模型和旧策略模型，q 和 o 是从问题数据集和旧策略 *πθ_old* 中采样的问题和输出。超参数 *ϵ* 用于稳定训练过程。优势 *A_i* 是通过广义优势估计（Generalized Advantage Estimation, GAE）计算的，计算过程 基于 奖励 {*ri*≥*j*} 和学习到的值函数 *Vπold*。为了减轻对奖励模型的过度优化，标准方法是**在每个标记的奖励中添加一个来自参考模型的每个标记的KL惩罚**，即：

![](https://pica.zhimg.com/v2-db4e32e853e05549914d3e21ef196da2_1440w.jpg)

其中， r 是奖励模型， πref 是参考模型，通常是初始的监督微调（SFT）模型，而 β 是 KL 惩罚项的系数。

***Q和V都是期望，Q是自状态St起针对某一个动作at的预期奖励（某个动作at对应的Rt），V是策略层面的，V是Q的加权平均；r是针对某一个动作的即时奖励，Rt是自t时刻状态St起的预期奖励（针对全部动作的奖励的预期）***

### **步骤**

1. **采样：**使用当前策略与环境交互，收集数据，在语言模型中，可以类比为生成补全（generating completions）。
2. **计算优势值：**基于收集的数据计算优势值函数 。
3. **优化目标函数：**通过**梯度上升**优化目标函数 Lclip(xita)。
4. **更新策略：**重复上述步骤，直到策略收敛。

算法流程伪代码如下：

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2010.png)

### **优点**

- **稳定性：**通过裁剪机制，避免策略更新过大。
- **高效性：**相比 TRPO，PPO 实现更简单，计算效率更高。

### 缺点

PPO 中的***值函数通常是一个与策略模型大小相当的模型***，***这带来了显著的内存和计算负担***。此外，在 LLMs 的上下文中，值函数在训练过程中被用作优势计算中的Baseline，但通常只有最后一个 token 会被奖励模型赋予奖励分数，这可能使得值函数的训练变得复杂。

### **补充**

在强化学习中，**策略的目标是最大化期望回报，而不是最小化损失**。所以，在PPO中使用的是**梯度上升**，原因在于它的优化目标是最大化目标函数（如强化学习中的期望回报），而不是最小化损失函数（如分类或回归问题）。

# DPO

DPO 是“新晋成员”——一种更简单、更高效的方式来进行偏好学习，跳过了 RL 的复杂性。

直截了当：DPO 就像是直接告诉 LLM：“响应 A 比响应 B 更好。多生成像 A 这样的响应，少生成像 B 这样的响应！”它省略了 RL 中用于策略优化的奖励模型这一中间环节。

DPO 避免了 PPO 的迭代 RL 循环。它直接基于人类偏好数据利用一个巧妙的损失函数对 LLM 进行优化。

### **DPO 训练流程**

- 偏好数据仍然是关键：与 PPO 一样，DPO 仍然从相同的关键成分开始：人类偏好数据（成对的响应，带有标签，指示哪个响应更受青睐）。人类反馈仍然是基础！
- 直接策略更新（分类式损失——直接使用 logits！）：这是 DPO 的魔法所在。**DPO 使用一个特殊的损失函数直接比较两个模型的 logits（概率之前的原始输出分数）**：
    - 增加首选响应的 logits（和概率）：让当前模型在未来更有可能生成像响应 A 这样的响应。
    - 减少非首选响应的 logits（和概率）：让当前模型在未来更不可能生成像响应 B 这样的响应。
    - 保持接近参考模型（隐式 KL 控制）：损失函数还隐式地鼓励当前模型在行为上保持与参考模型的接近（使用参考模型的 logits），这有助于稳定性，类似于 PPO 的 KL 惩罚，但直接嵌入在损失函数中！
    - 当前模型（正在训练中）：我们将首选响应（响应 A）和非首选响应（响应 B）都输入到我们正在训练的当前 LLM 中，得到两者的 logits。
    - 参考模型（旧版本）：我们还将响应 A 和响应 B 输入到一个参考模型中。这通常是 LLM 的旧版本（比如我们开始时的 SFT 模型）。我们也会从参考模型中得到 logits。
- DPO 的损失函数直接使用这两个模型的 logits 来计算损失，这与分类任务中使用的**二元交叉熵**损失非常相似。这个损失函数旨在：可以这样理解：DPO 的损失函数就像一个“偏好指南针”，直接根据首选和非首选响应的相对 logits 指导 LLM 的权重，而无需显式预测奖励。
    - 代码
        
        ```python
        # 注意：这不是实际公式。
        # 这是一个高度简化的预期目标版本
        def dpo_loss(policy_logits_preferred, policy_logits_dispreferred, ref_logits_preferred, ref_logits_dispreferred, beta_kl):
            """概念性 DPO 损失函数（简化版——直接使用 logits）。"""
        
            # 1. 从 logits 中获取对数概率（当前和参考模型的首选和非首选响应）
            policy_logprob_preferred = F.log_softmax(policy_logits_preferred, dim=-1).gather(...)  # 提取首选响应中实际标记的对数概率
            policy_logprob_dispreferred = F.log_softmax(policy_logits_dispreferred, dim=-1).gather(...)  # 提取非首选响应中实际标记的对数概率
            ref_policy_logprob_preferred = F.log_softmax(ref_logits_preferred, dim=-1).gather(...)  # 同样适用于参考模型
            ref_policy_logprob_dispreferred = F.log_softmax(ref_logits_dispreferred, dim=-1).gather(...)
        
            # 2. 计算对数比率（使用对数概率——如前所述）
            log_ratio = policy_logprob_preferred - policy_logprob_dispreferred - (ref_policy_logprob_preferred - ref_policy_logprob_dispreferred)
        
            # 3. 偏好概率（Bradley-Terry 模型——隐式奖励信号）
            preference_prob = 1 / (1 + np.exp(-beta_kl * log_ratio))
        
            # 4. 二元交叉熵损失（直接优化策略）
            dpo_loss = -np.log(preference_prob + 1e-8)
            return dpo_loss
        ```
        

# GRPO

https://zhuanlan.zhihu.com/p/25985130568：**揭开DeepSeek-R1的神秘面纱：GRPO 核心技术详解**

https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247569612&idx=3&sn=a3e0e3fd74c391da56fe39d2ac5ba62c&chksm=eab415c5169c91180f00186aa3bfd96a94950f17c03181a279828a0f097a44f874287699b45e&scene=27：**Deepseek的RL算法GRPO解读**

## 前置知识

### **强化学习基本概念**

在强化学习中，我们通常会讨论下面的一个问题背景：一个智能体（agent）在某个环境中可以执行一些动作（action），它在执行某个动作之后会从一个状态（state）切换到另外一个状态，同时所在的环境也会给它一个反馈（reward），而我们的目标则是最大化这个agent在环境中所能获得的reward，如下图所示：

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2011.png)

上面的问题背景通常被称为[马尔可夫决策过程](https://zhida.zhihu.com/search?content_id=254219327&content_type=Article&match_order=1&q=%E9%A9%AC%E5%B0%94%E5%8F%AF%E5%A4%AB%E5%86%B3%E7%AD%96%E8%BF%87%E7%A8%8B&zhida_source=entity)（Markov decision process)，简称MDP。

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2012.png)

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2013.png)

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2014.png)

### **强化学习算法的种类**

在强化学习问题中，根据**是否知道环境相关的信息**，我们通常把问题分为两类：**model-based和model-free**。**model-based是指我们能提前知道奖励函数r和状态转移概率P**，反之model-free则二者都不知道。model-based的问题解决有一个固定的套路可以使用，就是dynamic programming（[动态规划](https://zhida.zhihu.com/search?content_id=254219327&content_type=Article&match_order=1&q=%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92&zhida_source=entity)），这里就不细说了。

在现实世界中，model-free形式的强化学习问题要更为常见。为了解决这类问题，我们通常有两类方法：一种是**基于价值(value-based)的方法**（例如[Q-learning](https://zhida.zhihu.com/search?content_id=254219327&content_type=Article&match_order=1&q=Q-learning&zhida_source=entity)和DQN），另一种是**基于策略(policy-based)的方法**。二者的主要区别是：基于值函数的方法主要是学习值函数，然后根据值函数导出一个策略，学习过程中不存在一个显式的策略；而**基于策略的方法则是直接显式地学习一个目标策略**。今天我们所要讲解的**GRPO**算法则是一种基于策略的方法。

### **从[策略梯度算法](https://zhida.zhihu.com/search?content_id=254219327&content_type=Article&match_order=1&q=%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E7%AE%97%E6%B3%95&zhida_source=entity)说起**

GRPO算法本质上来说是从策略梯度算法发源而来，因此先给大家介绍相关概念以便于理解后面复杂的公式。

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2015.png)

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2016.png)

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2017.png)

为了最大化这个期望值，我们会采用梯度上升的方式进行参数更新：

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2018.png)

倒三角形符号（∇）表示**梯度**（Gradient），即函数关于参数的偏导数。在机器学习和强化学习中，梯度用于更新模型参数，以优化目标函数（如损失函数或期望回报）。

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2019.png)

但是这样会带来一个明显的问题，数据利用效率不高。每次参数更新前，都需要采样大量的数据，训练的时间开销全都集中在了数据采样上。为了解决采样时间开销大的问题，我们可以使用[**重要性采样**](https://zhida.zhihu.com/search?content_id=254219327&content_type=Article&match_order=1&q=%E9%87%8D%E8%A6%81%E6%80%A7%E9%87%87%E6%A0%B7&zhida_source=entity)技术。

**重要性采样**通常被用来**估计一个分布的期望值**。从数学上来说，它的原理是**通过不同分布的样本进行估计，然后乘上一个重要性权重（即两个分布概率的比值）**，这样可以**无需从原分布中采样，用另一个简单分布的样本也能计算原分布的期望：**

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2020.png)

- 重要性采样
    - **E[f]**：目标期望值，即我们要估计的期望值。
    - **π(x)**：目标分布的概率密度函数（PDF），即我们希望从中采样的分布（原分布）。
    - **p(x)**：重要性分布的概率密度函数，即我们实际从中采样的分布（简单分布）。
    - **f(x)**：我们感兴趣的函数，即我们要计算期望值的函数。
    - **π(x)/p(x)**：重要性权重，用于调整样本，使得它们能够正确反映目标分布的特性。

使用重要性采样的好处是，可以让我们使用旧策略θ′采样的数据，多次更新策略网络参数θ，从而大大节省了训练中采样数据的时间开销：

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2021.png)

事情到这里看起来比较完美了，但事实上，在使用过程中我们还会遇到一些问题：比如在很多游戏场景中，得到的奖励总是正的。如果所有动作的奖励均为正，那么策略更新时会盲目提高所有动作的概率（只是提升幅度不同），**但无法区分哪些动作比其他动作更好**，即**缺乏相对比较**。同时，还有可能引起高方差问题：原始奖励的绝对值差异极大，导致梯度更新的方差过大，训练不稳定。因此，我们通常**希望奖励值有正有负，通过将奖励减去一个基线b，我们就能达到目标**。而这个新的奖励值我们称为**优势函数**，**基线b我们通常取所有动作的平均预期奖励V：**

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2022.png)

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2023.png)

### [**TRPO算法](https://zhida.zhihu.com/search?content_id=254219327&content_type=Article&match_order=1&q=TRPO%E7%AE%97%E6%B3%95&zhida_source=entity)和[PPO算法](https://zhida.zhihu.com/search?content_id=254219327&content_type=Article&match_order=1&q=PPO%E7%AE%97%E6%B3%95&zhida_source=entity)**

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2024.png)

**TRPO算法求解的难度主要来源于它的KL约束条件**。熟悉凸优化的朋友可能知道，求解带约束问题和无约束问题难度相差可是非常大。那么有没有一种方法，让我们既可以约束θ和θ′的差异程度，又不将它作为约束条件呢。**PPO算法采用将KL散度项直接加入到优化目标中，将有约束优化问题转化为无约束优化问题进行求解**：

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2025.png)

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2026.png)

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2027.png)

取min的作用：在 PPO 的损失函数中，min 操作通过比较原始的重要性采样项和裁剪后的重要性采样项，**确保策略更新不会过于激进**，从而保持训练的稳定性。

## **语言模型中的PPO算法**

上面都是在纯强化背景下的PPO算法，那么在LLM领域，PPO算法又是如何应用的呢？通常来说，我们会构建四个模型：

- [Actor Model](https://zhida.zhihu.com/search?content_id=254219327&content_type=Article&match_order=1&q=Actor+Model&zhida_source=entity)：这其实代表的就是agent，在LLM领域里这就是我们需要训练的语言模型；
- [Critic Model](https://zhida.zhihu.com/search?content_id=254219327&content_type=Article&match_order=1&q=Critic+Model&zhida_source=entity)：它的作用是预估总收益，目的是学习一个值函数V;
- [Reward Model](https://zhida.zhihu.com/search?content_id=254219327&content_type=Article&match_order=1&q=Reward+Model&zhida_source=entity): 它的目标是计算即时收益，即学习奖励函数r；
- Reference Model: 它的作用是给语言模型增加一些约束，防止需要训练的语言模型训歪。

通常来说，在RLHF-PPO阶段，**Actor/Critic model都是需要训练的，而Reward/Reference Model则参数冻结。**我们通常采用下面的方式去初始化这些模型：actor/reference model**采用sft阶段训练后的语言模型初始化**，critic model和reward model则**采用语言模型的backbone + 各自的value head来进行初始化**。一个完整的RLHF-PPO训练过程如下：

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2028.png)

1. 将一个batch的prompts送进Actor语言模型，语言模型产生回答responses；
2. 将prompt + responses送进Critic/Reward/Reference Model，让他们生成用于计算actor/critic loss的数据。按照强化学习的术语，我们将这些数据称为经验；
3. 最后根据这些经验计算出actor/critic loss，然后更新actor和critic model。

其中，我们的actor loss为：

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2029.png)

- Advt
    
    ![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2030.png)
    
- Pold
    
    old就是reference model
    

为了区分动作A和优势函数，这里我们将优势函数记为Adv。而这个**actor loss正是我们的PPO2算法里的优化目标**。**Critic loss则定义为预估预期收益和实际预期收益的MSE loss（均方误差损失）**。

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2031.png)

## **GRPO算法**

铺垫了这么多，终于到我们的核心GRPO算法了。

在上面的PPO算法中，我们通常会引入一个critic model（或者称为value model）来判断每个动作的优劣，从而改进策略。但它的引入也同时带来了两个问题：**价值函数估计可能不准确，在LLM的语境里，通常只有一个完整的句子会容易给出一个好坏的判断，而对中间过程生成的token，我们很难给一个准确的奖励值，而价值函数估计不准确则会导致学习策略变差；**其次就是**模型显存占用高，消耗计算资源大**，因为critic model通常和actor model参数量差不多。为了解决这个问题，**GRPO算法直接在模型层面删掉了critic model**。如下图所示：

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2032.png)

具体来说，GRPO算法的流程如下：

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2033.png)

1. 在RL迭代中，**基于策略模型的采样结果为Reward model生成新的训练集，利用回放机制持续训练旧reward model**。之后**将当前策略模型设为当前参考模型，继续用更新后的奖励模型训练策略模型**；（5这一步可选，也就是可以训练也可以不训练奖励模型，可以更新参考模型也可以不更新参考模型）
2. 不断持续迭代，直到满足停止条件。

需要注意的是，GRPO算法存在两种类型的奖励：**过程奖励**和**结果奖励**。过程奖励会对模型输出的每一个token进行奖励，而结果奖励则只会对一个response中的最后一个token输出奖励。在计算优势函数A时，GRPO算法与之前的PPO算法也有所不同，具体而言：它**将奖励ri减去组平均奖励并除以组标准差进行归一化，将归一化后的奖励作为优势**，以结果奖励为例，优势函数可以写为：

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2034.png)

过程奖励也是类似，用reward model对输出的每一步打分，同样进行归一化，再**计算每个 token 之后步骤的归一化奖励之和作为优势**。

### GRPO原始的过程奖励和结果奖励

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2035.png)

### **结果监督强化学习与 GRPO：**

对于每个问题 q，从旧策略模型 πθ_old 中抽取一组输出 {o1, o2, ..., oG}。然后使用奖励模型对这些输出进行评分，产生相应的 G 个奖励 r={r1, r2, ..., rG}。随后，通过减去组平均值并除以组标准差来对这些奖励进行标准化处理。结果监督在每个输出 oi 的末尾提供标准化的奖励，并将输出中所有token的优势 Aˆi,t 设定为该标准化奖励，即 **Aˆi,t = (ri - mean(r)) / std(r)**，然后通过最大化方程（3）中定义的目标来优化策略。

### **过程监督强化学习与GRPO:**

结果监督仅在每个输出结束时提供奖励，这可能不足以有效监督复杂数学任务中的策略。遵循历史方法，我们还探讨了过程监督，它在每个推理步骤结束时提供奖励。

具体来说，给定问题 q 和 G 个抽样输出 {o1, o2, ..., oG}，使用过程奖励模型对每个输出步骤进行评分，从而得到相应的奖励：R={{rindex(1), ..., rindex(K1)}, ..., {rindex(1)11G, ..., rindex(KG)G}}，其中 index(j) 是第 j 步的结束标记索引，Ki 是第 i 个输出中的总步数。

我们也用平均值和标准差对这些奖励进行标准化处理，即 **˜rindex(j)i = (rindex(j) - mean(R)) / std(R)**。接下来，过程监督计算每个标记的优势作为后续步骤的标准化奖励之和，即 **Aˆi,t = ∑index(j)≥t ˜rindex(j)i**，并通过最大化方程（3）中定义的目标来优化策略。

### GRPO 的迭代强化学习

随着强化学习训练过程的进行，旧的奖励模型可能不足以监督当前的策略模型。因此如算法1中所示，**在迭代GRPO中，会基于策略模型的采样结果生成新的奖励模型训练集，并使用重放机制（包含10%的历史数据）不断训练旧的奖励模型。**随后，将参考模型设置为策略模型，并使用新的奖励模型不断训练策略模型。

注意，在deepseek中使用的是简化版的GRPO：舍弃了过程奖励，只保留结果奖励，因此直接舍弃了使用奖励模型，而**用简单的奖励函数代替了奖励模型的作用**：

Reward Modeling： DeepSeek-R1-Zero 中使用了**基于规则（rule-based）的 reward model**，由两部分组成：

Accuracy rewards：准确性奖励，准确性奖励模型用于评估响应是否正确。例如，对于结果确定的数学问题，模型需要以规定的格式（例如在一个框内）提供最终答案，从而可以进行可靠的基于规则的正确性验证。同样地，对于LeetCode问题，可以使用编译器基于预定义的测试用例生成反馈。

Format rewards：格式奖励，除了准确性奖励模型外，还使用格式奖励模型，要求模型将其思考过程放在 <think> 和 <\think> 标签之间。

简单总结一下GRPO算法和PPO算法的不同点：

- PPO算法属于actor-critic类型的强化学习算法，**需要训练一个与策略模型规模相当的价值函数作为基线**，而GRPO算法**丢弃了critic model，放弃了价值函数近似**，通过对同一问题采样多个输出，用这些输出的平均奖励作为基线；
- 优势函数的计算方式不同，PPO算法采用的**广义优势估计（GAE）基于奖励和学习到的价值函数计算优势At**，而GRPO算法**根据组内输出的相对奖励计算优势A^i,t**，【因此不需要训练critic model，critic model的省略得益于一个q对应一组生成】。

至此，我们也就讲完了GRPO算法，是不是也不太复杂。

### **GRPO算法在R1中的应用**

在DeepSeek-R1模型发布之前，大部分研究认为**使用过程奖励模型（PRM）要优于使用结果奖励**（也被称为规则奖励，rule-based reward，仅通过规则评判回答的正确性并给出奖励）。这一观点在 DeepSeek 提出 GRPO 的论文中有所体现，OpenAI所公开的研究成果也持类似看法。然而，**DeepSeek-R1 的出现打破了常规认知。它证明了仅使用rule-based reward就能在推理任务中取得令人瞩目的成果。**

而在R1的训练中，主要使用了以下两种的rule-based reward：

- **准确性奖励**：这一奖励模型专注于**评估模型的响应是否正确**。以数学问题为例，若问题有确定性的答案，**模型必须按照指定格式给出最终结果**，这样就能依据预设的规则验证答案的正确性。同样，在面对 LeetCode 编程题时，会借助编译器根据预先设定的测试用例生成反馈，以此判断模型给出的代码是否正确；
- **格式奖励**：除了准确性奖励，DeepSeek-R1 还引入了格式奖励模型。该模型**要求模型将思考过程严格置于和标签之间。**

仅仅通过这两项rule-based reward，并结合GRPO算法，DeepSeek-R1-Zero的推理能力就能获得不断提升。

### 按照标准的GRPO实现（一步只更新一次actor模型），loss中关于优势的部分值将一直为0，也就是说loss只包含KL散度

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2036.png)

![image.png](RLHF%EF%BC%9APPO%E3%80%81DPO%E3%80%81GRPO%201bbe64a5662180f1bab3e4aa07acfe6b/image%2037.png)

与 PPO 不同，**GRPO 通过直接使用奖励模型的输出来估计基线，避免了训练一个复杂的值函数**。此外，GRPO 通过直接在损失函数中加入策略模型和参考模型之间的 KL 散度来正则化，而不是在奖励中加入 KL 惩罚项，从而简化了训练过程。

与（2）中使用的 KL 惩罚项不同，GRPO使用下面的无偏估计来估计 KL 散度：

![](https://picx.zhimg.com/v2-faf116b9ee5719fbe511cf4fd6310ba1_1440w.jpg)

该值一定为正。

# PPO和GRPO的奖励函数的区别

[GRPO](https://zhida.zhihu.com/search?content_id=714488936&content_type=Answer&match_order=1&q=GRPO&zhida_source=entity) 与 [PPO](https://zhida.zhihu.com/search?content_id=714488936&content_type=Answer&match_order=1&q=PPO&zhida_source=entity) 的区别主要就是在优势函数的计算中

PPO 是通过拟合一个状态价值函数（critic）来计算中间过程中的 action 的价值

**而 GRPO 是为每个 action 采取了同样价值（一个组内整个 trajectory 的价值的平均）**

所以说 **PPO 是 step-wise 的**，会对不同 action 得到不同的价值，而 **GPRO 不是 step-wise 的**。

那么问题来了为什么 不是 step-wise 的 GRPO 在 LLM 训练中为什么同样有效，这就是接下来探讨的问题。

**step-wise reward（逐步奖励） 在大语言模型（LLM）训练中往往不太起作用，甚至可能会带来负面影响。**这主要是因为 **LLM 生成的文本质量通常是由整体决定的，而不是逐步累积的**。下面具体分析为什么 step-wise reward 对于 LLM 训练效果不好，以及为什么像 [**Outcome Supervision](https://zhida.zhihu.com/search?content_id=714488936&content_type=Answer&match_order=1&q=Outcome+Supervision&zhida_source=entity)（GRPO）** 这样的机制更合适。

## **1. 为什么 step-wise reward 在 LLM 训练中不起作用？**

在标准的强化学习（如机器人控制、游戏 AI）中，step-wise reward 是合理的，因为每个动作都会对最终目标产生直接影响。但是在 LLM 训练（例如 ChatGPT 调优）中，它存在以下问题：

### **(1) LLM 生成的是完整序列，而非逐步决策**

- **LLM 生成一个完整的文本序列，最终的质量通常由 整体连贯性、可读性、上下文相关性 等决定，而非单个 token 的选择。**
- 如果为每个 token 分配 step-wise reward，可能会误导模型，使其优化局部 token 而不是整体句子。

✅ **为什么 Outcome Supervision 更适合？**

- Outcome Supervision（如 GRPO）直接对整个输出序列进行评分，并将其作为整个序列的奖励信号，使得 LLM 关注整体优化，而非逐步微调。

### **(2) 奖励模型很难给每个 token 评分**

- 在 [RLHF](https://zhida.zhihu.com/search?content_id=714488936&content_type=Answer&match_order=1&q=RLHF&zhida_source=entity)（基于人类反馈的强化学习）中，奖励模型通常用于评估完整的句子、段落或对话，而不是逐个 token 评分。
- 例如：
- "这个回答听起来自然且符合事实。" → 适合整体评分
- "这个 token 需要 +0.2 的奖励，那个 token 需要 -0.1 奖励。" → 很难实现
- 由于 LLM 是通过 Transformer 机制 **同时预测多个 token**，其 **token 级别的概率并不独立**，因此给单个 token 评分并不合理。

✅ **为什么 Outcome Supervision 更适合？**

- 直接对整个回答评分，并用 **标准化奖励** r~\tilde{r} 作为全局 Advantage，让所有 token 共享相同的奖励信号。

### **(3) step-wise reward 可能会引导模型产生局部最优**

- 如果使用 step-wise reward，模型可能会为了优化局部 token 而牺牲整体连贯性。
- 例如，在对话任务中：
- 单独奖励“你好”或“谢谢”这样的 token 可能会让模型频繁使用它们，而忽略整体句子的质量。
- 可能会导致生成的句子更加模板化，失去多样性。

✅ **为什么 Outcome Supervision 更适合？**

- 通过对完整回答评分，使模型学习到 **全局最优策略**，而不是关注局部 token。

## **2. 什么时候 step-wise reward 可能有用？**

尽管 step-wise reward 在 LLM 训练中通常效果不佳，但在某些特定情况下可能仍然有用：

| 任务类型 | step-wise reward 是否有效 | 解释 |
| --- | --- | --- |
| 对话/文本生成 | 无效 | 文本质量是整体决定的，不能只优化单个 token。 |
| 代码生成 | 无效 | 代码的正确性取决于整体逻辑，而不是单个 token。 |
| 摘要生成 | 无效 | 摘要的质量依赖于全局信息，而非逐步构造。 |
| 机器人控制/游戏 AI | ✅ 有效 | 每个 step 的决策都会影响最终结果，适合 step-wise reward。 |
| 数学公式推理 | ✅ 部分有效 | 可能可以基于 intermediate step 进行奖励，但最终还是取决于整体正确性。 |