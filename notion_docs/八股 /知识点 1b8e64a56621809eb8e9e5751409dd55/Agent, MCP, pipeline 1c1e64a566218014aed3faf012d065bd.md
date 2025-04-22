# Agent, MCP, pipeline

State: 学习中

https://blog.csdn.net/m0_59235245/article/details/146416045

# Qwen-Agent

https://blog.csdn.net/shangguanliubei/article/details/145931328：**阿里 Qwen-Agent：开源Agent开发框架简介**

https://blog.csdn.net/2401_85373691/article/details/144924555：**大模型实战|Qwen-Agent 使用篇**

https://baijiahao.baidu.com/s?id=1801616412559701649&wfr=spider&for=pc：**百万上下文RAG，Agent还能这么玩**

https://juejin.cn/post/7454175860460765194：**Qwen-Agent：阿里通义开源 AI Agent 应用开发框架，支持构建多智能体，具备自动记忆上下文等能力**

## Qwen-Agent核心功能与特点

2.1 功能调用
**功能调用是Qwen-Agent的一项基础能力，它允许AI模型在处理任务时主动调用预定义的函数或工具**。这一特性使得Qwen-Agent能够与外部工具和API进行互动，从而扩展了AI应用的范围。例如，开发者可以定义一个用于生成图像的工具，当用户请求生成特定内容的图像时，Qwen-Agent可以调用该工具并返回生成的图像链接。

2.2 代码解释器
Qwen-Agent内置了代码解释器工具，能够执行用户提供的代码片段。这为开发者提供了极大的灵活性，使得**智能代理不仅能够理解和生成文本，还能够执行实际的操作**。例如，在处理图像生成任务时，Qwen-Agent可以先调用图像生成工具生成图像，然后通过代码解释器下载图像并进行进一步的处理，如翻转图像等。

2.3 多模态处理
Qwen-Agent**支持多模态输入和输出**，能够理解和生成文本、图像等多种类型的数据。这使得开发者可以构建更加丰富和交互式的应用。例如，一个图像理解与文本生成的Agent可以先理解用户提供的图像内容，然后生成与图像相关的文本描述或故事。

2.4 记忆能力
Qwen-Agent具备记忆能力，能够记住之前的对话内容和执行的操作。这**使得智能代理能够更好地理解用户的意图，并提供更加连贯和个性化的服务**。例如，在与用户进行多轮对话时，Qwen-Agent可以根据之前的对话内容提供更加准确的回答和建议。

## Qwen-Agent如何拓展上下文记忆到百万量级：借用RAG的思想

**Qwen-Agent**的设计思路虽然与**LangChain**相似，但其发布几个的Agent示例却很有意思。今天本文将深入探讨如何使用**Qwen-Agent**将上下文记忆扩展到**百万量级**，让Agent的智能得到更广泛的应用。

### 暴力关键字检索优于向量方案

在处理大规模文本数据时，一个关键的挑战是如何快速准确地定位到最相关的信息块。Qwen-Agent通过一种看似“暴力”的方法——**基于LLM判断相关性** AND **基于关键字检索**，解决了这一难题。这种方法虽然简单，但在实际操作中却显示出了意想不到的效果。

### 关键字检索的基本原理

**关键字检索**是一种直接且高效的方法，尤其是在面对大规模文本数据时。通过预先定义的关键字，我们可以快速定位到包含这些关键字的文本块。这种方法的优势在于其速度和简单性，尤其是在处理大规模数据时。

### 实现关键字检索的步骤

1. **预处理文本数据**：将大规模文本数据分块。
2. **基于LLM判断相关性**：并行处理每个分块，让聊天模型评估其与用户查询的相关性， 如果相关则输出相关句子用于后续检索。
    
    ![image.png](Agent,%20MCP,%20pipeline%201c1e64a566218014aed3faf012d065bd/image.png)
    
3. **分析用户指令，提取关键字**：通过LLM对用户指令进行两个方面的预处理。1.区分指令信息与非指令信息。2.从查询的信息部分推导出多语言关键词
    
    ![image.png](Agent,%20MCP,%20pipeline%201c1e64a566218014aed3faf012d065bd/image%201.png)
    

### **Qwen-Agent的关键字检索方法**

1. 将维基百科语料库切分为多个小的文本块，每个文本块包含512字。
2. 将用户问题通过LLM转换为检索关键字，例如“爱因斯坦”、“1905年”、“理论”。并区分检索信息与指令

```
{
"信息": ["爱因斯坦在1905年发表了什么重要的理论"],
"指令": ["用英文回复"]
}
```

1. 基于检索信息（**爱因斯坦在1905年发表了什么重要的理论**），并行过滤所有分块，查询相关性，并抽取相关语句。
2. 基于检索关键字（**“爱因斯坦”、“1905年”、“理论”**）检索分块。
3. **将匹配到的文本块输入到Qwen模型中，模型会根据这些文本块的内容推理出答案**：“爱因斯坦在1905年发表了狭义相对论。”

通过这种方式，Qwen-Agent可以更精准地定位到与用户查询相关的文本块，避免了无关信息的干扰，提高了检索效率和答案的准确性。

### 检索之前先做推理，多跳问题又快又准

在基于文档的问题回答中，一个典型的挑战是多跳推理。多跳推理是指需要结合多个文档的信息才能回答用户问题的情况。例如，用户可能会问“《红楼梦》的作者是谁的粉丝？”，要回答这个问题，就需要先找到《红楼梦》的作者是曹雪芹，然后找到曹雪芹是哪个朝代的人，最后才能找到答案。

### 什么是多跳推理

**多跳推理是指在回答一个问题时，需要跨越多个不同的文本块或信息源，逐步推理得到最终答案**。这种方法能够提供更准确和全面的回答，但也增加了计算复杂度。

### 多跳推理的实现步骤

1. **初步推理**：首先**将用户问题转化分解为逐级递进的子问题**。
2. **子问题检索**：调用上述RAG的能力，进行问题检索与回答。
3. **多跳推理**：**逐步在推理链上进行推理**，得到最终答案。

### 以用促训，Agent智能反哺模型

官方实验结果表明，4k-智能体在处理长上下文方面的表现优于32k-模型。这种分块阅读所有上下文的方式，使得Agent能够克服原生模型在长上下文训练上的不足。而Agent智能在使用过程中生产的数据，则能迭代用于后续长文本上下文的进一步微调。

## **开发自定义 Agent**

```python
!pip install -U "qwen-agent[gui,rag,code_interpreter,python_executor]"
# 或者使用 `pip install -U qwen-agent` 安装最小依赖。

import pprint
import urllib.parse
import json5
from qwen_agent.agents import Assistant
from qwen_agent.tools.base import BaseTool, register_tool

# 步骤 1（可选）：添加自定义工具 `my_image_gen`
@register_tool('my_image_gen')
class MyImageGen(BaseTool):
    description = 'AI 绘画（图像生成）服务，输入文本描述，返回基于文本信息绘制的图像 URL。'
    parameters = [{
        'name': 'prompt',
        'type': 'string',
        'description': '所需图像内容的详细描述，使用英文',
        'required': True
    }]

    def call(self, params: str, **kwargs) -> str:
        prompt = json5.loads(params)['prompt']
        prompt = urllib.parse.quote(prompt)
        return json5.dumps(
            {'image_url': f'https://image.pollinations.ai/prompt/{prompt}'},
            ensure_ascii=False)

# 步骤 2：配置使用的 LLM
llm_cfg = {
    'model': 'qwen-max',
    'model_server': 'dashscope',
    'generate_cfg': {
        'top_p': 0.8
    }
}

# 步骤 3：创建 Agent，需要用户自定义agent流程
system_instruction = '''你是一个有用的助手。
在收到用户的请求后，你应该：
- 首先绘制图像并获取图像 URL，
- 然后运行代码 `request.get(image_url)` 下载图像，
- 最后从给定的文档中选择一个图像操作来处理图像。
请使用 `plt.show()` 显示图像。'''
tools = ['my_image_gen', 'code_interpreter']
files = ['./examples/resource/doc.pdf']
bot = Assistant(llm=llm_cfg,
                system_message=system_instruction,
                function_list=tools,
                files=files)

# 步骤 4：运行 Agent
messages = []
while True:
    query = input('用户查询: ')
    messages.append({'role': 'user', 'content': query})
    response = []
    for response in bot.run(messages=messages):
        print('助手响应:')
        pprint.pprint(response, indent=2)
    messages.extend(response)

```

# AutoGLM沉思

AutoGLM 沉思的基础模型架构是这样的：

底层语言模型 GLM-4-Air-0414

中层推理和沉思模型 GLM-Z1-Air、GLM-Z1-Rumination

**加上工程/产品层的 AutoGLM 工具，就行程了 AutoGLM 沉思的整个技术栈。**

**智谱也计划在 4 月 14 日全面正式开源 AutoGLM 沉思背后的所有模型。**

![image.png](Agent,%20MCP,%20pipeline%201c1e64a566218014aed3faf012d065bd/image%202.png)

![image.png](Agent,%20MCP,%20pipeline%201c1e64a566218014aed3faf012d065bd/image%203.png)

## 测试案例

让它制作一份「不同于网上所有主流路线的日本两周小众经典行攻略，要求绝对不去最火的目的地，要小众景点，但也要评价比较好的。」

AutoGLM 沉思比较准确地拆解了需求，思考逻辑也比较清楚：它首先去搜了最简单的关键词「日本旅游」，了解主流路线和景点，然后又去搜索了「日本小众旅游景点」之类的关键词——通过这几个步骤，它在本次对话的记忆内部构建了一个知识库，也即什么是主流的，什么是小众的。

这个任务总共做了 20 多次思考。有时候几次思考之间会有重复，比如搜索的是相同的关键词，访问了相同或者相似的链接等。这有可能是因为单次搜索到的信息不足够，毕竟**沉思/深度搜索的本质其实也是不断地自我怀疑和推翻，直到达到足够置信度**时候才进入下一步。

APPSO 还注意到它有点**过度依赖特定的网站作为信息来源**，打开的所有 tab 里有 90% 都是小红书和知乎（各一半左右）。反而真正的旅行专业资料库，比如马蜂窝、穷游，或者哪怕是 OTA 平台，它一次没用过。

如果要做一份真正的小众攻略，重度依赖小红书的结果可能并不理想。毕竟能上小红书的热门笔记，这个景点应该并不真的小众。一个真正的小众景点旅行者，恐怕不想去 momo 们已经去过或者都想去的地方……

APPSO 注意到，**AutoGLM 沉思在沉思过后自己提出了「路线规划合理，不要有无意义的反折」、「行程节奏合理，别太特种兵」之类的要求。只是实际结果没有反映它自己提出的这些要求**：比如头几天在濑户内海来回折返，有时候一天内去两三个相隔一小时以上的地点，略微特种兵；第二周从青森向南到仙台，然后又从仙台飞机向北大跨度飞到了北海道，并且北海道只留了两天。考虑到日本大跨度旅行基本都靠 JR，票价昂贵，合理的路线应该是顺着一个方向不回头，除非不得不去大城市换车，一般不应该折返。

但总体来讲，这份攻略是有效的：**它呈现了一些提问者未曾考虑过的目的地，也试图在一次行程里去到季节、气候、风格完全不一样的地方（而不是围在大东京、富士山、京坂奈区域来回打转）。**

**从这个角度，它遵循了提示的要求，并且展现出了深度思考的结果。**

就像你不应该直接把 AI 生成的结果直接拿去用一样，这份攻略提供了一个还算不错的基础，让旅行者可以自行优化具体的目的地、路线和中间的交通方式。旅行不只是上车睡觉下车拍照，还应该兼顾人文和自然，深入当地文化传统，探索自然景观，以及至少感受一把在地最有特色的体验项目。

只要你的期待不是即问即用，AutoGLM 沉思给出的答案是足够令人满意的。

**Agent 「自动驾驶」能力，和路况、驾驶位有很大关系**

在其它更「轻松」的任务（比如做旅行规划、游戏攻略、查找简单信息等）当中，AutoGLM 工具的 browser use 能力是没有太大问题的。

**但 APPSO 发现，一旦当前网站的视觉设计相对复杂，或者设计的有一些陷阱，AutoGLM 工具就很容易被「使绊子」。**

一个最直接的例子就是电商网站。APPSO 给出明确提示，「去淘宝或京东购买一件重磅日系 T 恤」，AutoGLM 沉思制定了宏伟的计划和明确的分工——然而却连淘宝首页的山门都进不去，甚至找不到搜索框在哪里。而且它似乎被「找不到搜索框」这件事完全阻挡住了，甚至也没有去看网页的其它位置——如果它看了的话，肯定会发现相关商品早就出现在首页推荐里了。

对于这个测试中发现的意外情况，智谱 CEO 张鹏表示，「点背不能赖社会」，AutoGLM 沉思目前仍在 beta 阶段，还有很大的进化空间，而且目前的升级速度也很快（APPSO 在正式发布版上测试淘宝的使用效果已经没那么磕绊了）。

张鹏指出，在模型作为服务或作为产品 (MaaS） 的理念下，模型产品自己的能力要像木桶一样，高且全面。或许现在 AutoGLM 工具的视觉能力还不如人，处理意外情况的能力还不够，归根结底可能是泛化能力还不够，但这些能力的提升并不是模型问题，而是纯粹的工程层面——不需要担心。

从模型底座层面，AutoGLM 沉思也有提升的空间。

经常用大语言模型产品的朋友都知道，提示写的越具体，规则和边界设定的越明确，它的效果越好，越有希望生成符合用户提示的结果。基于大语言模型的 agent 也是一样。

但是提示不能无限扩展，就好比你招了一个秘书帮你干活，但你不应该总是每次都把「找谁」、「什么地点」、「什么时候」、「去哪」等一切的信息都讲清楚，ta 才能勉强顺利地帮你搞定一个饭局的准备工作。

大语言模型很强大，但也有它糟糕的地方：**只受到文本规则的约束，缺乏真正的实际问题的规划能力，任务过程中容易被卡住**；缺乏足够长的上下文记忆空间，任务持续时间太长就持续不下去；上一个步骤的错误会随着步骤逐渐放大，直至失败。

**AutoGLM 沉思也是一个基于大语言模型的 agent，即便在 agent 能力上做了很多工作，但仍然难免受到大语言模型的诅咒。思考能力越强，越容易想多、想歪。**

从 APPSO 的试用过程中可以看到，除了一些绝对基础的概念（比如「旅游」、「T 恤」、「公司」）之外，它并没有稍微复杂的上层知识。用户每次发出任何指令，它都要先自己打开浏览器，上网学习一遍，明确用户的所指，在本次对话的有限记忆空间内建立一个知识库，然后再去进行后续的步骤。

而就它目前最擅长和依赖的那几个信息来源来看，**一旦用户任务的复杂性、专业性「上了强度」，想要它在用户可接受的时间（目前官方定的是每任务总共 15 分钟左右）内，查到真实、准确和有价值的信息，就真的有点勉强了**，更别提给到用户有效的结果（APPSO 的测试中有一半无法输出完整的结果）。

# LangChain

https://python.langchain.com/docs/introduction/

https://python.langchain.com/docs/versions/v0_3/

https://blog.csdn.net/2301_81940605/article/details/137627288

https://zhuanlan.zhihu.com/p/717120883：**LangChain 结构化输出：Pydantic+OpenAI 案例演示**

https://baijiahao.baidu.com/s?id=1802732876402932399&wfr=spider&for=pc

## LangChain 的基本概念

LangChain 能解决大模型的两个痛点，包括模型接口复杂、输入长度受限离不开自己精心设计的模块。根据LangChain 的最新文档，目前在 LangChain 中一共有六大核心组件，分别是模型的输入输出 (Model I/O)、数据连接 (Data Connection)、内存记忆（Memory）、链（Chains）、代理（Agent）、回调（Callbacks）。下面我们将分别讲述每一个模块的功能和作用。

目前，最新的官网中将数据连接部分改为了检索（Retrieval），但基本内容差异不大。

## **Model I/O**

模型是任何 LLM 应用中最核心的一点，LangChain 可以让我们方便的接入各种各样的语言模型，并且提供了许多接口，主要有三个组件组成，包括**模型（Models），提示词（Prompts）和解析器（Output parsers）**。

![image.png](Agent,%20MCP,%20pipeline%201c1e64a566218014aed3faf012d065bd/image%204.png)

### **1.Models**

LangChain 中提供了多种不同的语言模型，按功能划分，主要有两种。

- 语言模型（LLMs）：我们通常说的语言模型，给定输入的一个文本，会返回一个相应的文本。常见的语言模型有 GPT3.5，chatglm，GPT4All 等。
- 聊天模型（Chat model）：可以看做是封装好的拥有对话能力的 LLM，这些模型允许你使用对话的形式和其进行交互，能够支持将聊天信息作为输入，并返回聊天信息。这些聊天信息都是封装好的结构体，而非一个简单的文本字符串。常见的聊天模型有 GPT4、Llama 和 Llama2，以及微软云 Azure 相关的 GPT 模型。

### 2.Prompts

提示词是模型的输入，通过编写提示词可以和模型进行交互。LangChain 中提供了许多模板和函数用于模块化构建提示词，这些模板可以提供更灵活的方法去生成提示词，具有更好的复用性。根据调用的模型方式不同，提示词模板主要分为普通模板以及聊天提示词模板。

**1. 提示模板（PromptTemplate）**

提示模板是一种生成提示的方式，包含一个带有可替换内容的模板，从用户那获取一组参数并生成提示
提示模板用来生成 LLMs 的提示，最简单的使用场景，比如“我希望你扮演一个代码专家的角色，告诉我这个方法的原理 {code}”。
类似于 python 中用字典的方式格式化字符串，但在 langchain 中都被封装成了对象
**2. 聊天提示模板（ChatPromptTemplate）**

- 聊天模型接收聊天消息作为输入，这些聊天消息通常称为 Message，**和原始的提示模板不一样的是，这些消息都会和一个角色进行关联。**
- 在使用聊天模型时，建议使用聊天提示词模板，这样可以充分发挥聊天模型的潜力。

**3. Output parsers**
语言模型输出的是普通的字符串，有的时候我们可能想得到结构化的表示，比如 JSON 或者 CSV，一个有效的方法就是使用输出解析器。

输出解析器是帮助构建语言模型输出的类，主要实现了两个功能：

获取格式指令，是一个文本字符串需要指明语言模型的输出应该如何被格式化
解析，一种接受字符串并将其解析成固定结构的方法，可以自定义解析字符串的方式

## Data Connection

有的时候，我们希望语言模型可以从自己的数据中进行查询，而不是仅依靠自己本身输出一个结果。数据连接器的组件就允许你使用内置的方法去读取、修改，存储和查询自己的数据，主要有下面几个组件组成。

文档加载器（Document loaders）：连接不同的数据源，加载文档。
文档转换器（Document transformers）：定义了常见的一些对文档加工的操作，比如切分文档，丢弃无用的数据
文本向量模型（Text embedding models）：将非结构化的文本数据转换成一个固定维度的浮点数向量
向量数据库（Vector stores）：存储和检索你的向量数据
检索器（Retrievers）：用于检索你的数据

## Chains

只使用一个 LLM 去开发应用，比如聊天机器人是很简单的，但更多的时候，我们需要用到许多 LLM 去共同完成一个任务，这样原来的模式就不足以支撑这种复杂的应用。

为此 LangChain 提出了 Chain 这个概念，也就是**一个所有组件的序列，能够把一个个独立的 LLM 链接成一个组件，从而可以完成更复杂的任务**。举个例子，我们可以创建一个 chain，用于接收用户的输入，然后使用提示词模板将其格式化，最后将格式化的结果输出到一个 LLM。通过这种链式的组合，就可以构成更多更复杂的 chain。

在 LangChain 中有许多实现好的 chain，以最基础的 LLMChain 为例，它主要实现的就是接收一个提示词模板，然后对用户输入进行格式化，然后输入到一个 LLM，最终返回 LLM 的输出。

- 代码
    
    ```python
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    
    llm = OpenAI(temperature=0.9)
    prompt = PromptTemplate(
        input_variables=["product"],
        template="What is a good name for a company that makes {product}?",
    )
    
    from langchain.chains import LLMChain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Run the chain only specifying the input variable.
    print(chain.run("colorful socks"))
    
    from langchain.chat_models import ChatOpenAI
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
    )
    human_message_prompt = HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                template="What is a good name for a company that makes {product}?",
                input_variables=["product"],
            )
        )
    chat_prompt_template = ChatPromptTemplate.from_messages([human_message_prompt])
    chat = ChatOpenAI(temperature=0.9)
    chain = LLMChain(llm=chat, prompt=chat_prompt_template)
    print(chain.run("colorful socks"))
    ```
    

## Memory

大多数的 LLM 应用程序都会有一个会话接口，允许我们和 LLM 进行多轮的对话，并有一定的上下文记忆能力。但**实际上，模型本身是不会记忆任何上下文的，只能依靠用户本身的输入去产生输出。而实现这个记忆功能，就需要额外的模块去保存我们和模型对话的上下文信息，然后在下一次请求时，把所有的历史信息都输入给模型**，让模型输出最终结果。

而在 LangChain 中，提供这个功能的模块就称为 Memory，用于存储用户和模型交互的历史信息。在 LangChain 中根据功能和返回值的不同，会有多种不同的 Memory 类型，主要可以分为以下几个类别：

- 对话缓冲区内存（ConversationBufferMemory），最基础的内存模块，用于存储历史的信息
- 对话缓冲器窗口内存（ConversationBufferWindowMemory），**只保存最后的 K 轮对话**的信息，因此这种内存空间使用会相对较少
- 对话摘要内存（ConversationSummaryMemory），这种模式会**对历史的所有信息进行抽取，生成摘要信息，然后将摘要信息作为历史信息进行保存**。
- 对话摘要缓存内存（ConversationSummaryBufferMemory），这个和上面的作用基本一致，但是有最大 token 数的限制，**达到这个最大 token 数的时候就会进行合并历史信息生成摘要**

值得注意的是，对话摘要内存的设计出发点就是语言模型能支持的上下文长度是有限的（一般是 2048），超过了这个长度的数据天然的就被截断了。这个类会根据对话的轮次进行合并，默认值是 2，也就是**每 2 轮就开启一次调用 LLM 去合并历史信息**。

- 代码
    
    ```python
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.memory import ConversationBufferMemory
    
    # llm 
    llm = OpenAI(temperature=0)
    # Notice that "chat_history" is present in the prompt template
    template = """You are a nice chatbot having a conversation with a human.
    
    Previous conversation:
    {chat_history}
    
    New human question: {question}
    Response:"""
    prompt = PromptTemplate.from_template(template)
    # Notice that we need to align the `memory_key`
    memory = ConversationBufferMemory(memory_key="chat_history")
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True,
        memory=memory
    )
    conversation({"question": "hi"})
    
    ```
    

## Agents

代理的核心思想就是使用 LLM 去选择对用户的输入，应该使用哪个特定的工具去进行操作。这里的工具可以是另外的一个 LLM，也可以是一个函数或者一个 chain。在代理模块中，有三个核心的概念。

1、代理（Agent），依托于强力的语言模型和提示词，代理是用来决定下一步要做什么，其核心也是构建一个优秀的提示词。这个提示词大致有下面几个作用：

- 角色定义，给代理设定一个符合自己的身份
- 上下文信息，提供给他更多的信息来要求他可以执行什么任务
丰富的提示策略，增加代理的推理能力

2、工具（Tools），代理会选择不同的工具去执行不同的任务。工具主要**给代理提供调用自己的方法**，并且会**描述自己如何被使用**。工具的这两点都十分重要，如果你没有提供可以调用工具的方法，那么代理就永远完不成自己的任务；同时如果没有正确的描述工具，代理就不知道如何去使用工具。

3、工具包（Toolkits），LangChain 提供了工具包的使用，在一个工具包里通常包含 3-5 个工具。

Agent 技术是目前大语言模型研究的一个前沿和热点方向，但是目前受限于大模型的实际效果，仅 GPT 4.0 可以有效的开展 Agent 相关的研究。我们相信在未来，随着大模型性能的优化和迭代，Agent 技术应该能有更好的发展和前景。

## Callbacks

回调，字面解释是**让系统回过来调用我们指定好的函数**。在 LangChain 中就提供了一个这样的回调系统，允许你进行日志的打印、监控，以及流式传输等其他任务。通过直接在 API 中提供的回调参数，就可以简单的实现回调的功能。LangChain 内置了许多可以实现回调功能的对象，我们通常称为 **handlers**，用于定义在不同事件触发的时候可以实现的功能。

不管使用 Chains、Models、Tools、Agents，去调用 handlers，均通过是使用 **callbacks** 参数，这个参数可以在两个不同的地方进行使用：

- **构造函数中**，但它的作用域只能是该对象。比如下面这个 LLMChain 的构造函数可以进行回调，但这个回调函数对于链接到它的 LLM 模型是不生效的。
    - LLMChain(callbacks=[handler], tags=['a-tag'])
- **在 run()/apply() 方法中调用**，只有当前这一次请求才会相应这个回调函数，但是当前请求包含的子请求都会调用这个回调。比如，使用了一个 chain 去触发这个请求，连接到它的 LLM 模型也会调用这个回调。
    - chain.run(input, callbacks=[handler])

## **pydantic 的关键作用**

1. **数据验证和模型定义**：pydantic 的 `BaseModel` 类允许我们定义数据模型（如 `Author`、`Book`、`Library`），并自动处理数据的验证和转换。这确保了输入和输出的数据符合预期的格式和类型要求。
2. **与 LangChain 的集成**：通过 `PydanticOutputParser`，pydantic 可以直接用于解析来自语言模型的输出，将其转换为预定义的 pydantic 数据模型。这种集成简化了将自然语言处理结果映射到结构化数据的过程。

## **LangChain PydanticOutputParser**

`LangChain PydanticOutputParser` 是 LangChain 库中的一个工具，用于将语言模型生成的输出解析并转换为结构化的 Python 数据模型。它的作用可以分为以下几个关键点：

**1，结构化输出：**

- `PydanticOutputParser` 通过使用 pydantic 库中的数据模型（如 `BaseModel` 派生的类）来定义输出的预期结构。语言模型生成的文本通常是非结构化的，而 `PydanticOutputParser` 可以将这些文本转换为结构化的数据对象，如字典、列表，或者更复杂的嵌套对象。

**2，自动解析与验证：**

- 当语言模型返回的文本被解析时，`PydanticOutputParser` 会将该文本映射到预定义的 pydantic 模型上。这不仅包括**数据的提取，还包括数据的类型验证和格式转换**。如果解析到的数据不符合模型的定义，pydantic 会自动抛出错误或进行纠正。

**3，增强代码的可维护性与可靠性：**

- 通过明确定义输出的结构，开发者可以确保从语言模型获得的输出是一致的、可预测的。这减少了在处理输出时可能出现的错误，并使代码更容易维护和调试。

**4，与 LangChain 的无缝集成：**

- `PydanticOutputParser` 是 LangChain 提供的工具之一，它使得在使用 LangChain 提供的语言模型时，能够直接解析模型的响应，并将其集成到更大的应用程序中。它简化了从文本输出到结构化数据的过渡，使得在构建复杂应用程序时，处理数据更为便捷。

`LangChain PydanticOutputParser` 的作用是在语言模型生成的非结构化文本输出与结构化的数据模型之间架起桥梁，确保输出数据符合预定义的结构和格式，并且易于集成和使用

### **例一：结构化输出**

```python
# 定义作者模型
class Author(BaseModel):
    name: str = Field(description="Name of the author")  # 作者姓名，类型为字符串，并附有描述
    birth_year: int = Field(description="Year the author was born")  # 作者出生年份，类型为整数，并附有描述

# 定义图书模型
class Book(BaseModel):
    title: str = Field(description="Title of the book")  # 书籍标题，类型为字符串，并附有描述
    author: Author = Field(description="Author of the book")  # 书籍作者，类型为Author模型，并附有描述
    publication_year: int = Field(description="Year the book was published")  # 书籍出版年份，类型为整数，并附有描述

# 定义图书馆模型
class Library(BaseModel):
    name: str = Field(description="Name of the library")  # 图书馆名称，类型为字符串，并附有描述
    books: List[Book] = Field(description="List of books in the library")  # 图书列表，类型为Book模型的列表，并附有描述

# 使用PydanticOutputParser解析输出，指定模型为Library
parser = PydanticOutputParser(pydantic_object=Library)
```

### **例二：数据验证**

```python
# 定义作者类，使用 Pydantic 提供的数据验证功能
class Author(BaseModel):
    # 作者的姓名字段
    name: str = Field(description="Name of the author")
    # 作者的出生年份字段
    birth_year: int = Field(description="Year the author was born")

    # 验证出生年份是否在有效范围内
    @validator('birth_year')
    def valid_year(cls, v):
        # 确保年份在0到2023之间
        if v < 0 or v > 2023:
            raise ValueError("Birth year must be between 0 and 2023")
        return v
```

## 基于 LangChain 的应用

从上文中，我们了解了 LangChain 的基本概念，以及主要的组件，利用这些能帮助我们快速上手构建 app。LangChain 能够在很多使用场景中进行应用，包括但不限于：

- 个人助手和聊天机器人；能够记住和你的每一次互动，并进行个性化的交互
- 基于文档的问答系统；在特定文档上回答问题，可以减少大模型的幻觉问题
- 表格数据查询；提供了对结构化数据的查询功能，如 CSV，PDF，SQL，DataFrame 等
- API 交互；可以对接不同语言模型的API，并产生交互和调用
- 信息提取；从文本中提取结构化的信息，并输出
- 文档总结；利用 LLM 和 embedding 对长文档进行压缩和总结

而且在 github 上也有很多人开源了基于 LangChain 开发的开源应用:

[gpt4-pdf-chatbot](https://link.juejin.cn/?target=https%3A%2F%2Fgithub.com%2Fmayooear%2Fgpt4-pdf-chatbot-langchain)

[chatPDF](https://link.juejin.cn/?target=https%3A%2F%2Fgithub.com%2Fgabacode%2FchatPDF)

[Langchain-Chatchat](https://link.juejin.cn/?target=https%3A%2F%2Fgithub.com%2Fchatchat-space%2FLangchain-Chatchat)

## LangChain 的缺点

从实际使用体验来讲，这并不是一个完美的框架，也存在不少问题。

比如，LangChain 的提示词模板其实就是封装了字符串的类，但是 LangChain 中有很多类型的提示词模板，没有看出明显区别，而且也没有安全性，冗余比较多。而且有一些提升词是默认写好的，要修改的话，需要看源码才知道应该修改什么地方。

LangChain 内部封装了很多调用过程，debug 的过程比较困难。一旦出现了问题，排查所花费的时间可能会比较长。

之前有爆出过 LangChain 的代码在调用 python 去执行 agent 的时候会存在安全漏洞，有可能通过注入攻击的方式产生危险。但是这些类似的漏洞，需要官方去修复才可以，会给我们开发带来不必要的麻烦。

LangChain 的文档过于简单了，如果要实现一些官方没有提供的方法就需要动一些脑筋。

# MCP：基于阿里云百炼

https://www.aliyun.com/product/bailian

## MCP：允许大模型连接到外部服务

https://baijiahao.baidu.com/s?id=1816760345396023729&wfr=spider&for=pc

这个名为MCP的开放协议，目标是实现LLM应用程序与外部数据源和工具之间的无缝集成。因为允许LLM访问和利用外部资源，它的功能性和实用性都会大大增强。

MCP的核心遵循**客户端-服务器架构**，其中多个服务可以连接到任何兼容的客户端。

其中Host主机是启动连接的LLM应用程序（如Claude Desktop或IDE），客户端在主机应用程序内与服务器保持1V1连接，服务器则向客户端提供上下文、工具和提示。

![image.png](Agent,%20MCP,%20pipeline%201c1e64a566218014aed3faf012d065bd/image%205.png)

### MCP 客户端

MCP客户端是模型上下文协议(MCP)架构中的核心组件，负责建立和管理与MCP服务器的连接。它实现了协议的客户端部分，处理以下功能：

协议版本协商以确保与服务器的兼容性
能力协商以确定可用功能
消息传输和JSON-RPC通信
工具发现和执行
资源访问和管理
提示系统交互
可选功能如根目录管理和采样支持

MCP client 的工作流程如下：

MCP client 首先**从 MCP server 获取可用的工具列表**。
**将用户的查询连同工具描述通过 function calling 一起发送给 LLM**。
**LLM 决定是否需要使用工具以及使用哪些工具**。
如果需要使用工具，**MCP client 会通过 MCP server 执行相应的工具调用**。
工具调用的结果会被发送回 LLM。
LLM 基于所有信息生成自然语言响应。
最后将响应展示给用户。

### MCP 服务端

MCP服务器是模型上下文协议(MCP)架构中的基础组件，为客户端提供工具、资源和功能。它实现了协议的服务器端，负责：

暴露客户端可以发现和执行的工具
管理基于URI的资源访问模式
**提供提示模板并处理提示请求**
支持与客户端的能力协商
实现服务器端协议操作
管理并发客户端连接
提供结构化日志和通知

具体来说，客户端是Claude Desktop、IDE或AI工具等应用程序，服务器则是公开数据源的轻型适配器。

![image.png](Agent,%20MCP,%20pipeline%201c1e64a566218014aed3faf012d065bd/image%206.png)

- MCP Hosts: 像 Claude Desktop、IDEs 或 AI 工具这样的程序，它们希望通过 MCP 访问资源
- **MCP Clients: 维护与servers 1:1 连接的协议客户端**
- MCP Servers: 轻量级程序，通过标准化的 Model Context Protocol 暴露特定功能
- Local Resources: 你的计算机资源（数据库、文件、服务），MCP 服务器可以安全地访问这些资源
- Remote Resources: 通过互联网可用的资源（例如，通过 APIs），MCP 服务器可以连接到这些资源

**客户端和服务器之间的连接，是通过stdio或SSE等传输方式建立的**。传输层处理消息成帧、传递和错误处理。MCP的强大之处在于，它**通过相同的协议同时处理本地资源（数据库、文件、服务）和远程资源（Slack或GitHub等API）**。**目前，MCP仅在本地受支持，服务器必须在自己的计算机上运行。**但是，Anthropic正在通过企业级身份验证构建远程服务器支持，以便团队可以在整个组织中安全地共享其上下文来源。**Anthropic的目标就是，构建一个AI通过单一、优雅的协议就能连接到任何数据源的世界，MCP就是其中的通用转换器。**只要将MCP集成到客户端一次，就可以连接到任何地方的数据源。

如今，各大AI公司都在尝试不同的方法：谷歌依赖于自己的内部服务：搜索、Gmail、日历；微软正在尝试使用其安全的Office Copilot应用程序获取企业用户上下文；苹果试图通过隐私保护继续获取用户上下文，同时允许访问ChatGPT进行高级查询；OpenAI已经尝试了GPT，现在正在尝试通过ChatGPT桌面应用程序连接应用程序。ChatGPT的愿景是通过屏幕共享控制用户桌面。

Anthropic与OpenAI的处境类似，因为他们没有现成的用户上下文。他们的解决方案似乎是，提供一个干净的协议，通过该协议，任何网站，API或系统资源都可以被任何AI访问。也就意味着，构建一次，让AI无处不在。

专注AI原生平台Ada CEO Mike Murchison深入分析了MCP三个重要影响：

- 应用集成护城河正被削弱。随着AI模型能够原生接入第三方数据源，**应用程序之前建立的独特数据集成优势正在消失**
- 前沿模型在「预集成」到各种内容商店能力上展开竞争。未来，**各大AI模型会竞相提供与不同内容库的原生连接能力**
- 将会看到**前沿AI模型与特定数据源公司建立独家的合作关系**

### **工作流程**

作者给出了一个案例：利用MCP实现Claude Desktop访问查询本地SQLite数据库，来说明整体的交互工作流程。

这个例子中使用MCP与Claude Desktop交互时的具体流程如下：

- Claude Desktop在启动时连接到用户配置的MCP服务器
- Claude Desktop确定哪个MCP服务器可以提供帮助，本例中就是SQLite
- 通过协议来确定功能的实现方式
- 从MCP服务器（SQLite）请求数据或操作

![image.png](Agent,%20MCP,%20pipeline%201c1e64a566218014aed3faf012d065bd/image%207.png)

# A2A：多智能体协议

https://mp.weixin.qq.com/s/WWF_ffC4WPwfTDaYy5HhsQ

AI 智能体通过自主处理许多日常重复性或复杂任务，为帮助人们提高生产力提供了独特的机会。如今，企业越来越多地构建和部署自主智能体，以帮助扩展、自动化和增强整个工作场所的流程——从订购新笔记本电脑，到协助客户服务代表，再到辅助供应链规划。

为了最大化智能体 AI 的效益，这些**智能体必须能够在跨越孤立数据系统和应用程序的动态、多智能体生态系统中进行协作**，这一点至关重要。使智能体能够相互互操作，即使它们是由不同的供应商或在不同的框架中构建的，也将提高自主性并倍增生产力收益，同时降低长期成本。

今天，我们推出了一个名为 **Agent2Agent (A2A)** 的全新开放协议，得到了超过 50 家技术合作伙伴的支持和贡献，例如 Atlassian、Box、Cohere、Intuit、Langchain、MongoDB、PayPal、Salesforce、SAP、ServiceNow、UKG 和 Workday；以及领先的服务提供商，包括 Accenture、BCG、Capgemini、Cognizant、Deloitte、HCLTech、Infosys、KPMG、McKinsey、PwC、TCS 和 Wipro。A2A 协议将允许 AI 智能体在各种企业平台或应用程序之上相互通信、安全地交换信息并协调行动。我们相信 A2A 框架将为客户带来显著价值，他们的 AI 智能体现在将能够在其整个企业应用程序资产中协同工作。

**A2A 是一个开放协议，它补充了 Anthropic 的模型上下文协议 (MCP)，后者为智能体提供有用的工具和上下文。借鉴 Google 在扩展智能体系统方面的内部专业知识，我们设计了 A2A 协议，以解决我们在为客户部署大规模、多智能体系统时发现的挑战。**A2A 使开发人员能够构建能够连接到使用该协议构建的任何其他智能体的智能体，并为用户提供了**组合来自不同提供商的智能体**的灵活性。至关重要的是，企业受益于一种跨不同平台和云环境管理其智能体的标准化方法。我们相信这种普遍的互操作性对于充分实现协作式 AI 智能体的潜力至关重要。

**A2A 设计原则**A2A 是一个开放协议，为智能体提供了一种标准的协作方式，无论其底层框架或供应商如何。在与我们的合作伙伴设计该协议时，我们遵循了五个关键原则：

1. **拥抱智能体能力：** A2A 专注于使智能体能够以其自然的、非结构化的方式进行协作，即使它们不共享内存、工具和上下文。我们正在实现真正的多智能体场景，而不将智能体限制为“工具”。
2. **基于现有标准构建：** 该协议建立在现有的、流行的标准之上，包括 HTTP、SSE、JSON-RPC，这意味着它更容易与企业日常使用的现有 IT 堆栈集成。
3. **默认安全：** A2A 旨在支持企业级身份验证和授权，在发布时与 OpenAPI 的身份验证方案保持同等水平。
4. **支持长时间运行的任务：** 我们将 A2A 设计得非常灵活，支持各种场景，无论是快速完成任务还是需要数小时甚至数天（当有人工介入时）的深度研究，它都能出色完成。在此过程中，A2A 可以向其用户提供实时反馈、通知和状态更新。
5. **模态无关：** 智能体的世界不仅仅局限于文本，这就是为什么我们将 A2A 设计为支持各种模态，包括音频和视频流。

## **A2A 工作原理**

（一个图示流程图，展示了远程智能体和客户端智能体之间的数据流，以实现安全协作、任务和状态管理、用户体验协商以及能力发现）

![A2A 工作原理](Agent,%20MCP,%20pipeline%201c1e64a566218014aed3faf012d065bd/image%208.png)

A2A 工作原理

A2A 促进了“**客户端”智能体和“远程”智能体之间的通信**。**客户端智能体负责制定和传达任务，而远程智能体负责执行这些任务**，以尝试提供正确的信息或采取正确的行动。这种交互涉及几个关键功能：

- **能力发现：** 智能体可以使用 JSON 格式的“智能体名片”（Agent Card）来宣传其能力，允许客户端智能体识别能够执行任务的最佳智能体，并利用 A2A 与远程智能体通信。
- **任务管理：** 客户端和远程智能体之间的**通信以任务完成为导向**，其中智能体致力于满足最终用户的请求。**这个“任务”对象由协议定义并具有生命周期**。它可以立即完成，或者对于长时间运行的任务，每个智能体可以相互通信，以就完成任务的最新状态保持同步。**任务的输出称为“工件”（artifact）。**
- **协作：** 智能体可以相互发送消息以传达上下文、回复、工件或用户指令。
- **用户体验协商：** 每条消息都包含“部分”（parts），这是一个完全形成的内容片段，例如生成的图像。**每个部分都有指定的内容类型，允许客户端和远程智能体协商所需的正确格式，**并明确包括对用户 UI 能力的协商——例如，iframe、视频、Web 表单等。