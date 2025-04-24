# langchian_rag_and_mcp
一个基于langchain的rag系统，实现了基于向量数据库的rag系统。

## 实现细节

基于python3.11，langchain、langgraph各种相关库，faiss-cpu，langchain-mcp-tools等主要库实现项目，可以参照requirements.txt文件的内容进行虚拟环境的构建。

rag数据库的数据来源是中文为主的notion（md文件格式）和pdf，基于继承RecursiveCharacterTextSplitter类的自定义类ChineseRecursiveTextSplitter实现了中文文本分割，基于marker工具（github开源项目）实现了pdf转md的功能，在md文件基础上利用MarkdownHeaderTextSplitter类实现了带有标题元数据的文本分割。

基于FAISS实现了向量数据库的构建和数据存储，基于BAAI/bge-base-zh-v1.5实现文本向量化和相似度检索。

基于deepseek-chat实现了大模型的调用，基于langgraph实现了rag系统的流程编排。

基于flask+vue实现了rag系统的前端展示。

## 使用方法

1、通过load_notion_docs.py和load_pdf.py加载notion和pdf数据到向量数据库

2、修改agent.py中的llm和embedding_model为自己的大模型和embedding模型

3、新建.env文件，添加你自己的DEEPSEEK_API_KEY（或其他大模型）、TAVILY_API_KEY

4、运行agent.py


## 待改进

1、【前端】未实现历史记录功能，点击超链接/刷新页面后会重新加载页面，清空原有回答；

2、【前端】“提交”按钮的逻辑有待优化；

3、【前端/后端】“停止生成”按钮的逻辑有待优化，目前未真正实现按下停止按钮立刻停止后台任务的目的；

4、【后端】未实现当前对话历史查询记忆功能；

5、【前端/后端】未实现历史对话存储功能。
