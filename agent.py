#!/usr/bin/env python
# coding: utf-8

# In[45]:


# set api key  
import os
from dotenv import load_dotenv
load_dotenv()


# In[46]:


# 导入相关库
from langchain.tools.retriever import create_retriever_tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers import MergerRetriever, BM25Retriever, EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.vectorstores import FAISS

# In[47]:


# 定义rag检索工具
embeddings = HuggingFaceEmbeddings(model_name="./local_models/bge-base-zh-v1.5", encode_kwargs={"normalize_embeddings": True})
retrieverList = []
collections=["notion", "pdf"]
all_docs = []
# type = "indexivfflat"
type = "indexflatl2"
# type = "indexivfpq"
for collection in collections:
    vectordb = FAISS.load_local(
        "./storage/faiss_"+type+"_"+collection,
        embeddings,
        allow_dangerous_deserialization=True
    )
    ids = []
    for index, docstore_id in vectordb.index_to_docstore_id.items():
        ids.append(docstore_id)
    all_docs.extend(vectordb.get_by_ids(ids))
    retriever = vectordb.as_retriever(search_kwargs={"k": 20}, search_type="similarity", search_distance="cosine", )
    retrieverList.append(retriever)

multi_retriever = MergerRetriever(retrievers=retrieverList)

# 定义一个bm25retriever，用于检索文本中的关键词
bm25_retriever = BM25Retriever.from_documents(all_docs, k=20)

# 定义一个ensemble retriever，用于混合检索多个指定向量数据库的工具
ensemble_retriever = EnsembleRetriever(
    retrievers=[
        multi_retriever,
        bm25_retriever
    ],
    weights=[0.6, 0.4]  
)

# 定义一个reranker，用于对检索结果进行排序
reranker_model = "./local_models/bge-reranker-base"
model_kwargs = {'device': 'cpu'}
reranker = HuggingFaceCrossEncoder(
    model_name=reranker_model,
    model_kwargs=model_kwargs
)

# 定义一个compression retriever，用于对检索结果进行压缩
compressor = CrossEncoderReranker(model=reranker, top_n=10)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever,
    top_n=10,
)

vector_search_tool = create_retriever_tool(
    name="VectorStoreSearch",
    description="用于混合检索多个指定向量数据库并重排检索结果的工具",
    retriever=compression_retriever,
)


# In[48]:


# 定义网页检索工具
from langchain_community.tools.tavily_search import TavilySearchResults
web_search_tool = TavilySearchResults(k=10)


# In[49]:


# tools = [vector_search_tool, web_search_tool]
tools = [vector_search_tool]


# In[50]:


# 定义一个logger
import logging

def init_logger() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,  # logging.DEBUG,
        format='\x1b[90m[%(levelname)s]\x1b[0m %(message)s'
    )
    return logging.getLogger()


# In[51]:


from langgraph.prebuilt import create_react_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_tools import convert_mcp_to_langchain_tools
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from langchain.tools.render import render_text_description_and_args  
from langchain_core.messages import AIMessage, ToolMessage
import asyncio
from flask import Flask, request, jsonify, render_template
import json
app = Flask(__name__)
current_task = None

async def run(query) -> dict:
    global current_task
    try:
        mcp_configs = {
            "tavily-mcp": {
                "command": "npx",
                "args": ["-y", "tavily-mcp@0.1.2"],
                "env": {
                    "TAVILY_API_KEY": os.environ["TAVILY_API_KEY"]
                }
            },
            # 下面的实现方式和上面等价，下面利用sse url连接实现，上面利用npm安装实现
            # "tavily-mcp": {
            #     "type": "sse",
            #     "url": "https://mcp.api-inference.modelscope.cn/sse/d9f1df868bac4e"
            #     # 请利用魔搭社区MCP广场Tavily智搜工具输入你的API Key，点击连接测试，复制生成的SSE URL，填入上面的url中
            # }
        }

        mcptools, cleanup = await convert_mcp_to_langchain_tools(
            mcp_configs,
            init_logger()
        )

        tools.extend(mcptools)
        model = init_chat_model(model="deepseek-chat", model_provider="deepseek", temperature=0.1)  

        system_prompt = """请尽可能以有帮助和准确的方式回应人类。你可以使用以下工具：{tools}。
        优先尝试使用向量数据库搜索工具在向量数据库中搜索相关信息，
        在当前行动轮次，如果在向量数据库中检索了1次仍未发现有效的相关信息或你认为相关信息不足时，请使用其他工具进行检索，
        注意每轮行动中向量数据库检索最多只能使用1次，其他工具最多只能使用2次。
        使用JSON对象通过提供action key（工具名称）和action_input key（工具输入）来指定工具。 
        有效的“action”值包括：“Final Answer”或者{tool_names}。每个JSON对象只提供一个action。"""

        # human prompt  
        human_prompt = """用户问题：{messages}。
        注意：始终以JSON对象进行响应。"""

        # create prompt template
        prompt = ChatPromptTemplate.from_messages(  
            [  
                ("system", system_prompt),  
                ("human", human_prompt),  
            ]  
        )
        prompt = prompt.partial(  
            tools=render_text_description_and_args(list(tools)),  
            tool_names=", ".join([t.name for t in tools]),  
        )

        agent = create_react_agent(
            model,
            tools,
            prompt=prompt
        )
        current_task = asyncio.current_task()

        async def inner_run():
            result = await agent.ainvoke({"messages": query})
            final_answer = ""
            intermediate_steps = []
            messages = result['messages']
            for i, message in enumerate(messages):
                if isinstance(message, AIMessage):
                    if message.additional_kwargs.get('tool_calls'):
                        tool_call = message.additional_kwargs['tool_calls'][0]
                        tool_name = tool_call['function']['name']
                        if i + 1 < len(messages) and isinstance(messages[i + 1], ToolMessage):
                            tool_content = messages[i + 1].content
                            intermediate_steps.append(({"tool": tool_name}, tool_content))
                    else:
                        content = message.content.strip()
                        if content.startswith("```json") and content.endswith("```"):
                            content = content[7:-3].strip()
                        try:
                            content_json = json.loads(content)
                            if content_json.get('action') == 'Final Answer':
                                final_answer = content_json.get('action_input', '')
                        except json.JSONDecodeError:
                            continue
            if final_answer == "":
                final_answer = messages[-1].content
            return {
                "final_answer": final_answer,
                "intermediate_steps": intermediate_steps
            }

        try:
            return await inner_run()
        except asyncio.CancelledError:
            print("任务已被取消")
            return {
                "error": "任务已被用户取消"
            }
    except Exception as e:
        return {
            "error": str(e)
        }
    finally:
        if cleanup:
            try:
                await cleanup()
            except Exception as e:
                print(f"Cleanup failed: {e}")
        current_task = None

@app.route('/') 
def index(): 
    return render_template('index.html')

@app.route('/query', methods=['POST'])
async def query():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({"error": "Missing query"}), 400
    result = await run(query)
    return jsonify(result)

@app.route('/cancel', methods=['POST'])
async def cancel():
    global current_task
    if current_task:
        current_task.cancel()
        return jsonify({"message": "任务已取消"}), 200
    else:
        return jsonify({"message": "没有正在运行的任务"}), 400

if __name__ == '__main__':
    # todo:实现记忆功能、研究怎么真正中止任务
    import uvicorn
    from uvicorn.middleware.wsgi import WSGIMiddleware
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app)
    # 使用 WSGIMiddleware 包装 Flask 应用
    asgi_app = WSGIMiddleware(app)
    uvicorn.run(asgi_app, host='127.0.0.1', port=5000)
    # llama和transformer有什么区别？
    # function calling是什么?
    # FAISS是什么？