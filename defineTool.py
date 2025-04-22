# create tool call for vector search and web search  
from langchain.tools import tool  
from langchain_mcp_tools import convert_mcp_to_langchain_tools
from langchain.tools.retriever import create_retriever_tool
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.retrievers import MergerRetriever
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
load_dotenv()

def create_tools():
    """创建工具"""
    # 向量检索工具
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5", encode_kwargs={"normalize_embeddings": True})
    retrieverList = []
    collections=["LLMDoc", "LLMBook"]
    for collection in collections:
        vectordb = Chroma(
            persist_directory="./storage/chroma_db",
            embedding_function=embeddings,
            collection_name=collection
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": 5}, search_type="similarity", search_distance="cosine", )
        retrieverList.append(retriever)
    
    multi_retriever = MergerRetriever(retrievers=retrieverList)
    vector_search_tool = create_retriever_tool(
        name="VectorStoreSearch",
        description="用于检索向量数据库的工具",
        retriever=multi_retriever,
        return_direct=True,
    )
    
    # 网页检索工具
    web_search_tool = TavilySearchResults(k=10)
    # web_search_tool = create_retriever_tool(
    #     name="WebSearch",
    #     description="用于检索网页的工具",
    #     retriever=web_search_tool_func,
    #     return_direct=True,
    # )
    
    # mcp协议检索工具
    # mcp_tool = create_retriever_tool(
    #     name="MCPWebSearchTool",
    #     description="用于通过mcp协议检索网页的工具",
    #     retriever=mcp_tool_func,
    #     return_direct=True,
    # )
    
    return [vector_search_tool, web_search_tool]

@tool  
def vector_search_tool_func(query: str, llm) -> str:
    """用于检索向量数据库的工具
    
    Args:
        query (str): 用户输入的问题
        llm: LLM模型，用于检索
    Returns:
        str: 检索到的结果
    """
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5", encode_kwargs={"normalize_embeddings": True})
    retrieverList = []
    collections=["LLMDoc", "LLMBook"]
    for collection in collections:
        vectordb = Chroma(
            persist_directory="./storage/chroma_db",
            embedding_function=embeddings,
            collection_name=collection
        )
        retriever = vectordb.as_retriever(search_kwargs={"k": 5}, search_type="similarity", search_distance="cosine", )
        retrieverList.append(retriever)
    
    multi_retriever = MergerRetriever(retrievers=retrieverList)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=multi_retriever)  
    return qa_chain.run(query) 
    # return tool

@tool
def web_search_tool_func(query: str) -> str:
    """用于检索网页的工具
    Args:
        query (str): 用户输入的问题
    Returns:
        str: 检索到的结果
    """
    web_search_tool = TavilySearchResults(k=10)
    return web_search_tool.run(query)

@tool
async def mcp_tool_func() -> str:
    """用于通过mcp协议检索网页的工具"""
    mcp_servers = {
        "tavily-mcp": {
            "command": "npx",
            "args": ["-y", "tavily-mcp@0.1.4"],
            "env": {
                "TAVILY_API_KEY": os.getenv["TAVILY_API_KEY"],
            },
            "disabled": False,
            "autoApprove": []
        }
    }
    tools, cleanup = await convert_mcp_to_langchain_tools(
        mcp_servers
    )
    return tools[0]


if __name__ == "__main__":

    model = ChatDeepSeek(model="deepseek-reasoner", api_key=os.environ["DS_API_KEY"])  

    # 测试向量检索工具
    vector_search_tool = vector_search_tool_func("你能告诉我关于深度学习的最新进展吗？", model)
    print(vector_search_tool)
    
    # 测试网页检索工具
    web_search_tool = web_search_tool_func("你能告诉我关于深度学习的最新进展吗？")
    print(web_search_tool)
    
    # # 测试mcp协议检索工具
    # mcp_tool = mcp_tool_func()
    # print(mcp_tool)