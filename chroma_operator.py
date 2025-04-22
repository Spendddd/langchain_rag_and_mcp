import os
import chromadb
# from data.get_news_from_api import get_news_from_api
from chromadb import QueryResult
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings


class VectorDBConnector:
    def __init__(self, collection_name: str = "LLMDoc", path="./storage/chromadb_client"):
        # client.reset()
        self.db_path = path
        self.collection_name = collection_name

        client = chromadb.PersistentClient(path=self.db_path)

        self.client = client
        # 如果要设置chromadb.utils 中 embedding_functions 模块支持的embedding model，在get_or_create_collection中添加参数 embedding_function=embedding_functions.指定EmbeddingFunction()
        # 如果要设置自定义的embedding model，在get_or_create_collection中添加参数 embedding_function=MyEmbeddingFunction()
        # 同时添加代码:
        """
        def my_embedding_function(texts):
            # 这里可以使用任何您喜欢的 embedding 模型
            embeddings = your_embedding_model.encode(texts)
            return embeddings
        """
        # 默认embedding模型是 all-MiniLM-L6-v2
        self.embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5", encode_kwargs={"normalize_embeddings": True})
        self.collection = self.client.get_or_create_collection(name=self.collection_name, embedding_function=self.embeddings)

    def add_documents(self, documents, metadata, ids=None):
        """向 collection 中添加文档与向量"""
        self.collection.add(
            documents=documents,
            # we handle tokenization, embedding, and indexing automatically. You can skip that and add your own embeddings as well
            metadatas=metadata,  # filter on these!
            ids=ids  # 每个文档的 id
        )

    def get_collection_count(self):
        return self.client.get_collection(name=self.collection_name).count()

    def get_all_documents(self):
        results = self.collection.get()
        return results

    def delete_all_documents(self):
        print("Delete total {} informations".format(self.get_collection_count()))
        self.client.delete_collection(name=self.collection.name)


class VectorStoreRetriever(BaseRetriever):
    """
  基于向量数据库的 Retriever 实现
  """

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """Retriever 的同步实现"""

        vector_db = VectorDBConnector()
        query_result: QueryResult = vector_db.search(query, top_n=5)
        if query_result:
            docs = query_result["documents"][0]
            if docs:
                return [Document(page_content=doc) for doc in docs]
        return []


if __name__ == "__main__":
    # setup Chroma in-memory, for easy prototyping. Can add persistence easily!
    collection_name = "LLMDoc"
    vector_db = VectorDBConnector(collection_name=collection_name)

    vector_db.delete_all_documents()
    start_id = vector_db.get_collection_count()
    ids = ["id"+str(i) for i in range(start_id, start_id + 10)]