# Load Notion page as a markdownfile file
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
from chineseRecursiveTextSplitter import ChineseRecursiveTextSplitter
# from milvus import default_server
# from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma, FAISS

def add_docs(docs, all_splits, markdown_splitter, text_splitter):
    for doc in docs:
        # print(doc)
        if type(doc) == list:
            add_docs(doc, all_splits, markdown_splitter, text_splitter)
        else:
            md_file=doc.page_content
            # Let's create groups based on the section headers in our page
            md_header_splits = markdown_splitter.split_text(md_file)
            all_splits.extend(text_splitter.split_documents(md_header_splits))

def load_notions():
    # default_server.start()
    path='notion_docs'
    loader = NotionDirectoryLoader(path)
    docs = loader.load()
    headers_to_split_on = [
        ("#", "章"),
        ("##", "节"),
        ("###", "小节"),
    ]
    all_splits = []
    # Define our text splitter
    chunk_size = 1000
    chunk_overlap = 150
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = ChineseRecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    add_docs(docs, all_splits, markdown_splitter, text_splitter)
    # print(all_splits[:10])
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5", encode_kwargs={"normalize_embeddings": True})

    # 修改为使用 Faiss 进行向量存储
    vectordb = FAISS.from_documents(
        documents=all_splits,
        embedding=embeddings
    )

    # 保存 Faiss 索引到本地
    vectordb.save_local("./storage/faiss_index_notion")


    # 后续使用时可以通过以下方式加载已有索引
    # vectordb = Faiss.load_local("./storage/faiss_index", embeddings)

if __name__ == "__main__":
    load_notions()
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-zh-v1.5", encode_kwargs={"normalize_embeddings": True})
    vectordb = FAISS.load_local("./storage/faiss_index_notion", embeddings, allow_dangerous_deserialization=True)
    print(vectordb.similarity_search_with_score("llama"))