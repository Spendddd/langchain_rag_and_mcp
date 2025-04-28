from huggingface_hub import snapshot_download
 
snapshot_download(
    repo_id="BAAI/bge-reranker-base",
    local_dir="bge-reranker-base/",
)