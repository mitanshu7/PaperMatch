# Run milvus db app to search vectors for similarity in arxiv abstract embeddings

## Usage:
0. Have podman installed on either Oracle or Fedora.
1. `bash standalone_embed.sh start`
2. `python prepare_milvus.py`
3. `python app.py`
