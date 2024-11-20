#!/bin/bash

source /home/milvus/miniforge3/bin/activate search_arxiv

python /home/milvus/PaperMatch/prepare_milvus.py >> /home/milvus/PaperMatch/update_milvus.log 2>&1

systemctl --user restart papermatch.service
