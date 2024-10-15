# Import required libraries
from huggingface_hub import snapshot_download # To download vector data
from pymilvus import MilvusClient, DataType 
import requests
from time import sleep
from glob import glob

################################################################################
# Download vector dataset
# https://huggingface.co/docs/huggingface_hub/v0.24.6/en/package_reference/file_download#huggingface_hub.snapshot_download

# Setup transaction details
repo_id = "bluuebunny/arxiv_abstract_embedding_mxbai_large_v1_milvus"
# repo_id = "bluuebunny/tmp"
repo_type = "dataset"
local_dir = "volumes/milvus"
allow_patterns = "*.parquet"

# Download the repo
snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=local_dir, allow_patterns=allow_patterns)


################################################################################
# Create collection
# Define client
client = MilvusClient("http://localhost:19530")

# Drop any of the pre-existing collections
# Need to drop it because otherwise milvus does not check for (and keeps)
# duplicate records
client.drop_collection(
    collection_name="arxiv_abstracts"
)

# Dataset schema
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True
)

# Add the fields to the schema
schema.add_field(field_name="id", datatype=DataType.VARCHAR, is_primary=True, max_length=512)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)

print("Issues with scheme: ", schema.verify())

# Create a collection
client.create_collection(
    collection_name="arxiv_abstracts",
    schema=schema
)

################################################################################
# Create import job
# https://milvus.io/docs/import-data.md
# Gather files

files = glob('volumes/milvus/data/*.parquet')
files = [ ["/var/lib/milvus/data/" + i.split('/')[-1]] for i in files ]

# Define the API endpoint
job_url = f"http://localhost:19530/v2/vectordb/jobs/import/create"

# Define the headers
headers = {
    "Content-Type": "application/json"
}

# Define the data payload
job_url_data = {
    "files": files,
    "collectionName": "arxiv_abstracts"
}

# Make the POST request
job_response = requests.post(job_url, headers=headers, json=job_url_data)
job_json = job_response.json()

# Print the response
print("Job details:")
print(job_response.status_code)
print(job_json)

# Extract jobId
job_id = job_json['data']['jobId']

# Periodically check on import status
progress_url = "http://localhost:19530/v2/vectordb/jobs/import/get_progress"

progress_url_data = {
    "jobId": f"{job_id}"
}

while True:

    print('*'*80)

    # Sleep a bit
    seconds = 10
    print(f"Sleeping for {seconds} seconds")
    sleep(seconds)

    
    # Make the POST request
    progress_response = requests.post(progress_url, headers=headers, json=progress_url_data)

    progress_json = progress_response.json()
    # print(progress_json)

    progress_percent = progress_json['data']['progress']
    progress_state = progress_json['data']['state']

    if progress_state == 'Pending' or progress_state == 'Importing':

        print(f"Job: {progress_state}.")
        print(f"Finised: {progress_percent}%.")

    elif progress_state == 'Completed':

        print(f"Job: {progress_state}.")
        print(f"Imported {progress_json['data']['totalRows']} rows.")

        break

    elif progress_state == 'Failed':

        print(f"Job: {progress_state}.")
        print(progress_json)

        print("Exiting...")
        exit()

################################################################################
# Create index

# Set up the index parameters
index_params = MilvusClient.prepare_index_params()

# Add an index on the vector field.
index_params.add_index(
    field_name="vector",
    metric_type="COSINE",
    index_type="FLAT",
    index_name="vector_index",
)

print("Creating Index file.")

# Create an index file
res = client.create_index(
    collection_name="arxiv_abstracts",
    index_params=index_params,
    sync=True # Wait for index creation to complete before returning. 
)

print(res)

print("Listing indexes.")

# List indexes
res = client.list_indexes(
    collection_name="arxiv_abstracts"
)

print(res)

print("Describing Index.")

# Describe index
res = client.describe_index(
    collection_name="arxiv_abstracts",
    index_name="vector_index"
)

print(res)

################################################################################

# Load the collection

print("Loading Collection")

client.load_collection(
    collection_name="arxiv_abstracts",
    replica_number=1 # Number of replicas to create on query nodes. 
)

res = client.get_load_state(
    collection_name="arxiv_abstracts"
)

print("Collection load state:")
print(res)


