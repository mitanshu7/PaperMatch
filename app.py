# Import required libraries
import gradio as gr
from pymilvus import MilvusClient
import json
import numpy as np
import ast

################################################################################

# Define client
client = MilvusClient("http://localhost:19530")


# Single vector search
def predict(vector_string, limit):

    # Parse the string to a Python list
    vector_list = ast.literal_eval(vector_string)

    # Now, convert it to a NumPy array with dtype=float
    vector = np.array(vector_list, dtype=float)

    # print(type(vector))
    # vector = np.array(vector, dtype=float)
    # print(type(vector))

    result = client.search(
        collection_name="arxiv_abstracts", # Replace with the actual name of your collection
        # Replace with your query vector
        data=[vector],
        limit=limit, # Max. number of search results to return
        search_params={"metric_type": "COSINE"} # Search parameters
    )

    # Convert the output to a formatted JSON string
    result_json = json.dumps(result, indent=4)

    return result_json

# Gradio app interface
gradio_app = gr.Interface(
    predict,
    inputs=[gr.Textbox(placeholder="Insert Vector of dimensions 1024", label='Vector'), "slider"], 
    outputs="json",
    title="Milvus vector similarity search",
    description="Search the embedded arxiv abstracts (till Aug-2024). \nEmbedding model used: mixedbread-ai/mxbai-embed-large-v1."
)

if __name__ == "__main__":
    gradio_app.launch()