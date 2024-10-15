# Import required libraries
import gradio as gr
from pymilvus import MilvusClient
import numpy as np
import arxiv
from mixedbread_ai.client import MixedbreadAI
from dotenv import dotenv_values
import re
from functools import cache
import pandas as pd

################################################################################
# Configuration

# Define Milvus client
milvus_client = MilvusClient("http://localhost:19530")

# Construct the Arxiv API client.
arxiv_client = arxiv.Client(page_size=1, delay_seconds=1)

# Import secrets
config = dotenv_values(".env")

# Setup mxbai
mxbai_api_key = config["MXBAI_API_KEY"]
mxbai = MixedbreadAI(api_key=mxbai_api_key)

################################################################################

# Function to search ArXiv by ID
@cache
def fetch_arxiv_by_id(arxiv_id):

    # Search for the paper using the Arxiv API
    search = arxiv.Search(id_list=[arxiv_id])

    try:

        # Fetch the paper metadata using the Arxiv API
        paper = next(arxiv_client.results(search), None)

        # Extract the relevant metadata from the paper object
        return {
                "Title": paper.title,
                "Authors": ", ".join([str(author) for author in paper.authors]),
                "Abstract": paper.summary,
                "URL": paper.pdf_url
            }

    except Exception as e:

        # Raise an exception if the request was not successful
        raise gr.Error( f"Failed to fetch metadata for ID '{arxiv_id}'. Error: {e}")

################################################################################
# Function to embed text
@cache
def embed(text):

    res = mxbai.embeddings(
    model='mixedbread-ai/mxbai-embed-large-v1',
    input=text,
    normalized=True,
    encoding_format='float',
    truncation_strategy='end'
    )

    vector = np.array(res.data[0].embedding)

    return vector

################################################################################
# Single vector search

def search(vector, limit):

    result = milvus_client.search(
        collection_name="arxiv_abstracts", # Replace with the actual name of your collection
        # Replace with your query vector
        data=[vector],
        limit=limit, # Max. number of search results to return
        search_params={"metric_type": "COSINE"}, # Search parameters
        output_fields=["$meta"] # Output fields to return
    )

    # returns a list of dictionaries with id and distance as keys
    return result[0]

################################################################################
# Function to 

def fetch_all_details(search_results):

    all_details = []

    for search_result in search_results:

        paper_details = search_result['entity']

        paper_details['Similarity Score'] = np.round(search_result['distance']*100, 2)

        all_details.append(paper_details)

    # Convert to dataframe
    df = pd.DataFrame(all_details)

    # Convert to HTML table and return
    html = df.to_html(render_links=True, index=False)

    return html

################################################################################

# Function to handle the UI logic
@cache
def predict(input_type, input_text, limit):

    # When input is arxiv id
    if input_type == "ArXiv ID":

        # Check if input is empty
        if input_text == "":
            raise gr.Error("Please enter a ArXiv ID", 10)

        # Search if id is already in database
        id_in_db = milvus_client.get(collection_name="arxiv_abstracts",ids=[input_text])

        # If the id is already in database
        if bool(id_in_db):

            # Get the vector
            abstract_vector = id_in_db[0]['vector']

        else:

            # Search arxiv for paper details
            arxiv_json = fetch_arxiv_by_id(input_text)

            # Embed abstract
            abstract_vector = embed(arxiv_json['Abstract'])

        # Search database
        search_results = search(abstract_vector, limit)

        # Gather details about the found papers
        all_details = fetch_all_details(search_results)

        return all_details
    
    elif input_type == "Abstract or Description":

        # Check if input is empty
        if input_text == "":
            raise gr.Error("Please enter an abstract or description", 10)

        abstract_vector = embed(input_text)

        search_results = search(abstract_vector, limit)

        all_details = fetch_all_details(search_results)
        
        return all_details

    else:
        return "Please provide either an ArXiv ID or an abstract."
            

contact_text = """
# Contact Information

👤  [Mitanshu Sukhwani](https://www.linkedin.com/in/mitanshusukhwani/)

✉️  mitanshu.sukhwani@gmail.com

🐙  [mitanshu7](https://github.com/mitanshu7)
"""

examples = [
    ["ArXiv ID", "2401.07215"],
    ["Abstract or Description", "Game theory applications in marine biology"]
]

################################################################################
# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title='PaperMatch') as demo:

    # Title and description
    gr.Markdown("# PaperMatch: Discover Related Research Papers")
    gr.Markdown("## Enter either an [ArXiv ID](https://info.arxiv.org/help/arxiv_identifier.html) or paste an abstract to explore papers based on semantic similarity.")
    gr.Markdown("### _ArXiv Database last updated: August 2024_")
    
    # Input Section
    with gr.Row():
        input_type = gr.Dropdown(
            choices=["ArXiv ID", "Abstract or Description"],
            label="Input Type",
            value="ArXiv ID",
            interactive=True,
        )
        id_or_text_input = gr.Textbox(
            label="Enter ArXiv ID or Abstract", 
            placeholder="e.g., 1706.03762 or an abstract...",
        )
    
    # Example inputs
    gr.Examples(
        examples=examples, 
        inputs=[input_type, id_or_text_input],
        label="Example Queries"
    )

    # Slider for results count
    slider_input = gr.Slider(
        minimum=1, maximum=25, value=5, step=1, 
        label="Number of Similar Papers"
    )

    # Submission Button
    submit_btn = gr.Button("Find Papers")
    
    # Output section
    output = gr.HTML(label="Related Papers")

    # Attribution
    gr.Markdown(contact_text)
    gr.Markdown("_Thanks to [ArXiv](https://arxiv.org) for their open access interoperability._")

    # Link button click to the prediction function
    submit_btn.click(predict, [input_type, id_or_text_input, slider_input], output)


################################################################################

if __name__ == "__main__":
    demo.launch(server_port=7861, favicon_path='logo.png')
