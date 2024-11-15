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
# Function to extract arXiv ID from a given text
def extract_arxiv_id(text):

    # Define regex patterns for pre-2007 and post-2007 arXiv IDs
    pre_2007_pattern = re.compile(r"(?:^|\s|\/|arXiv:)([a-z-]+(?:\.[A-Z]{2})?\/\d{2}(?:0[1-9]|1[012])\d{3})(?:v\d+)?(?=$|\s)", re.IGNORECASE|re.MULTILINE)
    post_2007_pattern = re.compile(r"(?:^|\s|\/|arXiv:)(\d{4}\.\d{4,5})(?:v\d+)?(?=$|\s)", re.IGNORECASE|re.MULTILINE)

    # Search for matches
    pre_match = pre_2007_pattern.search(text)
    post_match = post_2007_pattern.search(text)

    # Combine the matches
    match = pre_match or post_match

    # Return the match if found, otherwise return None
    return match.group(1) if match else None

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

    # Make a card for each row
    cards = ""

    for index, row in df.iterrows():

    # chr(10) is a new line character, replace to avoid formatting issues
        card = f"""
### [{row["Title"].replace(chr(10),"")}]({row["URL"]})
> {row["Authors"]} \n
{row["Abstract"]}
***
"""
    
        cards +=card
    
    return cards

################################################################################

# Function to handle the UI logic
@cache
def predict(input_text, limit):

    # Check if input is empty
    if input_text == "":
        raise gr.Error("Please provide either an ArXiv ID or an abstract.", 10)
    
    # Extract arxiv id, if any
    arxiv_id = extract_arxiv_id(input_text)

    # When arxiv id is found in input_text 
    if arxiv_id:

        # Search if id is already in database
        id_in_db = milvus_client.get(collection_name="arxiv_abstracts",ids=[arxiv_id])

        # If the id is already in database
        if bool(id_in_db):

            # Get the vector
            abstract_vector = id_in_db[0]['vector']

        # If the id is not already in database
        else:

            # Search arxiv for paper details
            arxiv_json = fetch_arxiv_by_id(arxiv_id)

            # Embed abstract
            abstract_vector = embed(arxiv_json['Abstract'])

        # Search database
        search_results = search(abstract_vector, limit)

        # Gather details about the found papers
        all_details = fetch_all_details(search_results)

        return all_details
    
    # When arxiv id is not found in input_text, treat input_text as abstract
    else:
        
        # Embed abstract
        abstract_vector = embed(input_text)

        # Search database
        search_results = search(abstract_vector, limit)

        # Gather details about the found papers
        all_details = fetch_all_details(search_results)
        
        return all_details
            

contact_text = """
# Contact Information

👤  [Mitanshu Sukhwani](https://www.linkedin.com/in/mitanshusukhwani/)

✉️  mitanshu.sukhwani@gmail.com

🐙  [mitanshu7](https://github.com/mitanshu7)
"""

examples = [
    "2401.07215",
    "Game theory applications in marine biology"
]

################################################################################
# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(font=gr.themes.GoogleFont("Helvetica"), 
                                    font_mono=gr.themes.GoogleFont("Roboto Mono")), 
                                    title='PaperMatch') as demo:

    # Title and description
    gr.Markdown("# PaperMatch: Discover Related Research Papers")
    gr.Markdown("## Enter either an [ArXiv ID](https://info.arxiv.org/help/arxiv_identifier.html) or paste an abstract to explore papers based on semantic similarity.")
    gr.Markdown("### Visit [PaperMatchMed](https://papermatchmed.mitanshu.tech) for [MedRiv](https://medrxiv.org/) and [PaperMatchBio](https://papermatchbio.mitanshu.tech) for [BioRxiv](https://www.biorxiv.org/) alternatives.")
    gr.Markdown("### _ArXiv Database last updated: 6th November 2024_")
    
    # Input Section
    with gr.Row():
        input_text = gr.Textbox(
            label="Enter ArXiv ID or Abstract", 
            placeholder="e.g., 1706.03762 or an abstract...",
        )
    
    # Example inputs
    gr.Examples(
        examples=examples, 
        inputs=input_text,
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
    output = gr.Markdown(label="Related Papers", latex_delimiters=[{ "left": "$", "right": "$", "display": False}])

    # Attribution
    gr.Markdown(contact_text)
    gr.Markdown("_Thanks to [ArXiv](https://arxiv.org) for their open access interoperability._")

    # Link button click to the prediction function
    submit_btn.click(predict, [input_text, slider_input], output)


################################################################################

if __name__ == "__main__":
    demo.launch(server_port=7861, favicon_path='logo.png')
