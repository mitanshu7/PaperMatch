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

    search = arxiv.Search(id_list=[arxiv_id])

    paper = next(arxiv_client.results(search), None)

    if paper:
        return {
            "Title": paper.title,
            "Authors": ", ".join([str(author) for author in paper.authors]),
            "Abstract": paper.summary,
            "Link": paper.pdf_url
        }
    return "No paper found."

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
        search_params={"metric_type": "COSINE"} # Search parameters
    )

    # returns a list of dictionaries with id and distance as keys
    return result[0]

################################################################################
# Function to 

def fetch_all_details(search_results):

    all_details = []

    for search_result in search_results:

        paper_details = fetch_arxiv_by_id(search_result['id'])

        paper_details['similarity'] = search_result['distance']

        all_details.append(paper_details)

    return all_details

################################################################################
@cache
def make_clickable(val):
        # Regex to detect URLs in the value
        if re.match(r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', val):
            return f"[{val}]({val})"
        return val

################################################################################

# Function to convert list of dictionaries to a styled HTML table

def parse_output(data):
    
    df = pd.DataFrame(data)

    df['Link'] = df['Link'].apply(make_clickable)

    return df

################################################################################

# Function to handle the UI logic
@cache
def predict(input_type, input_text, limit):

    # When input is arxiv id
    if input_type == "ArXiv ID":

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
        

        df = parse_output(all_details)

        return df
    
    elif input_type == "Abstract or Paper Description":

        abstract_vector = embed(input_text)

        search_results = search(abstract_vector, limit)

        all_details = fetch_all_details(search_results)
        
        df = parse_output(all_details)

        return df

    else:
        return "Please provide either an ArXiv ID or an abstract."
            

contact_text = """
# Contact Information

👤  [Mitanshu Sukhwani](https://www.linkedin.com/in/mitanshusukhwani/)

✉️  mitanshu.sukhwani@gmail.com

🐙  [mitanshu7](https://github.com/mitanshu7)
"""

examples = [
    ["1706.03762"],
    ["2401.07215"],
    ["Deep learning models assisting in aaa games"],
    ["Game theory applications in marine biology"],
    ["The modern coffee market aims to provide products which are both consistent and have desirable flavour characteristics. Espresso, one of the most widely consumed coffee beverage formats, is also the most susceptible to variation in quality. Yet, the origin of this inconsistency has traditionally, and incorrectly, been attributed to human variations. This study's mathematical model, paired with experiment, has elucidated that the grinder and water pressure play pivotal roles in achieving beverage reproducibility. We suggest novel brewing protocols that not only reduce beverage variation but also decrease the mass of coffee used per espresso by up to 25%. If widely implemented, this protocol will have significant economic impact and create a more sustainable coffee-consuming future."]
]

################################################################################
# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:

    # Title and Description
    gr.Markdown("# PaperMatch: Find Related Research Papers")
    gr.Markdown("## Simply enter an [ArXiv ID](https://info.arxiv.org/help/arxiv_identifier.html) or paste an abstract to discover similar papers based on semantic similarity.")
    gr.Markdown("### ArXiv Search Database last updated: Aug-2024")
    gr.Markdown("#### *** Please refresh page after each result to avoid eternal buffering. It's a known bug. *** ")
    
    # Dropdown to select input type
    input_type = gr.Dropdown(
        choices=["ArXiv ID", "Abstract or Paper Description"],
        label="Select Input Type",
        value="ArXiv ID"
    )
    
    # Input: Textbox for user input (alphanumeric ID, text, or text with numbers)
    id_or_text_input = gr.Textbox(label="Enter Input")

    # Examples
    examples = gr.Examples(examples, id_or_text_input)
    
    # Slider
    slider_input = gr.Slider(minimum=1, maximum=25, value=5, step=1, label="Top-k results")

    # Button to trigger the process
    submit_btn = gr.Button("Submit")
    
    # Output: HTML table for list of dictionaries
    output = gr.DataFrame(wrap=True, datatype=["str", "str", "str", "markdown", "number"], label="Search Results", show_label=True)

    # Required Attribution
    gr.Markdown(contact_text)
    gr.Markdown("Thank you to ArXiv for use of its open access interoperability.")

    # Link button click to the function
    submit_btn.click(predict, [input_type, id_or_text_input, slider_input], output)

################################################################################

if __name__ == "__main__":
    demo.launch()