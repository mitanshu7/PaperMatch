# Import required libraries
import gradio as gr
from pymilvus import MilvusClient
import numpy as np
import arxiv
from mixedbread_ai.client import MixedbreadAI
from dotenv import dotenv_values
import re

################################################################################

# Define client
client = MilvusClient("http://localhost:19530")

# Import secrets
config = dotenv_values(".env")

# Setup mxbai
mxbai_api_key = config["MXBAI_API_KEY"]
mxbai = MixedbreadAI(api_key=mxbai_api_key)

################################################################################

# Function to search ArXiv by ID
def fetch_arxiv_by_id(arxiv_id):

    search = arxiv.Search(id_list=[arxiv_id])

    paper = next(search.results(), None)

    if paper:
        return {
            "Title": paper.title,
            "Authors": ", ".join([str(author) for author in paper.authors]),
            "Abstract": paper.summary,
            "Link": paper.pdf_url
        }
    return "No paper found."
################################################################################

def fetch_all_details(search_results):

    all_details = []

    for search_result in search_results:

        paper_details = fetch_arxiv_by_id(search_result['id'])

        paper_details['similarity'] = search_result['distance']

        all_details.append(paper_details)

    return all_details

################################################################################

def make_clickable(val):
        # Regex to detect URLs in the value
        if re.match(r'^https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', val):
            return f'<a href="{val}" target="_blank" style="color: #007BFF;">{val}</a>'
        return val

################################################################################

# Function to convert list of dictionaries to a styled HTML table
def dict_list_to_pretty_table(data):
    html = """
    <style>
    table {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    th, td {
        border: 1px solid #dddddd;
        text-align: left;
        padding: 10px;
        font-size: 14px;
    }
    th {
        background-color: #6EC1E4;
        color: white;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    tr:nth-child(odd) {
        background-color: #ffffff;
    }
    tr:hover {
        background-color: #d1ecf1;
        cursor: pointer;
    }
    td {
        color: #333;
    }
    </style>
    <table>
    """
    
    # Create table header
    html += "<tr>"
    for key in data[0].keys():
        html += f"<th>{key}</th>"
    html += "</tr>"
    
    # Add data rows
    for entry in data:
        html += "<tr>"
        for value in entry.values():
            clickable_value = make_clickable(str(value))
            html += f"<td>{clickable_value}</td>"
        html += "</tr>"
    
    html += "</table>"
    return html

################################################################################

# Function to embed text
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

    result = client.search(
        collection_name="arxiv_abstracts", # Replace with the actual name of your collection
        # Replace with your query vector
        data=[vector],
        limit=limit, # Max. number of search results to return
        search_params={"metric_type": "COSINE"} # Search parameters
    )

    # returns a list of dictionaries with id and distance as keys
    return result[0]

################################################################################

# Function to handle the UI logic
def predict(input_type, input_text, limit):

    if input_type == "ArXiv ID":

        arxiv_json = fetch_arxiv_by_id(input_text)

        abstract_vector = embed(arxiv_json['Abstract'])

        search_results = search(abstract_vector, limit)

        all_details = fetch_all_details(search_results)
        
        html = dict_list_to_pretty_table(all_details)

        return html
    
    elif input_type == "Abstract or Paper Description":

        abstract_vector = embed(input_text)

        search_results = search(abstract_vector, limit)

        all_details = fetch_all_details(search_results)
        
        html = dict_list_to_pretty_table(all_details)

        return html

    else:
        return "Please provide either an ArXiv ID or an abstract."
            

contact_text = """
# Contact Information

👤  Mitanshu Sukhwani 

✉️  mitanshu.sukhwani@gmail.com

🐙  [mitanshu7](https://github.com/mitanshu7)
"""

################################################################################
# Create the Gradio interface
with gr.Blocks() as demo:

    # Title and Description
    gr.Markdown("# PaperMatch: Find Related Research Papers")
    gr.Markdown("## Simply enter an ArXiv ID or paste an abstract to discover similar papers based on semantic similarity.")
    gr.Markdown("### ArXiv Search Database last updated: Aug-2024")
    
    # Dropdown to select input type
    input_type = gr.Dropdown(
        choices=["ArXiv ID", "Abstract or Paper Description"],
        label="Select Input Type",
        value="ArXiv ID"
    )
    
    # Input: Textbox for user input (alphanumeric ID, text, or text with numbers)
    id_or_text_input = gr.Textbox(label="Enter Input")
    
    # Slider
    slider_input = gr.Slider(minimum=0, maximum=50, value=5, label="Top-k results")

    # Button to trigger the process
    submit_btn = gr.Button("Submit")
    
    # Output: HTML table for list of dictionaries
    output = gr.HTML(label="Pretty Output")

    # Required Attribution
    gr.Markdown(contact_text)
    gr.Markdown("Thank you to ArXiv for use of its open access interoperability.")

    # Link button click to the function
    submit_btn.click(predict, [input_type, id_or_text_input, slider_input], output)

################################################################################

if __name__ == "__main__":
    demo.launch()