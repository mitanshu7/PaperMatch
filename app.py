# Import required libraries
import gradio as gr
from pymilvus import MilvusClient
import numpy as np
import arxiv
from mixedbread_ai.client import MixedbreadAI
from dotenv import dotenv_values

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

# Function to convert list of dictionaries to a styled HTML table
def dict_list_to_pretty_table(data):
    html = """
    <style>
    table {
        font-family: Arial, sans-serif;
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
    }
    th, td {
        border: 1px solid #dddddd;
        text-align: left;
        padding: 8px;
    }
    tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    th {
        background-color: #4CAF50;
        color: white;
    }
    tr:hover {
        background-color: #ddd;
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
            html += f"<td>{value}</td>"
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

    if input_type == "Arxiv ID":

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
            


################################################################################
# Create the Gradio interface
with gr.Blocks() as demo:

    # Title or header
    gr.Markdown("# Input Type Selector App")
    
    # Dropdown to select input type
    input_type = gr.Dropdown(
        choices=["Arxiv ID", "Abstract or Paper Description"],
        label="Select Input Type",
        value="Arxiv ID"
    )
    
    # Input: Textbox for user input (alphanumeric ID, text, or text with numbers)
    id_or_text_input = gr.Textbox(label="Enter Input")
    
    # Slider
    slider_input = gr.Slider(minimum=0, maximum=50, value=5, label="Slider")

    # Button to trigger the process
    submit_btn = gr.Button("Submit")
    
    # Output: HTML table for list of dictionaries
    output = gr.HTML(label="Pretty Output")
    

    # Link button click to the function
    submit_btn.click(predict, [input_type, id_or_text_input, slider_input], output)

################################################################################

if __name__ == "__main__":
    demo.launch()