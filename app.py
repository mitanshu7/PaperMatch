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
## [{row["Title"].replace(chr(10),"")}]({row["URL"]})
> {row["Authors"]} \n
{row["Abstract"]}
***
"""
    
        cards +=card
    
    return cards

################################################################################

# Function to handle the UI logic
@cache
def predict(input_text, limit=5, increment=5):

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

        return all_details, gr.update(visible=True), limit+increment
    
    # When arxiv id is not found in input_text, treat input_text as abstract
    else:
        
        # Embed abstract
        abstract_vector = embed(input_text)

        # Search database
        search_results = search(abstract_vector, limit)

        # Gather details about the found papers
        all_details = fetch_all_details(search_results)
        
        return all_details, gr.update(visible=True), limit+increment

################################################################################

# Variable to store contact information
contact_text = """
<div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
    <h3>Made with ❤️ by <a href="https://www.linkedin.com/in/mitanshusukhwani/" target="_blank">Mitanshu Sukhwani</a></h3>
</div>
"""

# Examples to display
examples = [
    "2401.07215",
    "Smart TV and privacy"
]

# Show total number of entries in database
num_entries = format(milvus_client.get_collection_stats(collection_name="arxiv_abstracts")['row_count'], ",")

# Create a back to top button
back_to_top_btn_html = '''
<button id="toTopBtn" onclick="'parentIFrame' in window ? window.parentIFrame.scrollTo({top: 0, behavior:'smooth'}) : window.scrollTo({ top: 0 })">
    <a style="color:#6366f1; text-decoration:none;">&#8593;</a> <!-- Use the ^ character -->
</button>'''

# CSS for the back to top button
style = """
#toTopBtn {
    position: fixed;
    bottom: 10px;
    right: 10px; /* Adjust this value to position it at the bottom-right corner */
    height: 40px; /* Increase the height for a better look */
    width: 40px; /* Set a fixed width for the button */
    font-size: 20px; /* Set font size for the ^ icon */
    border-color: #e0e7ff; /* Change border color using hex */
    background-color: #e0e7ff; /* Change background color using hex */
    text-align: center; /* Align the text in the center */
    display: flex;
    justify-content: center;
    align-items: center;
    border-radius: 50%; /* Make it circular */
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2); /* Add shadow for better visibility */
}

#toTopBtn:hover {
    background-color: #c7d4ff; /* Change background color on hover */
}
"""

################################################################################
# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(font=gr.themes.GoogleFont("Helvetica"), 
                                    font_mono=gr.themes.GoogleFont("Roboto Mono")), 
                                    title='PaperMatch', css=style) as demo:

    # Title and description
    gr.HTML('<h1><a href="https://papermatch.mitanshu.tech" style="font-weight: bold; text-decoration: none;">PaperMatch</a></h1>')
    gr.Markdown("### Discover Relevant Research, Instantly ⚡")

    # Input Section
    with gr.Row():
        input_text = gr.Textbox(
            placeholder=f"Search {num_entries} papers on arXiv",
            autofocus=True,
            submit_btn=True,
            show_label=False
        )

    # State to track the current page limit
    page_limit = gr.State(5)

    # Define the increment for the "Load More" button
    increment = gr.State(5)

    # Output section, displays the search results
    output = gr.Markdown(label="Related Papers", latex_delimiters=[{ "left": "$", "right": "$", "display": False}])

    # Hidden by default, appears after the first search
    load_more_button = gr.Button("Load More", visible=False)

    # Event handler for the input text box, triggers the search function
    input_text.submit(predict, [input_text, page_limit, increment], [output, load_more_button, page_limit])

    # Event handler for the "Load More" button
    load_more_button.click(predict, [input_text, page_limit, increment], [output, load_more_button, page_limit])

    # Example inputs
    gr.Examples(
        examples=examples, 
        inputs=input_text,
        outputs=[output, load_more_button, page_limit],
        fn=predict,
        label="Try:",
        run_on_click=True)

    # Back to top button
    gr.HTML(back_to_top_btn_html)

    # Attribution
    gr.HTML(contact_text)

################################################################################

if __name__ == "__main__":
    demo.launch(server_port=7861, favicon_path='logo.png')
