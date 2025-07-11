# Import required libraries
import re
from datetime import datetime
from functools import cache

import arxiv
import gradio as gr
import numpy as np
import torch
from dotenv import dotenv_values
from mixedbread import Mixedbread
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

################################################################################
# Configuration

# Set to True if you want to use local recources (cpu/gpu) or False if you want to use MixedBread.ai
LOCAL = False

# Set to True if you want to use the fp32 embbedings or False if you want to use the binary embbedings
FLOAT = False

# Get current year
current_year = str(datetime.now().year)

# Define Milvus client
milvus_client = MilvusClient("http://localhost:19530")

# Construct the Arxiv API client.
arxiv_client = arxiv.Client(page_size=1, delay_seconds=1)

# Load Model
# Model to use for embedding
model_name = "mixedbread-ai/mxbai-embed-large-v1"

if LOCAL:
    # Make the app device agnostic
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load a pretrained Sentence Transformer model and move it to the appropriate device
    print(f"Loading model {model_name} to device: {device}")
    model = SentenceTransformer(model_name).to(device)

else:
    # Import secrets
    config = dotenv_values(".env")

    # Setup mxbai client
    mxbai_api_key = config["MXBAI_API_KEY"]
    mxbai = Mixedbread(api_key=mxbai_api_key)


################################################################################
# Function to extract arXiv ID from a given text
def extract_arxiv_id(text: str) -> str | None:
    # Define regex patterns for pre-2007 and post-2007 arXiv IDs
    pre_2007_pattern = re.compile(
        r"(?:^|\s|\/|arXiv:)([a-z-]+(?:\.[A-Z]{2})?\/\d{2}(?:0[1-9]|1[012])\d{3})(?:v\d+)?(?=$|\s)",
        re.IGNORECASE | re.MULTILINE,
    )
    post_2007_pattern = re.compile(
        r"(?:^|\s|\/|arXiv:)(\d{4}\.\d{4,5})(?:v\d+)?(?=$|\s)",
        re.IGNORECASE | re.MULTILINE,
    )

    # Search for matches
    pre_match = pre_2007_pattern.search(text)
    post_match = post_2007_pattern.search(text)

    # Combine the matches
    match = pre_match or post_match

    # Return the match if found, otherwise return None
    return match.group(1) if match else None


################################################################################


# Function to search ArXiv by ID
@cache
def fetch_arxiv_by_id(arxiv_id: str) -> dict | gr.Error:
    # Search for the paper using the Arxiv API
    search = arxiv.Search(id_list=[arxiv_id])

    try:
        # Fetch the paper metadata using the Arxiv API
        paper = next(arxiv_client.results(search), None)

        # Extract the relevant metadata from the paper object
        return {
            "id": extract_arxiv_id(paper.entry_id),
            "title": paper.title.replace("\n", " "),
            "authors": ", ".join(
                [str(author).replace("\n", " ") for author in paper.authors]
            ),
            "abstract": paper.summary.replace("\n", " "),
            "url": paper.pdf_url,
            "month": paper.published.strftime("%B"),
            "year": paper.published.year,
            "categories": ", ".join(paper.categories).replace("\n", " "),
        }

    except Exception as e:
        # Raise an exception if the request was not successful
        raise gr.Error(f"Failed to fetch metadata for ID '{arxiv_id}'. Error: {e}")


################################################################################
# Function to convert dense vector to binary vector
def dense_to_binary(dense_vector: np.ndarray) -> bytes:
    return np.packbits(np.where(dense_vector >= 0, 1, 0)).tobytes()


# Function to embed text
@cache
def embed(text: str) -> np.ndarray | bytes:
    # Check if the embedding should be a float or binary vector
    if FLOAT:
        # Check if the embedding should be generated locally or using the MixedBread.ai API
        if LOCAL:
            # Calculate embeddings by calling model.encode(), specifying the device
            embedding = model.encode(text, device=device, precision="float32")

            # Enforce 32-bit float precision
            embedding = np.array(embedding, dtype=np.float32)

        else:
            # Call the MixedBread.ai API to generate the embedding
            result = mxbai.embed(
                model="mixedbread-ai/mxbai-embed-large-v1",
                input=text,
                normalized=True,
                encoding_format="float",
                dimensions=1024,
            )

            embedding = np.array(result.data[0].embedding, dtype=np.float32)

    # If the embedding should be a binary vector
    else:
        # Check if the embedding should be generated locally or using the MixedBread.ai API
        if LOCAL:
            # Calculate embeddings by calling model.encode(), specifying the device
            embedding = model.encode(text, device=device, precision="float32")

            # Enforce 32-bit float precision
            embedding = np.array(embedding, dtype=np.float32)

            # Convert the dense vector to a binary vector
            embedding = dense_to_binary(embedding)

        else:
            # Call the MixedBread.ai API to generate the embedding
            result = mxbai.embed(
                model="mixedbread-ai/mxbai-embed-large-v1",
                input=text,
                normalized=True,
                encoding_format="ubinary",
                dimensions=1024,
            )

            # Convert the embedding to a numpy array of uint8 encoding and then to bytes
            embedding = np.array(result.data[0].embedding, dtype=np.uint8).tobytes()

    return embedding


################################################################################
# Single vector search


def search(vector: np.ndarray, limit: int, filter: str = "") -> list[dict]:
    # Logic for converting the filter to a valid format
    if filter == "This Year":
        filter = f"year == {int(current_year)}"
    elif filter == "Last 5 Years":
        filter = f"year >= {int(current_year) - 5}"
    elif filter == "Last 10 Years":
        filter = f"year >= {int(current_year) - 10}"
    elif filter == "All":
        filter = ""

    result = milvus_client.search(
        collection_name="arxiv_abstracts",  # Collection to search in
        data=[vector],  # Vector to search for
        limit=limit,  # Max. number of search results to return
        output_fields=[
            "id",
            "vector",
            "title",
            "abstract",
            "authors",
            "categories",
            "month",
            "year",
            "url",
        ],  # Output fields to return
        filter=filter,  # Filter to apply to the search
    )

    # returns a list of dictionaries with id and distance as keys
    return result[0]


################################################################################
# Function to fetch paper details of all results
def fetch_all_details(search_results: list[dict]) -> str:
    # Initialize an empty string to store the cards
    cards = ""

    for search_result in search_results:
        paper_details = search_result["entity"]

        # chr(10) is a new line character, replace to avoid formatting issues
        card = f"""
## [{paper_details["title"]}]({paper_details["url"]})
> **{paper_details["authors"]}** | _{paper_details["month"]} {paper_details["year"]}_ \n
{paper_details["abstract"]} \n
[OpenReview](https://duckduckgo.com/?q={paper_details["title"].replace(" ", "+")}+site:openreview.net)/[Google Scholar](https://scholar.google.com/scholar?q={paper_details["title"].replace(" ", "+")})
***
"""

        cards += card

    return cards


################################################################################


# Function to handle the UI logic
def predict(
    input_text: str, limit: int = 5, increment: int = 5, filter: str = ""
) -> tuple[str, gr.update, int]:
    # Check if input is empty
    if input_text == "":
        raise gr.Error("Please provide either an ArXiv ID or an abstract.", 10)

    # Define extra outputs to pass
    # This hack shows the load_more button once the search has been made
    show_load_more = gr.update(visible=True)
    show_date_filter = gr.update(visible=True)

    # This variable is used to increment the search limit when the load_more button is clicked
    new_limit = limit + increment

    # Extract arxiv id, if any
    arxiv_id = extract_arxiv_id(input_text)

    # When arxiv id is found in input text
    if arxiv_id:
        # Search if id is already in database
        id_in_db = milvus_client.get(collection_name="arxiv_abstracts", ids=[arxiv_id])

        # If the id is already in database
        if bool(id_in_db):
            # Get the 1024-dimensional dense vector
            if FLOAT:
                abstract_vector = id_in_db[0]["vector"]

            # Get the bytes of a binary vector
            else:
                abstract_vector = id_in_db[0]["vector"][0]

        # If the id is not already in database
        else:
            # Search arxiv for paper details
            arxiv_json = fetch_arxiv_by_id(arxiv_id)

            # Embed abstract
            abstract_vector = embed(arxiv_json["abstract"])

    # When arxiv id is not found in input text, treat input text as abstract
    else:
        # Embed abstract
        abstract_vector = embed(input_text)

    # Search database
    search_results = search(abstract_vector, limit, filter)

    # Gather details about the found papers
    all_details = fetch_all_details(search_results)

    return all_details, show_load_more, show_date_filter, new_limit


################################################################################

# Variable to store contact information
contact_text = """
<div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
    <h3>Crafted with ❤️ by <a href="https://mitanshu7.github.io" target="_blank">Mitanshu Sukhwani</a></h3>
    <h4>Discover more at <a href="https://bio.papermatch.me" target="_blank">PaperMatchBio</a></h4>
</div>
"""

# Examples to display
examples = ["2401.07215", "Smart TV and privacy"]

# Show total number of entries in database
num_entries = format(
    milvus_client.get_collection_stats(collection_name="arxiv_abstracts")["row_count"],
    ",",
)

# Create a back to top button
back_to_top_btn_html = """
<button id="toTopBtn" onclick="'parentIFrame' in window ? window.parentIFrame.scrollTo({top: 0, behavior:'smooth'}) : window.scrollTo({ top: 0 })">
    <a style="color:#6366f1; text-decoration:none;">&#8593;</a> <!-- Use the ^ character -->
</button>"""

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

# Markdown for about page
about_markdown = """
**PaperMatch** is a semantic search engine. Unlike regular search engines that match keywords in a query, semantic search engines convert text into vectors — essentially lists of numbers — using an *embedding model* (a type of neural network). These vectors aim to capture the **semantics** (meaning) of the text.

Because numbers can be compared (`a > b`), we can **indirectly compare text** by comparing their corresponding vectors. This is the core idea behind PaperMatch.

**PaperMatch** converts the abstract of an arXiv paper into a vector and performs a similarity search over a corpus of other papers.

---

## Guide to Using PaperMatch

### 🔍 Search by arXiv ID

* Enter the arXiv identifier (e.g., `1706.03762`) to search for similar papers.
* You can also paste the full arXiv URL — PaperMatch will automatically extract the ID.

### 📝 Search by Text

* Enter natural language describing the kind of paper you're looking for.
* Keep in mind: since the system matches by **semantics** (not keywords), exact terms may not appear in the results — it's the **meaning** that matters.
"""

################################################################################
# Create the Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(
        font=gr.themes.GoogleFont("Helvetica"),
        font_mono=gr.themes.GoogleFont("Roboto Mono"),
    ),
    title="PaperMatch",
    css=style,
    analytics_enabled=False,
) as demo:
    # Title and Subtitle
    gr.HTML(
        '<h1><a href="https://papermatch.me" style="font-weight: bold; text-decoration: none;">PaperMatch</a></h1>'
    )
    gr.HTML("<h3> Discover Relevant Research, Instantly ⚡</h3>")

    # Input Section
    with gr.Row():
        input_text = gr.Textbox(
            placeholder=f"Search {num_entries} papers on arXiv",
            autofocus=True,
            submit_btn=True,
            show_label=False,
        )

    with gr.Row():
        # Add the date filter
        with gr.Column(scale=4):
            date_filter = gr.Dropdown(
                label="Filter by Year",
                choices=["This Year", "Last 5 Years", "Last 10 Years", "All"],
                value="All",
                visible=False,
                multiselect=False,
                allow_custom_value=False,
                filterable=False,
            )

        # # Add sorting options
        # with gr.Column(scale=1):

    # Define the initial page limit
    page_limit = gr.State(5)

    # Define the increment for the "Load More" button
    increment = gr.State(5)

    # Define new page limit
    new_page_limit = gr.State(page_limit.value + increment.value)

    # Output section, displays the search results
    output = gr.Markdown(
        label="Related Papers",
        latex_delimiters=[{"left": "$", "right": "$", "display": False}],
        padding=True,
    )

    # Hidden by default, appears after the first search
    load_more_button = gr.Button("More results ⬇️", visible=False)

    # Event handler for the input text box, triggers the search function
    input_text.submit(
        predict,
        [input_text, page_limit, increment, date_filter],
        [output, load_more_button, date_filter, new_page_limit],
        api_name="search",
    )

    # Event handler for the date filter dropbox
    date_filter.change(
        predict,
        [input_text, page_limit, increment, date_filter],
        [output, load_more_button, date_filter, new_page_limit],
        api_name=False,
    )

    # Event handler for the "Load More" button
    load_more_button.click(
        predict,
        [input_text, new_page_limit, increment, date_filter],
        [output, load_more_button, date_filter, new_page_limit],
        api_name=False,
    )

    # Example inputs
    gr.Examples(
        examples=examples,
        inputs=input_text,
        outputs=[output, load_more_button, date_filter, new_page_limit],
        fn=predict,
        label="Try:",
        run_on_click=True,
        cache_examples=False,
    )

    # Back to top button
    gr.HTML(back_to_top_btn_html)

    # Attribution
    gr.HTML(contact_text)

with demo.route("About", "/about"):
    # Title and Subtitle
    gr.HTML(
        '<h1><a href="https://papermatch.me" style="font-weight: bold; text-decoration: none;">PaperMatch</a></h1>'
    )
    gr.HTML("<h3> Discover Relevant Research, Instantly ⚡</h3>")

    # The about text
    gr.Markdown(about_markdown, padding=True)

    # Attribution
    gr.HTML(contact_text)


################################################################################

if __name__ == "__main__":
    demo.launch(server_port=7860, favicon_path="logo.png", show_api=False, pwa=True)
