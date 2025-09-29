# Import required libraries
import re
from datetime import datetime
from functools import cache

import arxiv
import numpy as np
from dotenv import dotenv_values
from mixedbread import Mixedbread
from pymilvus import MilvusClient
from fastapi import FastAPI
from schemas import TextRequest, ArxivPaper
import backoff
################################################################################
# Configuration

app = FastAPI()

# Get current year
current_year = str(datetime.now().year)

# Import secrets
config = dotenv_values(".env")

# Define Milvus client
# ENDPOINT = config['ENDPOINT']
# TOKEN = config['TOKEN']
# milvus_client = MilvusClient(uri=ENDPOINT, token=TOKEN)

# Setup mxbai client
mxbai_api_key = config["MXBAI_API_KEY"]
mxbai = Mixedbread(api_key=mxbai_api_key)

# Construct the Arxiv API client.
arxiv_client = arxiv.Client(page_size=1, delay_seconds=0)

    
# Define regex patterns for pre-2007 and post-2007 arXiv IDs
pre_2007_pattern = re.compile(
    r"(?:^|\s|\/|arXiv:)([a-z-]+(?:\.[A-Z]{2})?\/\d{2}(?:0[1-9]|1[012])\d{3})(?:v\d+)?(?=$|\s)",
    re.IGNORECASE | re.MULTILINE,
)

post_2007_pattern = re.compile(
    r"(?:^|\s|\/|arXiv:)(\d{4}\.\d{4,5})(?:v\d+)?(?=$|\s)",
    re.IGNORECASE | re.MULTILINE,
)

################################################################################

# Function to extract arXiv ID from a given text
@app.post("/extract_arxiv_id_from_text")
def extract_arxiv_id_from_text(request: TextRequest) -> str | None:
    
    # Extract the text form the request
    text = request.text

    # Search for matches
    pre_match = pre_2007_pattern.search(text)
    post_match = post_2007_pattern.search(text)

    # Combine the matches
    # first (left) match will be prioritised if both are found 
    match = pre_match or post_match

    # Return the match if found, otherwise return None
    return match.group(1) if match else None


################################################################################
@app.get("/extract_arxiv_id_from_url/")
def extract_arxiv_id_from_url(arxiv_url:str) -> str:
    
    id_with_version = arxiv_url.split('/')[-1]
        
    id = id_with_version.split('v')[0]
    
    return id

# Function to search ArXiv by ID
@app.get("/fetch_arxiv_by_id/{arxiv_id}")
@backoff.on_exception(wait_gen=backoff.expo, exception=arxiv.HTTPError, max_tries=3, jitter=backoff.full_jitter)
def fetch_arxiv_by_id(arxiv_id: str) -> ArxivPaper:
    # Search for the paper using the Arxiv API
    search = arxiv.Search(id_list=[arxiv_id])

    # Fetch the paper metadata using the Arxiv API
    paper = next(arxiv_client.results(search), None)
        
    ArxivPaper.id =  extract_arxiv_id_from_url(paper.entry_id)
    ArxivPaper.title = paper.title.replace("\n", " ")
    ArxivPaper.authors = [str(author) for author in paper.authors]
    ArxivPaper.abstract = paper.summary.replace("\n", " ")
    ArxivPaper.url = paper.entry_id
    ArxivPaper.pdf = paper.pdf_url
    ArxivPaper.month = paper.published.month
    ArxivPaper.year = paper.published.year
    ArxivPaper.categories = paper.categories

    # Extract the relevant metadata from the paper object
    return ArxivPaper



################################################################################
# Function to convert dense vector to binary vector
def dense_to_binary(dense_vector: np.ndarray) -> bytes:
    return np.packbits(np.where(dense_vector >= 0, 1, 0)).tobytes()


# Function to embed text
@cache
def embed(text: str) -> np.ndarray | bytes:


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
        collection_name="arxiv",  # Collection to search in
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

