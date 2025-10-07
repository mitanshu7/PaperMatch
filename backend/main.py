# Import required libraries
import re
from datetime import datetime
from functools import cache
import os

import arxiv
import backoff
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mixedbread import Mixedbread
from pymilvus import MilvusClient
from schemas import ArxivPaper, TextRequest

################################################################################
# Configuration

app = FastAPI()

# TODO: MAKE IT SECURE
# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5500"] if serving static files
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get current year
current_year = str(datetime.now().year)

# Import secrets
load_dotenv()

# Connect to Zilliz via Milvus client
ENDPOINT = os.getenv("ENDPOINT")
TOKEN = os.getenv("TOKEN")
milvus_client = MilvusClient(uri=ENDPOINT, token=TOKEN)

# Setup search parameters
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT"))

# Setup mxbai client
mxbai_api_key = os.getenv("MXBAI_API_KEY")
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
def extract_arxiv_id_from_text(text: str) -> str | None:
    # Search for matches
    pre_match = pre_2007_pattern.search(text)
    post_match = post_2007_pattern.search(text)

    # Combine the matches
    # first (left) match will be prioritised if both are found
    match = pre_match or post_match

    # Return the match if found, otherwise return None
    return match.group(1) if match else None


################################################################################
# Function to extract arxiv id from the url
# Helpful for returning id from arxiv api results
def extract_arxiv_id_from_url(arxiv_url: str) -> str:
    id_with_version = arxiv_url.split("/")[-1]

    id = id_with_version.split("v")[0]

    return id


# Function to search ArXiv by ID
@backoff.on_exception(
    wait_gen=backoff.expo,
    exception=arxiv.HTTPError,
    max_tries=3,
    jitter=backoff.full_jitter,
)
def fetch_arxiv_by_id(arxiv_id: str) -> ArxivPaper:
    # Search for the paper using the Arxiv API
    search = arxiv.Search(id_list=[arxiv_id])

    # Fetch the paper metadata using the Arxiv API
    paper = next(arxiv_client.results(search), None)

    # Create the result model
    ArxivPaper.id = extract_arxiv_id_from_url(paper.entry_id)
    ArxivPaper.title = paper.title.replace("\n", " ")
    ArxivPaper.authors = [str(author) for author in paper.authors]
    ArxivPaper.abstract = paper.summary.replace("\n", " ")
    ArxivPaper.url = paper.entry_id
    ArxivPaper.pdf = paper.pdf_url
    ArxivPaper.month = paper.published.month
    ArxivPaper.year = paper.published.year
    ArxivPaper.categories = paper.categories

    return ArxivPaper


################################################################################


# Function to embed text using https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
@cache
def embed_text(text: str) -> bytes:
    # Call the MixedBread.ai API to generate the embedding
    result = mxbai.embed(
        model="mixedbread-ai/mxbai-embed-large-v1",
        input=text,
        normalized=True,
        encoding_format="ubinary",
        dimensions=1024,
    )

    # Extract the embedding from the response
    embedding = result.data[0].embedding

    # Convert the embedding to a numpy array of uint8 encoding and then to bytes
    vector_bytes = np.array(embedding, dtype=np.uint8).tobytes()

    return vector_bytes


################################################################################
# Single vector search
def search_by_vector(vector: bytes, filter: str = "") -> list[dict]:
    # Request zilliz for the vector search
    result = milvus_client.search(
        collection_name=COLLECTION_NAME,  # Collection to search in
        data=[vector],  # Vector to search for
        limit=SEARCH_LIMIT,  # Max. number of search results to return
        output_fields=[
            "id",
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


# Search the collection using text
@app.post("/search_by_text")
def search_by_text(request: TextRequest) -> list[dict]:
    # Extract objects?
    text = request.text
    filter = request.filter

    # Embed the text
    embedding = embed_text(text)

    # Send vector for search
    results = search_by_vector(vector=embedding, filter=filter)

    return results


################################################################################


# Search by known id
# The onus is on the user to make sure the id exists
# Use with similar results feature
@app.get("/search_by_known_id/{arxiv_id}")
def search_by_known_id(arxiv_id: str, filter: str = "") -> list[dict]:
    # Get the id which is already in database
    id_in_db = milvus_client.get(collection_name=COLLECTION_NAME, ids=[arxiv_id])

    # Get the bytes of a binary vector
    embedding = id_in_db[0]["vector"][0]

    # Run similarity search
    results = search_by_vector(vector=embedding, filter=filter)

    return results


################################################################################


# Search by id. this will first hit the db to get vector
# else use abstract from site to arxiv
@app.get("/search_by_id/{arxiv_id}")
def search_by_id(arxiv_id: str, filter: str = "") -> list[dict]:
    # Search if id is already in database
    id_in_db = milvus_client.get(collection_name=COLLECTION_NAME, ids=[arxiv_id])

    # If the id is already in database
    if bool(id_in_db):
        # Get the bytes of a binary vector
        embedding = id_in_db[0]["vector"][0]

    # If the id is not already in database
    else:
        # Search arxiv for paper details
        arxiv_paper = fetch_arxiv_by_id(arxiv_id)

        # Embed abstract
        embedding = embed_text(arxiv_paper.abstract)

    results = search_by_vector(vector=embedding, filter=filter)

    return results


################################################################################


# Simulate a search point which automatically figures out if the search is using
# id or text
@app.post("/search")
def search(request: TextRequest) -> list[dict]:
    text = request.text
    filter = request.filter

    id_in_text = extract_arxiv_id_from_text(text)

    if id_in_text:
        results = search_by_id(id_in_text, filter)

    else:
        results = search_by_text(request)

    return results
