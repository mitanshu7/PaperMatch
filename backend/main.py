# Import required libraries
import os
import re
from datetime import datetime
from functools import cache

import arxiv
import backoff
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mixedbread import Mixedbread
from mixedbread.types.rerank_response import Data
from pymilvus import MilvusClient
from schemas import ArxivPaper, SearchResult, TextRequest

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
RERANK_INPUT_SEARCH_LIMIT = int(os.getenv("RERANK_INPUT_SEARCH_LIMIT"))

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

    # Create the result model instance
    arxiv_paper = ArxivPaper(
        id=extract_arxiv_id_from_url(paper.entry_id),
        title=paper.title.replace("\n", " "),
        authors=[str(author) for author in paper.authors],
        abstract=paper.summary.replace("\n", " "),
        url=paper.entry_id,
        pdf=paper.pdf_url,
        month=paper.published.month,
        year=paper.published.year,
        categories=paper.categories,
    )

    return arxiv_paper


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
def search_by_vector(
    vector: bytes,
    filter: str = "",
    search_limit: int = SEARCH_LIMIT,
) -> list[SearchResult]:
    # Request zilliz for the vector search
    result = milvus_client.search(
        collection_name=COLLECTION_NAME,  # Collection to search in
        data=[vector],  # Vector to search for
        limit=search_limit,  # Max. number of search results to return
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

    search_results = [
        SearchResult.model_validate(search_result) for search_result in result[0]
    ]

    # returns a list of dictionaries with id and distance as keys
    return search_results


################################################################################


# Search the collection using text
@app.post("/search_by_text")
def search_by_text(request: TextRequest) -> list[SearchResult]:
    # Extract objects?
    text = request.text
    filter = request.filter
    search_limit = request.search_limit

    # Embed the text
    embedding = embed_text(text)

    # Send vector for search
    results = search_by_vector(
        vector=embedding,
        filter=filter,
        search_limit=search_limit,
    )

    return results


################################################################################


# Search by known id
# The onus is on the user to make sure the id exists
# Use with similar results feature
@app.get("/search_by_known_id/{arxiv_id}")
def search_by_known_id(
    arxiv_id: str,
    filter: str = "",
    search_limit: int = SEARCH_LIMIT,
) -> list[SearchResult]:
    # Get the id which is already in database
    id_in_db = milvus_client.get(collection_name=COLLECTION_NAME, ids=[arxiv_id])

    # Get the bytes of a binary vector
    embedding = id_in_db[0]["vector"][0]

    # Run similarity search
    results = search_by_vector(
        vector=embedding,
        filter=filter,
        search_limit=search_limit,
    )

    return results

################################################################################


# Search by id. this will first hit the db to get vector
# else use abstract from site to arxiv
@app.get("/search_by_unknown_id/{arxiv_id}")
def search_by_unknown_id(
    arxiv_id: str,
    filter: str = "",
    search_limit: int = SEARCH_LIMIT,
) -> list[SearchResult]:

    # Search arxiv for paper details
    arxiv_paper = fetch_arxiv_by_id(arxiv_id)

    # Embed abstract
    embedding = embed_text(arxiv_paper.abstract)

    results = search_by_vector(
        vector=embedding,
        filter=filter,
        search_limit=search_limit,
    )

    return results


################################################################################


# Search by id. this will first hit the db to get vector
# else use abstract from site to arxiv
@app.get("/search_by_id/{arxiv_id}")
def search_by_id(
    arxiv_id: str,
    filter: str = "",
    search_limit: int = SEARCH_LIMIT,
) -> list[SearchResult]:
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

    results = search_by_vector(
        vector=embedding,
        filter=filter,
        search_limit=search_limit,
    )

    return results


################################################################################


# Simulate a search point which automatically figures out if the search is using
# id or text
@app.post("/search")
def search(request: TextRequest) -> list[SearchResult]:
    text = request.text
    filter = request.filter
    search_limit = request.search_limit

    id_in_text = extract_arxiv_id_from_text(text)

    if id_in_text:
        results = search_by_unknown_id(
            id_in_text,
            filter,
            search_limit,
        )

    else:
        results = search_by_text(request)

    return results


################################################################################


# @app.post("/rerank")
# def rerank(
#     query: str,
#     documents: list[str],
#     top_k: int = SEARCH_LIMIT,
# ):
#     response = mxbai.rerank(
#         model="mixedbread-ai/mxbai-rerank-large-v2",
#         query=query,
#         input=documents,
#         top_k=top_k,
#         return_input=True,
#     )

#     return response.data


################################################################################


def prettify_rerank_search_results(rerank_results: list[Data]):
    """
    The reranker has extra fields in response, we only need the data we originally
    inputted. Hence we extract the `input` attribute from the results.
    """
    pretty_data = [search_result.input for search_result in rerank_results]
    return pretty_data


@app.post("/rerank_search_results")
def rerank_search_results(
    query: str,
    documents: list[dict],
    rank_fields: list[str] = ["entity.abstract"],
    top_k: int = SEARCH_LIMIT,
) -> list[SearchResult]:
    """
    Rerank search results using mixedbread's reranker
    """
    response = mxbai.rerank(
        model="mixedbread-ai/mxbai-rerank-large-v2",
        query=query,
        input=documents,
        top_k=top_k,
        rank_fields=rank_fields,
        return_input=True,
    )

    rerank_results = response.data

    return prettify_rerank_search_results(rerank_results)


################################################################################


def serialise_for_reranker(search_results: list[SearchResult]) -> list[dict]:
    """
    Function to create a list of dicts from the search results as sending the
    Pydantic model alone results in loss of information from Mixedbread's side.
    """
    serialised_search_results = [
        search_result.model_dump() for search_result in search_results
    ]
    return serialised_search_results


# Rerank the search
@app.post("/reranked_search")
def reranked_search(request: TextRequest) -> list[SearchResult]:
    """
    Function to wrap all the above functions and behave (in request and response)
    same as the search endpoint.
    """

    # Increase the search limit for semantic search
    request.search_limit = RERANK_INPUT_SEARCH_LIMIT

    # Perform regular semantic search
    search_results = search(request)

    # Extract user query from request
    query = request.text

    # Rerank the search results
    reranked_search_results = rerank_search_results(
        query,
        serialise_for_reranker(search_results),
    )

    return reranked_search_results
