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
from sentence_transformers import SentenceTransformer

from backend.schemas import ArxivPaper, SearchResult, TextRequest

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
# Load the embedding model
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")


# Function to binarize the float embeddings
def dense_to_binary(dense_vector: np.ndarray) -> bytes:
    return np.packbits(np.where(dense_vector >= 0, 1, 0)).tobytes()


# Function to unpack the bytes created using the dense_to_binary function
def binary_to_dense(binary_vector: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(binary_vector, dtype=np.uint8))


# Function to embed text using https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
@cache
def embed_text(text: str, binarise: bool = True) -> np.ndarray | bytes:
    try:
        # TODO: Verify the API results, cannot do it right now since no credits
        # Call the MixedBread.ai API to generate the embedding
        result = mxbai.embed(
            model="mixedbread-ai/mxbai-embed-large-v1",
            input=text,
            normalized=True,
            encoding_format="float",
            dimensions=1024,
        )

        # Extract the embedding from the response
        embedding = result.data[0].embedding

        if binarise:
            # Convert the embedding to bytes
            embedding = dense_to_binary(embedding)

        # Otherwise return float embeddings
        return embedding
    except:
        # Generate the embedding from the locally loaded model
        embedding = model.encode(
            text,
            precision="float32",
            convert_to_numpy=True,
        )

        # Binarize and return the bytes for futher search
        if binarise:
            # Convert the embedding to bytes
            embedding = dense_to_binary(embedding)

        # Otherwise return float embeddings
        return embedding


################################################################################
def transform_result_vector(result: dict) -> dict:
    """
    Function to convert the bytes in the milvus search results to a binary numpy array of uint8 dtype
    """
    result["entity"]["vector"] = binary_to_dense(result["vector"]).tolist()
    return result


# Single vector search
def search_by_vector(
    vector: bytes,
    filter: str = "",
    search_limit: int = SEARCH_LIMIT,
    return_vector: bool = False,
) -> list[SearchResult]:

    output_fields = [
        "id",
        "title",
        "abstract",
        "authors",
        "categories",
        "month",
        "year",
        "url",
    ]

    if return_vector:
        output_fields = output_fields + ["vector"]

    # Request zilliz for the vector search
    result = milvus_client.search(
        collection_name=COLLECTION_NAME,  # Collection to search in
        data=[vector],  # Vector to search for
        limit=search_limit,  # Max. number of search results to return
        output_fields=output_fields,  # Output fields to return
        filter=filter,  # Filter to apply to the search
    )

    if return_vector:
        results = [transform_result_vector(result) for result in result[0]]
    else:
        results = result[0]

    search_results = [
        SearchResult.model_validate(search_result) for search_result in results
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


# TODO: Fix inconsistent inputs. Some functions are taking TextRequest, some are taking values
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


# TODO: Fix inconsistent inputs. Some functions are taking TextRequest, some are taking values
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
        results = search_by_id(
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


def compose_reranking_query(text: str) -> str:
    """
    This function helps maintain the search by arXiv ID and search by text workflow.
    If there is ID in the text, it first tries to fetch the abstract from DB, otherwise
    it gets the results from arxiv itself.

    If there is no arXiv ID, then it simply returns the text as is.
    """
    id_in_text = extract_arxiv_id_from_text(text)

    if id_in_text:
        # Search if id is already in database
        id_in_db = milvus_client.get(collection_name=COLLECTION_NAME, ids=[id_in_text])
        print("Printing id_in_db:")
        print(id_in_db)

        # If the id is already in database
        if bool(id_in_db):
            # Get the abstract from db itself
            abstract = id_in_db[0]["abstract"]

            return abstract
        # If the id is not already in database
        else:
            # Search arxiv for paper details
            arxiv_paper = fetch_arxiv_by_id(id_in_text)

            # And then return the fetched abstract
            return arxiv_paper.abstract
    # If no arxiv id is found in text, simply return that to the reranker
    else:
        return text


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
    # We can't return plain text, as the user can also
    # search using the arXiv ID.
    query = compose_reranking_query(request.text)

    # Rerank the search results
    reranked_search_results = rerank_search_results(
        query,
        serialise_for_reranker(search_results),
    )

    return reranked_search_results


################################################################################


def rerank_search_results_yamada(
    float_query_vector: np.ndarray,
    binary_search_results: list[SearchResult],
    top_k: int = SEARCH_LIMIT,
) -> list[SearchResult]:

    # Normalise the vectors so that the dot products are 
    # between -1 and 1
    float_query_vector_normalised = float_query_vector/np.linalg.norm(float_query_vector)

    # Iterate over the search results
    for search_result in binary_search_results:
        # Extract the binary vector
        binary_vector = search_result.entity.vector

        # Normalise the vectors so that the dot products are 
        # between -1 and 1
        binary_vector_normalised = binary_vector/np.linalg.norm(binary_vector)

        # Calculate the dot product and then linearly map the answer
        # from the domain [-1024,1024] to [0,1024]. Enforce int for pydantic model
        search_result.distance = np.dot(float_query_vector_normalised, binary_vector_normalised)

        # TODO: Fix the hack, delete vector altogether
        # Empty the binary vector to present the results properly
        search_result.entity.vector = []

    # Sort the results using the newly calculated dot products
    # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    reranked_results = sorted(
        binary_search_results,
        key=lambda search_result: search_result.distance,
        reverse=True
    )

    # Return the ranked results upto top_k
    return reranked_results[:top_k]


# Rerank the search using Yamada et al. (2021) https://arxiv.org/abs/2106.00882
@app.post("/reranked_search_yamada")
def reranked_search_yamada(request: TextRequest) -> list[SearchResult]:
    """
    Function to use the reranking trick introduced by Yamada et al.
    Here, we first retreive `multiplier * top_k` results
    """

    # Extract arXiv ID from text
    id_in_text = extract_arxiv_id_from_text(request.text)

    # Do not do any re-ranking if the search is preformed using ID
    if id_in_text:
        return search_by_id(
            id_in_text,
            request.filter,
            request.search_limit,
        )

    # Now that the request only contains english text (and not arXiv ID),
    # we can generate the float embeddings
    float_query_vector = np.array(
        embed_text(request.text, binarise=False),
        dtype=np.float32,
    )

    # Convert the float vector to binary to perform similarity search
    binary_query_vector = dense_to_binary(float_query_vector)
    # print(f"{binary_query_vector = }")

    # Increase the search limit and perform vector search
    binary_search_results = search_by_vector(
        vector=binary_query_vector,
        filter=request.filter,
        search_limit=RERANK_INPUT_SEARCH_LIMIT,
        return_vector=True,
    )

    # Rerank the results
    reranked_results = rerank_search_results_yamada(
        float_query_vector,
        binary_search_results,
        request.search_limit,
    )

    return reranked_results
