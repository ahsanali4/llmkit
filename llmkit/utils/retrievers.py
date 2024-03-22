import logging
from typing import List

from langchain.docstore.document import Document
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers import (
    BM25Retriever,
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import BaseRetriever

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


def get_bm25_retriever(texts: List[Document], number_of_docs: int = 4):
    bm25_retriever = BM25Retriever.from_documents(texts)
    bm25_retriever.k = number_of_docs
    return bm25_retriever


def get_ensemble_retriever(retreivers: List[BaseRetriever], weights: List[float]):
    if len(retreivers) == len(weights):
        return EnsembleRetriever(retrievers=retreivers, weights=weights)
    else:
        print("number of given retreivers and weights are not matching")


# EmbeddingsFilter
# Making an extra LLM call over each retrieved document is expensive and slow.
# The EmbeddingsFilter provides a cheaper and faster option by embedding the documents
# and query and only returning those documents which have sufficiently similar embeddings
# to the query.
def get_compression_retreiver(text_splitter, embeddings, retreiver):
    redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
    relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.60)
    pipeline_compressor = DocumentCompressorPipeline(
        transformers=[text_splitter, redundant_filter, relevant_filter]
    )
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=retreiver
    )
    return compression_retriever


# MultiQueryRetriever
# Distance-based vector database retrieval embeds (represents) queries in high-dimensional space
# and finds similar embedded documents based on “distance”.
# But, retrieval may produce different results with subtle changes in query wording or
# if the embeddings do not capture the semantics of the data well.
# Prompt engineering / tuning is sometimes done to manually address these problems,
# but can be tedious.

# The MultiQueryRetriever automates the process of prompt tuning by using an
# LLM to generate multiple queries from different perspectives for a given user input query.
# For each query, it retrieves a set of relevant documents and takes the unique union across
# all queries to get a larger set of potentially relevant documents.
# By generating multiple perspectives on the same question,
# the MultiQueryRetriever might be able to overcome some of the limitations of
# the distance-based retrieval and get a richer set of results.


def get_multi_query_retreiver(retreiver, llm):
    retriever = MultiQueryRetriever.from_llm(retriever=retreiver, llm=llm)
    return retriever
