from typing import List, Optional

from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma


def load_text_documents(text_file: str = "data/Heidelberg.txt"):
    loader = TextLoader(text_file, autodetect_encoding=True)
    return loader.load()


def text_splitter(
    documents: List[Document],
    separators: Optional[List[str]] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """recursively try to eliminate text based on the given seperators

    Args:
        documents (List[Document]): list of documents
        separators (Optional[List[str]], optional): custom seperators. Defaults to None.
        chunk_size (int, optional): chunk size of single document. Defaults to 1000.
        chunk_overlap (int, optional): overlap between chunks. Defaults to 200.

    Returns:
        List[Document]: chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators
    )
    texts = text_splitter.split_documents(documents)
    return texts


def create_chroma_db(
    embedding, texts: list, collection_name: str = "langchain", persist_directory: str = "db"
) -> Chroma:
    """Embed and store the texts. Supplying a persist_directory will store the embeddings on disk


    Args:
        embedding (_type_): _description_
        texts (list): actual list fo documents
        collection_name (str, optional): collection name for storage. Defaults to 'langchain'.
        persist_directory (str, optional): local path to store the db. Defaults to 'db'.

    Returns:
        Chroma: instance of chromadb
    """

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )

    return vectordb


def load_chroma_db(embedding_model, collection_name: str, persist_directory: str):
    return Chroma(
        embedding_function=embedding_model,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
