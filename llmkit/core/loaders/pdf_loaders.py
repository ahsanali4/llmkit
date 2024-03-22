from typing import List, Optional, Union

import fitz
from langchain.document_loaders import (
    OnlinePDFLoader,
    PDFPlumberLoader,
    PyMuPDFLoader,
    PyPDFDirectoryLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from langchain.schema import Document
from langchain.text_splitter import TextSplitter


def get_pypdfloader(
    file_path: str, load_pages=True, text_spliter: Optional[TextSplitter] = None
) -> Union[List[Document], PyPDFLoader]:
    loader = PyPDFLoader(file_path)
    if load_pages:
        pages = loader.load_and_split(text_spliter)
        return pages
    return loader


def get_unstructured_pdf_loader(file_path: str) -> List[Document]:
    loader = UnstructuredPDFLoader(file_path)
    data = loader.load()
    return data


def get_online_pdfs(url: str) -> List[Document]:
    loader = OnlinePDFLoader(url)
    data = loader.load()
    return data


def get_pymupdf_loader(
    file_path: str, extract_images=True, flags=fitz.TEXTFLAGS_SEARCH
) -> List[Document]:
    loader = PyMuPDFLoader(file_path, extract_images=True)
    data = loader.load(flags=flags, sort=True)
    return data


def get_pypdf_directory_loader(directory_path: str) -> List[Document]:
    loader = PyPDFDirectoryLoader(directory_path)
    docs = loader.load()
    return docs


def get_pdf_plumber(file_path: str) -> List[Document]:
    loader = PDFPlumberLoader(file_path)
    data = loader.load()
    return data
