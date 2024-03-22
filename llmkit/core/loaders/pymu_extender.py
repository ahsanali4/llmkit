import io
from typing import Any, Generator

import fitz
import ocrmypdf
from langchain.docstore.document import Document
from langchain.document_loaders.blob_loaders import Blob

tess = "tesseract stdin stdout --psm 7 -l eng+deu"
mat = fitz.Matrix(5, 5)  # high resolution matrix

# NOTE : Its computationallly too expensive .
# Its going to take around an hour to even do full page OCR of 1000 page book.
# This means line ocr is going to take more than that.
# I would recommend to test the basic pymuloader first from pdf loaders file.


# NOTE : Another approach could be to extract only the images from the doc
# page by page and only use OCR to extract the text from those images.
# This is feasible and quite fast as well depending on th number
# of images and pages.
# Check extract_images function
# https://api.python.langchain.com/en/latest/_modules\
# /langchain_community/document_loaders/parsers/pdf.html#PyMuPDFParser.__init__


class PYMUEXTENDER:
    def __init__(self, file_path: str) -> None:
        """Initialize with file path."""
        self.file_path = file_path

    def read_file(
        self, text_kwargs: dict, use_ocr: bool = False, full_page_ocr: bool = False
    ) -> Generator[Document, Any, Any]:
        """read the file using pymu reader and use ocr if the quality of text is not good.

        Args:
            text_kwargs (dict): option that can be passed to get_text method of pymu .
                        https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_text
            use_ocr (bool, optional): if ocr needs to be done to improve text quality.
                                    Defaults to False. More expensive
            full_page_ocr (bool, optional): Do full page ocr instead of the lines.
                                             Defaults to False. Less expensive

        Yields:
            Generator[Document, Any, Any]: _description_
        """
        blob = Blob.from_path(self.file_path)
        with blob.as_bytes_io() as file_path:
            doc = fitz.open(file_path)  # open document

            yield from [
                Document(
                    page_content=self.extract_text(page, use_ocr, full_page_ocr, text_kwargs),
                    metadata=dict(
                        {
                            "source": blob.source,
                            "file_path": blob.source,
                            "page": page.number,
                            "total_pages": len(doc),
                        },
                        **{
                            k: doc.metadata[k]
                            for k in doc.metadata
                            if type(doc.metadata[k]) in [str, int]
                        },
                    ),
                )
                for page in doc
            ]

    def get_tessocr(self, page, bbox, text_kwargs) -> str:
        """Return OCR-ed span text using Tesseract.

        Args:
            page: fitz.Page
            bbox: fitz.Rect or its tuple
        Returns:
            The OCR-ed text of the bbox.
        """
        global tess, mat
        pix = page.get_pixmap(
            matrix=mat,
            clip=bbox,
        )
        ocrpdf = fitz.open("pdf", pix.pdfocr_tobytes(language="deu"))
        ocrpage = ocrpdf[0]
        text = ocrpage.get_text(**text_kwargs)
        return text

    def ocr_text_extraction(self, page, full_page: bool, text_kwargs: dict) -> str:
        """Use ocr to extract text from the page

        Args:
            page (_type_): page object of the mu reader
            full_page (bool): if full page ocr is required instead of lines with invalid characters
            text_kwargs (dict): option that can be passed to get_text method of pymu.

        Returns:
            str: the original text of the page
        """
        full_text = ""
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_SEARCH, sort=True)["blocks"]
        for b in blocks:
            for line in b["lines"]:
                for s in line["spans"]:
                    text = s["text"]
                    if chr(65533) in text:  # invalid characters encountered!
                        # invoke OCR
                        if full_page:
                            return self.ocr_the_page(page, text_kwargs)
                        text1 = text.lstrip()
                        sb = " " * (len(text) - len(text1))  # leading spaces
                        text1 = text.rstrip()
                        sa = " " * (len(text) - len(text1))  # trailing spaces
                        new_text = sb + self.get_tessocr(page, s["bbox"], text_kwargs) + sa
                        full_text = full_text + new_text
                    else:
                        full_text = "".join([full_text, text])
        return full_text

    def ocr_the_page(self, page, text_kwargs: dict) -> str:
        """Extract the text from passed-in PDF page."""
        src = page.parent  # the page's document
        doc = fitz.open()  # make temporary 1-pager
        doc.insert_pdf(src, from_page=page.number, to_page=page.number)
        pdfbytes = doc.tobytes()
        inbytes = io.BytesIO(pdfbytes)  # transform to BytesIO object
        outbytes = io.BytesIO()  # let ocrmypdf store its result pdf here
        ocrmypdf.ocr(
            inbytes,  # input 1-pager
            outbytes,  # ouput 1-pager
            language="deu",  # modify as required
            output_type="pdf",  # only need simple PDF format
            force_ocr=True
            # add more paramneters, e.g. to enforce OCR-ing, etc.
        )
        ocr_pdf = fitz.open("pdf", outbytes.getvalue())  # read output as fitz PDF
        text = ocr_pdf[0].get_text(**text_kwargs)  # ...and extract text from the page
        return text  # return it

    def extract_text(self, page, use_ocr: bool, full_page_ocr: bool, text_kwargs: dict):
        if use_ocr:
            return self.ocr_text_extraction(page, full_page_ocr, text_kwargs)
        return page.get_text(**text_kwargs)
