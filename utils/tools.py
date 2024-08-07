import io
import os
import tempfile
from typing import List, Optional
import time
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.prompts import WELCOME_PROMPT


def load_file_to_dir(file, suffix: str) -> str | None:
    """
    Saves an uploaded file to a temporary memory location and returns
    the path to the temporary file.

    :param file: The file to save in temporary location for processing.
    :param suffix: The suffix to be added to the temporary file name (e.g. pdf).
    :return: The temporary path of the uploaded file.
    """
    if isinstance(file, io.IOBase):
        # Save uploaded file to a temporary file and get the path
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file.getbuffer())
            file_path = tmp_file.name
        return file_path
    else:
        raise TypeError("Unsupported file format!")


def remove_file(file_path: str) -> None:
    """
    Deletes temporary file from the saved path.
    
    :param file_path: The path to the temporary file.
    """
    os.remove(file_path)
    print("Successfully removed temporary file:", file_path)


def process_file(file_path: str, file_type: str) -> List:
    """
    Processes files adequately according to file type and returns processed
    documents ready for embedding.

    :param file_path: Temporary path to the uploaded file.
    :param file_type: The type of the uploaded file (e.g. docx).
    :return: Processed documents in form of a list.
    :rtype: List
    """
    if file_type.lower() == "docx":
        text = ""
        loader = Docx2txtLoader(file_path)
        docs = loader.load()

        # Append each page to create an individual text string
        # for page in docs:
        #     text += page.page_content

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.split_documents(docs)

        # # print(texts)print(len(texts))
        #
        return texts
    elif file_type.lower() == "pdf":
        text = ""
        loader = PyPDFLoader(file_path)

        pages = loader.load()
        # Append each page to create an individual text string
        # for page in loader.load():
        #     text += page.page_content

        # Replace tab spaces with single spaces (if any)
        # text = text.replace('\t', ' ')

        # Splitting the document into chunks of texts
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        # Create documents from list of texts
        texts = text_splitter.split_documents(pages)

        # print(len(texts))
        # print(texts)
        return texts


def stream_data(data: Optional[str] = None):
    """
    Creates a stream text output. Displays the introduction text if no parameter is passed.
    :param data: The text to stream.
    """
    if not data:
        for word in WELCOME_PROMPT.split(" "):
            yield word + " "
            time.sleep(0.1)
    else:
        for word in data.split(" "):
            yield word + " "
            time.sleep(0.1)

# doc = process_file(file_path=r"C:\Users\Deju\Downloads\human-resources-resume-template.docx", file_type="docx")
