import io
import os
import tempfile
from typing import List, Optional
import time
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.utils.math import cosine_similarity
# from langchain_core.prompts import PromptTemplate
#
# from learning_assistant import PersonalLearningAssistant
from utils import prompts
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())


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
        for page in docs:
            text += page.page_content

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        texts = text_splitter.create_documents([text])

        # # print(texts)print(len(texts))
        #
        return texts
    elif file_type.lower() == "pdf":
        text = ""
        loader = PyMuPDFLoader(file_path)

        # pages = loader.load()
        # Append each page to create an individual text string
        for page in loader.load():
            text += page.page_content

        # Replace tab spaces with single spaces (if any)
        text = text.replace('\t', ' ')

        # Splitting the document into chunks of texts
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
            is_separator_regex=False,
        )
        # Create documents from list of texts
        texts = text_splitter.create_documents([text])

        # print(len(texts))
        # print(texts)
        return texts


def stream_data(data: Optional[str] = None):
    """
    Creates a stream text output. Displays the introduction text if no parameter is passed.
    :param data: The text to stream.
    """
    if not data:
        for word in prompts.WELCOME_TEXT.split(" "):
            yield word + " "
            time.sleep(0.1)
    else:
        for word in data.split(" "):
            yield word + " "
            time.sleep(0.1)

# doc = process_file(file_path=r"C:\Users\Deju\Downloads\human-resources-resume-template.docx", file_type="docx")

# physics_template = """You are a very smart physics professor. \
# You are great at answering questions about physics in a concise and easy to understand manner. \
# When you don't know the answer to a question you admit that you don't know.
#
# Here is a question:
# {query}"""
#
# math_template = """You are a very good mathematician. You are great at answering math questions. \
# You are so good because you are able to break down hard problems into their component parts, \
# answer the component parts, and then put them together to answer the broader question.
#
# Here is a question:
# {query}"""

# -------------------------------- PROMPT ROUTING -------------------------------------- #
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# embeddings = HuggingFaceInferenceAPIEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-l6-v2",
#     api_key=os.getenv("HUGGINGFACE_API_KEY"),
# )

# llm_query = prompts.LLM_TEMPLATE
# rag_query = prompts.RAG_TEMPLATE
#
# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#
# prompt_templates = [llm_query, rag_query]
# prompt_embeddings = embeddings.embed_documents(prompt_templates)
#
# assistant = PersonalLearningAssistant()
#
#
# def prompt_router(input_):
#     query_embedding = embeddings.embed_query(input_["input"])
#     similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
#     most_similar = prompt_templates[similarity.argmax()]
#     print("Retrieval Query" if most_similar == rag_query else "LLM Query")
#     if most_similar == rag_query:
#         if assistant.retriever:
#             try:
#                 return assistant.initialize_retrieval_chain(self.retrieve)
#             except Exception as e:
#                 print("Error accessing retriever:", e)
#         else:
#             print("No document loaded to vectorstore!")
#     return PromptTemplate.from_template(llm_query)
