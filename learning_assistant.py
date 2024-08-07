import os
# import streamlit as st
# from typing import (List, Dict)
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma

from utils import prompts

load_dotenv()

GEMINI_LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

# llama-3.1-70b-versatile
# mixtral-8x7b-32768

GROQ_LLM = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
)

OLLAMA_LLM = ChatOllama(
    model="llama3",
    temperature=0,
)

GOOGLE_EMBEDDINGS = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

OPENAI_EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-large")

HUGGINGFACE_EMBEDDINGS = HuggingFaceInferenceAPIEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
)

OLLAMA_EMBEDDINGS = OllamaEmbeddings(model="llama3")


class PersonalLearningAssistant:
    def __init__(self, llm=GROQ_LLM, embeddings=GOOGLE_EMBEDDINGS, _prompts=prompts):
        self.llm = llm
        self.embeddings = embeddings
        self.prompts = prompts

    def load_vectorstore_as_retriever(self, documents, search_type="mmr", k=4):
        """
        Loads documents into a vectorstore and returns a retriever object with the documents
        saved as embeddings
        :param documents: The documents to store in the vector database.
        :param search_type: The search criteria for the retriever.
        :param k: The number of retrieved documents for a query.
        :return: A vectorstore as a retriever with uploaded documents embedded.
        """
        # Create vectorstore and store documents as embeddings
        vector = FAISS.from_documents(documents, self.embeddings)
        return vector.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

    def initialize_retrieval_chain(self, retriever) -> Runnable:
        """
        Creates a chain for retrieval of information from a vectorstore.
        :param retriever: The retriever object with the stored information.
        :return: Runnable: Retrieval chain
        """
        # Prompt for handling chat history
        contextualize_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", self.prompts.CTQ_PROMPT),
        ])

        retriever_chain = create_history_aware_retriever(
            llm=self.llm,
            retriever=retriever,
            prompt=contextualize_prompt,
        )

        # Question-answer prompt
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompts.QA_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
        ])

        # Document chain: feed documents to the pipeline
        document_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        # Final chain: retrieval chain
        retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
        return retrieval_chain

# test = PersonalLearningAssistant()
# test.test()
