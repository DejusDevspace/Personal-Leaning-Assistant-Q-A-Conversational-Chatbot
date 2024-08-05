from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# import streamlit as st
from typing import (List, Dict)
from dotenv import load_dotenv
import os

from utils import tools, prompts

load_dotenv()

LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

EMBEDDINGS = HuggingFaceInferenceAPIEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
)

GOOGLE_EMBEDDINGS = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
)


class PersonalLearningAssistant:
    def __init__(self, llm=LLM, embeddings=GOOGLE_EMBEDDINGS, prompts_=prompts):
        self.llm = llm
        self.embeddings = embeddings
        self.prompts = prompts_
        # self.vectorstore = self.initialize_vectorstore()

    # @st.cache_resource
    def initialize_vectorstore(self, file=None, file_type=None):
        """
        Loads an uploaded file and embeds it in a vectorstore for reference.

        :param file: The uploaded file to load to vectorstore.
        :param file_type: The type of the uploaded file.
        """
        try:
            # Load the uploaded file to temporary location
            file_path = tools.load_file_to_dir(file, file_type)
            # File processing...
            docs = tools.process_file(file_path, file_type)

            # Create FAISS vectorstore from documents.
            vectorstore = FAISS.from_documents(docs, self.embeddings)
            return vectorstore
        except Exception as e:
            print("Error:", e)

    def initialize_retrieval_chain(self, retriever):
        """
        Creates a retrieval chain for retrieval augmented generation.
        :param retriever: The vectorstore object as the retriever
        :return: Conversational retrieval chain
        """
        # Query transformation prompt
        q_transform_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
            ("user", self.prompts.CTQ_PROMPT),
        ])

        # Create the query transformation chain
        q_transform_chain = create_history_aware_retriever(
            llm=self.llm,
            retriever=retriever,
            prompt=q_transform_prompt,
        )

        # Question-answer prompt
        q_answer_prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompts.QA_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{question}"),
        ])

        # Create documents chain
        document_chain = create_stuff_documents_chain(llm=self.llm, prompt=q_answer_prompt)

        # Create retrieval chain
        retrieval_chain = create_retrieval_chain(q_transform_chain, document_chain)
        return retrieval_chain

    def query(self, question: str, chat_history: List) -> Dict:
        """
        Accepts query from the user and invokes a retrieval chain to get information
        from the uploaded document.
        :param question: Question regarding the uploaded documents or previous conversation.
        :param chat_history: Chat history from current session state
        :return: Response
        """
        try:
            retriever = self.vectorstore.as_retriever()
            chain = self.initialize_retrieval_chain(retriever)
            # Invoke and return the chain
            return chain.invoke({"question": question, "chat_history": chat_history})
        except Exception as e:
            print("Error:", e)
