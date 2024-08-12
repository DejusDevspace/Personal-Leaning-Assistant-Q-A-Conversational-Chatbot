import os
# import streamlit as st
# from typing import (List, Dict)
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
# from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate, PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
# from langchain.utils.math import cosine_similarity
# from langchain_chroma import Chroma
from typing import List
# from operator import itemgetter

from utils import prompts

load_dotenv()

GEMINI_LLM = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2,
)

# llama-3.1-70b-versatile
# mixtral-8x7b-32768
# llama3-70b-8192

GROQ_LLM = ChatGroq(
    # model="llama3-70b-8192",
    temperature=0.3,
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
    def __init__(self, model=None, temperature=0, embeddings=GOOGLE_EMBEDDINGS, _prompts=prompts):
        if model:
            try:
                self.llm = ChatGroq(
                    model=model,
                    temperature=temperature,
                )
            except Exception as e:
                print(f"Error loading model {model}:", e)
        else:
            self.llm = GEMINI_LLM
        self.embeddings = embeddings
        self.prompts = prompts
        self.retriever = None

    # @st.cache_resource
    def load_vectorstore_as_retriever(self, documents, search_type="mmr", k=5):
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
        retriever = vector.as_retriever(

            search_type=search_type,
            search_kwargs={"k": k}
        )
        self.retriever = retriever

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
            ("system", self.prompts.RAG_TEMPLATE),
            MessagesPlaceholder(variable_name="chat_history"),
        ])

        # Document chain: feed documents to the pipeline
        document_chain = create_stuff_documents_chain(self.llm, qa_prompt)

        # Final chain: retrieval chain
        retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
        return retrieval_chain

    def initialize_llm_chain(self) -> Runnable:
        """
        Creates a chain for the llm to respond to a user's query
        :return: Runnable: LLM chain
        """
        # Prompt for handling chat history
        contextualize_prompt = ChatPromptTemplate.from_messages([
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            ("user", self.prompts.CTQ_PROMPT),
        ])
        contextualize_chain = contextualize_prompt | self.llm

        prompt = PromptTemplate.from_template(
            self.prompts.LLM_TEMPLATE,
            partial_variables={"chat_history": "chat_history"},
        )

        llm_chain = prompt | self.llm | StrOutputParser()

        final_chain = contextualize_chain | llm_chain
        return final_chain

    def route(self, question: str, chat_history: List) -> Runnable:
        """
        Takes a user's question and routes it to the suitable chain.
        :param question: The user's input message to determine the chain to handle the inquiry.
        :param chat_history: List containing previous conversations (if any)
        :return: The chain the user's inquiry is routed to.
        """
        chain = (
                ChatPromptTemplate.from_messages([
                    ("system", self.prompts.ROUTE_TEMPLATE),
                    MessagesPlaceholder(variable_name="chat_history")
                ])
                | self.llm
                | StrOutputParser()
        )
        response = chain.invoke({"input": question, "chat_history": chat_history})
        # print(response)
        if "general" in response.lower():
            chain = self.initialize_llm_chain()
            return chain
        elif "retrieval" in response.lower():
            if self.retriever:
                chain = self.initialize_retrieval_chain(self.retriever)
                return chain
            else:
                chain = self.initialize_llm_chain()
                return chain

    def query(self, question: str, chat_history: List) -> str:
        """
        Accepts a query from the user and returns the response to the query.
        :param question: User's query.
        :param chat_history: List of previous conversations (if any).
        :return: The AI's response to the user's query.
        """
        # print(chat_history)
        chain = self.route(question, chat_history)
        # print(chat_history)
        response = chain.invoke({
            "input": question,
            "chat_history": chat_history
        })
        if not isinstance(response, str):
            return response["answer"]
        return response

    # def prompt_router(self, input_):
    #     llm_query = self.prompts.LLM_TEMPLATE
    #     rag_query = self.prompts.RAG_TEMPLATE
    #     print(input_)
    #     print("_" * 50)
    #
    #     prompt_templates = [llm_query, rag_query]
    #
    #     query = input_["input"]
    #     history = input_["chat_history"]
    #
    #     prompt_embeddings = self.embeddings.embed_documents(prompt_templates)
    #     query_embedding = self.embeddings.embed_query(query)
    #
    #     similarity = cosine_similarity([query_embedding], prompt_embeddings)[0]
    #     most_similar = prompt_templates[similarity.argmax()]
    #     print("Retrieval Query" if most_similar == rag_query else "LLM Query")
    #
    #     if most_similar == rag_query:
    #         if self.retriever:
    #             try:
    #                 retrieval_chain = self.initialize_retrieval_chain(self.retriever)
    #                 output = retrieval_chain.invoke({"input": query, "chat_history": history})
    #                 return output
    #             except Exception as e:
    #                 print("Error accessing retriever:", e)
    #         else:
    #             print("No document loaded to vectorstore!")
    #     else:
    #         contextualize_prompt = ChatPromptTemplate.from_messages([
    #             MessagesPlaceholder(variable_name="chat_history"),
    #             ("user", "{input}"),
    #             ("user", self.prompts.CTQ_PROMPT),
    #         ])
    #         contextualize_chain = contextualize_prompt | self.llm
    #         prompt = ChatPromptTemplate.from_template(llm_query)
    #
    #         llm_chain = contextualize_chain | prompt | self.llm
    #         response = llm_chain.invoke({"input": query, "chat_history": history})
    #         return response

    # chain = (
    #         {
    #             "input": itemgetter("input"),
    #             "chat_history": itemgetter("chat_history")
    #         }
    #         # | RunnableLambda(self.prompt_router)
    #         | StrOutputParser()
    # )
    # return chain.invoke({"input": question, "chat_history": chat_history})
