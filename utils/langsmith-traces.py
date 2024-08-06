from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv

from langsmith import traceable
from utils import prompts

_ = load_dotenv(find_dotenv())

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def main():
    simple_retrieval_chain()


@traceable(
    run_type="llm",
    name="Assistant Retrieval Chain",
    tags=["assistantretrieval"],
    metadata={"chainname": "assistantretrieval"}
)
def simple_retrieval_chain():
    doc_path = r"C:\Users\Deju\Desktop\home\Bowen\100 lvl\2nd Semester\EES 102 Fundamentals Of Enterprenuership 3.pdf"
    loader = PyPDFLoader(doc_path)

    docs = loader.load()

    text_splitter = load_splitter(chunk_size=1000)
    documents = text_splitter.split_documents(docs)

    vector = FAISS.from_documents(documents, embeddings)
    retriever = load_retriever(vector, k=3)

    prompt = ChatPromptTemplate.from_template(prompts.QA_SYSTEM_PROMPT)
    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": "Summarize Alibaba success story for me"})
    print("\nSimple Retrieval Chain\n", "-" * 50, "\n", response["answer"])


def load_splitter(chunk_size=500, chunk_overlap=50):
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


def load_retriever(vector, search_type="mmr", k=5):
    return vector.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k},
    )


if __name__ == "__main__":
    main()
