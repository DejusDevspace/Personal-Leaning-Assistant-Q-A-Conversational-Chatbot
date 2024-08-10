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
# from langchain_core.output_parsers import StrOutputParser
# from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

from langsmith import traceable
from utils import prompts

_ = load_dotenv(find_dotenv())

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)
# llm = ChatGroq(
#     model="llama3-70b-8192",
#     temperature=0.3,
# )
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def main():
    conversational_retrieval_chain()
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
    retriever = load_retriever(vector)

    prompt = ChatPromptTemplate.from_template(prompts.RAG_TEMPLATE)
    document_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({"input": "Summarize Alibaba success story for me"})
    pretty_print("Simple Retrieval Chain", response["answer"])


@traceable(
    run_type="llm",
    name="Assistant Conversational Chain",
    tags=["assistantconvchain"],
    metadata={"chainname": "assistantconvchain"}
)
def conversational_retrieval_chain():
    doc_path = r"C:\Users\Deju\Desktop\home\Bowen\400 lvl\MCE 415\MCE415 note 1_022800.pdf"
    loader = PyPDFLoader(doc_path)

    docs = loader.load()

    text_splitter = load_splitter()
    documents = text_splitter.split_documents(docs)

    vector = FAISS.from_documents(documents, embeddings)
    retriever = load_retriever(vector, search_type="similarity", k=4)

    contextualize_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", prompts.CTQ_PROMPT)
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # Sample question and answer as chat history
    sample_question = "What is weak AI?"
    sample_answer = """
    Weak AI is the view that intelligent behavior can be modeled and used by 
    computers to solve complex problems.
    """

    chat_history = [
        HumanMessage(content=sample_question),
        AIMessage(content=sample_answer),
    ]

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", prompts.RAG_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
    ])
    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

    response = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": "How does it compare to strong AI?"
    })
    pretty_print("Conversational Retrieval Chain", response["answer"])


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


def pretty_print(name: str, text: str) -> None:
    print("\n{name}\n".format(name=name), "-" * 50, "\n", text)


if __name__ == "__main__":
    main()
