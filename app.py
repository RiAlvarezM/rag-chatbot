# Streamlit Chatbot APP
############################################################
# Libraries 
import streamlit as st
import os

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceHubEmbeddingss
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

############################################################
HUFFING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
CHROMA_DB_DIR = "embeddings/test"

############################################################
# Streamlit Interface
st.set_page_config(page_title="RAG Gen AI Chatbot Demo", layout="wide")
st.title("Demo - 📄🤖 Gen AI Chatbot with PDF document understanding")


############################################################
# Functions
# PDF Loader Function
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

# Text Splitter
def split_text(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return text_splitter.split_documents(docs)

# Emedding loader
def get_embeddings():
    return HuggingFaceHubEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HUFFING_FACE_TOKEN
        )

# Create embeddings using FAISS
def create_vector_db(texts):
    embeddings = get_embeddings()
    db = FAISS.from_documents(
        texts, 
        embeddings)

    return db

# GET LLM model
def get_llm():
    return HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 0.3, "max_length": 256})


# Crear el chatbot con RAG
def create_chatbot(db):
    retriever = db.as_retriever()
    llm = get_llm()
    return RetrievalQA(llm=llm, retriever=retriever)



############################################################
# Subir archivo PDF
uploaded_file = st.file_uploader("📂 Upload your PDF", type=["pdf"])

if uploaded_file:
    st.write("📚 Processing the file...")

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
        # 1. PDF Loading
        docs = load_pdf("temp.pdf")

        # 2. Splitting the PDF for Vector Embeddings
        texts = split_text(docs)

        # 3. Storing Vector Embeddings
        db = create_vector_db(texts)

        # 4. Create the chatbot
        chatbot = create_chatbot(db)