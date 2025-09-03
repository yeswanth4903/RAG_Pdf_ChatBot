#PDF ingestion
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import tempfile
import os

load_dotenv()

groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

if not groq_api_key:
    st.error("❌ Missing GROQ_API_KEY. Set it in Streamlit secrets or in a local .env file.")
    st.stop()

st.title("Chat With Your PDF")
uploaded_file = st.file_uploader("Upload PDF",type='pdf')

if uploaded_file is not None:
    st.success("PDF Uploaded Sucessfully!!!")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size = 1000 , chunk_overlap = 100)
        chunks = splitter.split_documents(docs)


        #Embedding and vector_store
        emb = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(
            chunks,
            embedding=emb
        )



        #Retriever with RAG chain
        llm = ChatGroq(model="gemma2-9b-it",temperature=0.2) 

        retriever = RetrievalQA.from_chain_type(
            llm=llm,
            retriever = vector_store.as_retriever(),
            chain_type = "stuff"
        )
        query = st.text_input("Ask your query: ")
        if(query):
            ans = retriever.invoke(query)
            st.write("AI: ",ans['result'])
else:
    print("Something went wrong,Please Upload Again!!")