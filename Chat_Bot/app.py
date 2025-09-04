#PDF ingestion
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
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

        #LLM
        llm = ChatGroq(model="gemma2-9b-it",temperature=0.2) 

        #Memory
        memory = ConversationBufferMemory(memory_key="chat_history" , return_messages=True)

        #Retriever with RAG chain
        retriever = RetrievalQA.from_chain_type(
            llm=llm,
            retriever = vector_store.as_retriever(),
            chain_type = "stuff"
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []  

        prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="""
            You are an intelligent assistant for a PDF chatbot. 
            Your job is to validate user queries before searching the PDF.

            Rules:
            1. If the query is relevant to the provided PDF context, return a cleaned and well-structured version of the query.
            2. If the query is irrelevant (e.g., asking about world news when PDF is about medicine), respond with: "This question is not related to the document."
            3. If the query is vague, reframe it into a clearer query.

            PDF context: {context}
            User query: {query}

            Final Validated Query (or response):
            """
        )

        query = llm.invoke(
            prompt.format(
                query = st.text_input("Enter your query: "),
                context="This is a PDF chatbot. Answer only from the document."
            )
        ).content

        if(query):
            ans = retriever.invoke(query)
            st.session_state.chat_history.append(("You", query))
            st.session_state.chat_history.append(("AI", ans['answer']))
            st.write("AI: ",ans['result'])
        for role, msg in st.session_state.chat_history:
            st.write(f"**{role}:** {msg}")
else:
    print("Something went wrong,Please Upload Again!!")