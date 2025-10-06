import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

# -------------------------------------------------
# Sidebar: Secure Groq API Key Input
# -------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("🔑 Enter Groq API Key", type="password", key="api_key")

    st.markdown("---")
    st.write("💡 **Instructions:**")
    st.write("1. Enter your Groq API key above.")
    st.write("2. Click **Document Embedding** to load research papers.")
    st.write("3. Ask any question about the documents below!")

# -------------------------------------------------
# Main Page
# -------------------------------------------------
st.title("📄 RAG Document Q&A with Groq LLM + HuggingFace Embeddings")

# Ensure API key is entered
if not groq_api_key:
    st.warning("⚠️ Please enter your Groq API Key in the sidebar to continue.")
    st.stop()

# Initialize Groq LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

# -------------------------------------------------
# Prompt Template
# -------------------------------------------------
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the context only.
    Please provide the most accurate answer based on the question.

    <context>
    {context}
    <context>
    Question: {input}
    """
)

# -------------------------------------------------
# Function to Create Vector Embeddings
# -------------------------------------------------
def create_vector_embeddings():
    if "vectors" not in st.session_state:
        with st.spinner("🔄 Creating vector embeddings..."):
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.loader = PyPDFDirectoryLoader("research_papers")  # fixed path
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs
            )
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, st.session_state.embeddings
            )
        st.success("✅ Vector Database is Ready!")

# -------------------------------------------------
# Main Q&A Section
# -------------------------------------------------
user_prompt = st.text_input("💬 Ask a question about the documents")

if st.button("📚 Document Embedding"):
    create_vector_embeddings()

if user_prompt:
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please click '📚 Document Embedding' first to load documents.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        st.write(f"⏱ Response time: {time.process_time() - start:.2f} seconds")

        # Debug - show raw response
        with st.expander("🔍 Raw Response"):
            st.write(response)

        # Display Answer
        if "answer" in response:
            st.subheader("🧠 Answer:")
            st.write(response["answer"])
        elif "output_text" in response:
            st.subheader("🧠 Answer:")
            st.write(response["output_text"])

        # Show retrieved context
        with st.expander("📑 Document Similarity Search"):
            for i, doc in enumerate(response.get("context", [])):
                st.write(doc.page_content)
                st.write("----------")
