import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np
from groq import Groq

# PAGE SETTINGS
st.set_page_config(page_title="📄🤖 PDF Chatbot (RAG + LLM)", layout="wide")
st.title("📄🤖 PDF Chatbot (RAG + LLM)")

# SIDEBAR — Upload PDF
st.sidebar.header("📤 Upload PDF")
pdf_file = st.sidebar.file_uploader("Upload a PDF file:", type=["pdf"])


# LOAD GROQ KEY
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# SESSION STATE for Chat
if "messages" not in st.session_state:
    st.session_state["messages"] = []     # stores chat history

# Chat container
chat_box = st.container()

# PROCESS PDF
if pdf_file:
    # Extract text
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = splitter.split_text(text)

    # Embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    # Setup FAISS
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    id_to_chunk = {i: chunks[i] for i in range(len(chunks))}

    # USER QUESTION
    user_query = st.chat_input("Ask something from the PDF...")

    if user_query:

        # Add to history
        st.session_state["messages"].append({"role": "user", "content": user_query})

        # Retrieve top chunks
        q_vec = model.encode([user_query]).astype("float32")
        distances, indices = index.search(q_vec, 5)

        context = ""
        for idx in indices[0]:
            context += id_to_chunk[idx] + "\n"

        # LLM Prompt
        prompt = f"""
Use ONLY this PDF context to answer:

{context}

Question: {user_query}

Reply in simple English. If answer is not in context, say:
"The PDF does not contain this information."
"""

        # Call Groq LLM
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300
        )

        answer = response.choices[0].message.content

        # Add assistant reply
        st.session_state["messages"].append({"role": "assistant", "content": answer})
    # DISPLAY CHAT HISTORY
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            with chat_box.chat_message("user"):
                st.write(msg["content"])
        else:
            with chat_box.chat_message("assistant"):
                st.write(msg["content"])
else:
    st.info("👈 Please upload a PDF from the sidebar to start chatting.")
