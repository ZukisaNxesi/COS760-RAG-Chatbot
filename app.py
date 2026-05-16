# Note that this Streamlit Demo was AI generated for class demonstration purposes.
import os
import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from PyPDF2 import PdfReader

# ── Page config ────────────────────────────────────────────────────────
st.set_page_config(page_title="COS 760 Course Assistant", page_icon="📚", layout="wide")
st.title("📚 COS 760 Course Assistant")
st.caption("RAG-powered chatbot over the course materials")

# ── Sidebar: config ────────────────────────────────────────────────────
with st.sidebar:
    st.header("Setup")
    groq_key = st.text_input("Groq API key", type="password",
                              value=os.getenv("GROQ_API_KEY", ""))
    uploaded_files = st.file_uploader("Upload course PDFs",
                                       type="pdf", accept_multiple_files=True)
    chunk_size = st.slider("Chunk size (words)", 50, 500, 300)
    overlap    = st.slider("Overlap (words)",    0,  100,  50)
    top_k      = st.slider("Chunks to retrieve", 1, 8, 3)
    build_btn  = st.button("Build index", type="primary")

# ── Helpers ────────────────────────────────────────────────────────────
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(file) -> str:
    reader = PdfReader(file)
    return " ".join(p.extract_text() or "" for p in reader.pages)

def chunk_text(text, size, overlap):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+size]))
        i += size - overlap
    return chunks

def build_index(chunks, model):
    vecs = model.encode(chunks, show_progress_bar=False).astype("float32")
    idx  = faiss.IndexFlatL2(vecs.shape[1])
    idx.add(vecs)
    return idx, vecs

def retrieve(query, index, chunks, model, k):
    qv = model.encode([query]).astype("float32")
    _, ids = index.search(qv, k)
    return [chunks[i] for i in ids[0] if i < len(chunks)]

def ask(question, context_chunks, client):
    context = "\n---\n".join(f"[{i+1}] {c}" for i, c in enumerate(context_chunks))
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content":
             "Answer using ONLY the context below. "
             "If the answer isn't there, say so clearly."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        temperature=0.2
    )
    return resp.choices[0].message.content

# ── Index building ──────────────────────────────────────────────────────
embed_model = load_embed_model()

if build_btn:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF first.")
    elif not groq_key:
        st.sidebar.error("Enter your Groq API key.")
    else:
        with st.spinner("Reading PDFs and building index..."):
            all_text = "\n\n".join(extract_text(f) for f in uploaded_files)
            chunks   = chunk_text(all_text, chunk_size, overlap)
            index, _ = build_index(chunks, embed_model)
            st.session_state["chunks"] = chunks
            st.session_state["index"]  = index
            st.session_state["client"] = OpenAI(
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1"
            )
        st.sidebar.success(f"Index built: {len(chunks)} chunks from "
                           f"{len(uploaded_files)} file(s)")

# ── Chat UI ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Retrieved sources"):
                for i, s in enumerate(msg["sources"]):
                    st.caption(f"Chunk {i+1}: {s[:200]}...")

if prompt := st.chat_input("Ask a question about your course..."):
    if "index" not in st.session_state:
        st.warning("Build the index first using the sidebar.")
    else:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                sources = retrieve(prompt, st.session_state["index"],
                                   st.session_state["chunks"], embed_model, top_k)
                answer  = ask(prompt, sources, st.session_state["client"])
            st.markdown(answer)
            with st.expander("Retrieved sources"):
                for i, s in enumerate(sources):
                    st.caption(f"Chunk {i+1}: {s[:200]}...")

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

