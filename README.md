# COS760-RAG-Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Python and Streamlit that allows users to upload PDF documents and ask questions based on the uploaded content.

## Features

- Upload and analyze PDF documents
- Semantic search using vector embeddings
- Context-aware question answering
- Interactive Streamlit interface
- FAISS vector database integration
- SentenceTransformer embeddings
- Groq LLM API integration

---

## Technologies Used

- Python
- Streamlit
- FAISS
- SentenceTransformers
- PyPDF2
- NumPy
- OpenAI SDK
- Groq API

---

## Project Structure

```text
COS760-RAG-Chatbot/
│
├── app.py
├── requirements.txt
├── README.md

```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/COS760-RAG-Chatbot.git
cd COS760-RAG-Chatbot
```

### 2. Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Getting a Groq API Key

This project uses the Groq API for LLM responses.

1. Visit https://console.groq.com
2. Create a free account
3. Generate an API key
4. Copy the key

---

## Running the Application

Start the Streamlit app:

```bash
streamlit run app.py
```

After running the command, open the local URL shown in the terminal:

```text
http://localhost:8501
```

---

## How to Use

1. Enter your Groq API key in the sidebar
2. Upload one or more PDF documents
3. Click **Build Index**
4. Ask questions about the uploaded documents

---

## Future Improvements

- Chat history memory
- Better retrieval strategies
- Support for DOCX and TXT files
- Deployment to Streamlit Cloud
- Source highlighting and citations

---
Heres provided screenshot
<img width="1920" height="1017" alt="Screenshot (162)" src="https://github.com/user-attachments/assets/983e20c7-2feb-4a94-9f73-749a44e37733" />

