
# RAG-Based Document Assistant

## Project Overview
An NLP-based application that answers questions from a university programme handbook using Retrieval-Augmented Generation (RAG).

## Technologies Used
- Python
- Streamlit
- Sentence Transformers
- FAISS
- Ollama (Llama 3.2)
- PyPDF
- NLP

## Project Workflow
1. Upload the university handbook PDF.
2. Extract text from PDF pages.
3. Clean and split text into chunks.
4. Generate text embeddings.
5. Store embeddings in FAISS.
6. Retrieve relevant chunks using semantic search.
7. Generate answers using the Ollama language model.
8. Display answers with source page references.

## How to Run
1. Install the required libraries:
   pip install -r requirements.txt

2. Start Ollama and ensure llama3.2:1b is available.

3. Run the application:
   streamlit run app.py

## Dataset
K.R. Mangalam University B.Tech CSE (AI & ML) Programme Handbook.

## Output
The application provides answers to user questions along with relevant source text and page numbers.
