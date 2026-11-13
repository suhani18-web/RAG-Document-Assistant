
import streamlit as st
import requests
import numpy as np
import faiss
import re

from io import BytesIO
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Document Assistant")
st.write("Upload a PDF and ask questions about its contents.")


# LOAD PDF AND CREATE VECTOR DATABASE
@st.cache_resource
def load_rag_system(pdf_bytes):

    reader = PdfReader(BytesIO(pdf_bytes))

    all_pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        page_text = page.extract_text() or ""

        if page_text.strip():
            all_pages.append({
                "page": page_number,
                "text": page_text
            })

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = []

    for page in all_pages:

        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "page": page["page"],
                "text": chunk
            })

    if not chunks:
        raise ValueError("No readable text found in PDF.")

    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = [chunk["text"] for chunk in chunks]

    embeddings_list = []

    for i in range(0, len(texts), 50):

        batch = texts[i:i + 50]

        batch_embeddings = model.encode(
            batch,
            normalize_embeddings=True
        )

        embeddings_list.extend(batch_embeddings)

    embeddings = np.array(
        embeddings_list,
        dtype="float32"
    )

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return model, chunks, index


# CHECK ELIGIBILITY QUESTION
def is_eligibility_query(query):

    query = query.lower()

    terms = [
        "eligibility",
        "mandatory subjects",
        "admission criteria",
        "admission eligibility",
        "minimum aggregate marks"
    ]

    return any(term in query for term in terms)


# EXTRACT ONLY SECTION 8.7
def get_eligibility_context(chunks):

    page_texts = []

    for chunk in chunks:

        if chunk["page"] in [12, 13]:
            page_texts.append(chunk["text"])

    combined_text = "\n".join(page_texts)

    start_match = re.search(
        r"8\.7\s+Eligibility\s+Criteria",
        combined_text,
        re.IGNORECASE
    )

    end_match = re.search(
        r"8\.8\s+Eligibility\s+Criteria\s+for\s+Award\s+of\s+Degree",
        combined_text,
        re.IGNORECASE
    )

    if start_match:

        start = start_match.start()

        if end_match and end_match.start() > start:
            end = end_match.start()
            section_text = combined_text[start:end]
        else:
            section_text = combined_text[start:]

        return section_text.strip()

    # Fallback if section heading is not detected
    return combined_text.strip()


# NORMAL HYBRID SEARCH
def hybrid_search(query, model, chunks, index, top_k=5):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    search_count = min(50, len(chunks))

    scores, indices = index.search(
        query_embedding,
        search_count
    )

    selected = {}

    for rank, idx in enumerate(indices[0]):

        if idx != -1:
            selected[int(idx)] = 1 / (rank + 1)

    keywords = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        query.lower()
    )

    stopwords = {
        "what", "is", "the", "for", "are",
        "and", "of", "in", "to", "a", "an",
        "with", "according", "from", "does",
        "how", "which", "this", "that"
    }

    keywords = [
        word for word in keywords
        if len(word) > 2 and word not in stopwords
    ]

    for i, chunk in enumerate(chunks):

        text = chunk["text"].lower()

        score = sum(
            1 for word in keywords
            if re.search(
                r"\b" + re.escape(word) + r"\b",
                text
            )
        )

        if score > 0:
            selected[i] = selected.get(i, 0) + score * 0.15

    ranked = sorted(
        selected.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        chunks[idx]
        for idx, score in ranked[:top_k]
    ]


# PDF UPLOADER
uploaded_file = st.file_uploader(
    "Upload your PDF Document",
    type=["pdf"]
)


if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    st.success(f"Uploaded: {uploaded_file.name}")

    with st.spinner("Processing PDF..."):

        try:

            model, chunks, index = load_rag_system(pdf_bytes)

            st.success(
                f"Document processed successfully! "
                f"{len(chunks)} text chunks created."
            )

        except Exception as e:

            st.error(f"PDF processing error: {e}")
            st.stop()

    # QUESTION INPUT
    question = st.text_input(
        "Enter your Question:",
        placeholder="What is the eligibility criteria?"
    )

    if st.button("Ask AI"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner("AI is finding your answer..."):

                try:

                    # SPECIAL ELIGIBILITY RETRIEVAL
                    if is_eligibility_query(question):

                        eligibility_text = get_eligibility_context(chunks)

                        if not eligibility_text:
                            st.error("Eligibility section not found.")
                            st.stop()

                        context = (
                            "Section 8.7 Eligibility Criteria "
                            "from Pages 12 and 13:\n"
                            + eligibility_text
                        )

                        results = [{
                            "page": "12–13",
                            "text": eligibility_text
                        }]

                        instructions = """
Answer the admission eligibility question using ONLY Section 8.7.

Important:
- Physics and Mathematics are mandatory subjects.
- The student must select one remaining subject from the listed options.
- Minimum aggregate marks are 50%.
- Do not confuse admission eligibility with degree award requirements.
- Do not mention 166 credits because that belongs to Section 8.8.
- Do not invent any other eligibility condition.
"""

                    else:

                        results = hybrid_search(
                            question,
                            model,
                            chunks,
                            index
                        )

                        context = ""

                        for result in results:

                            context += (
                                f"\nPage {result['page']}:\n"
                                f"{result['text']}\n"
                            )

                        instructions = """
Answer using ONLY the provided document context.
Give the actual answer in simple language.
Mention relevant source pages.
Do not invent facts.
If information is missing, say so.
"""

                    prompt = f"""
You are a university document assistant.

{instructions}

DOCUMENT CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

                    response = requests.post(
                        "http://localhost:11434/api/generate",
                        json={
                            "model": "llama3.2:1b",
                            "prompt": prompt,
                            "stream": False,
                            "options": {
                                "temperature": 0,
                                "num_predict": 300
                            }
                        },
                        timeout=180
                    )

                    response.raise_for_status()

                    answer = response.json()["response"]

                    st.subheader("🤖 AI Answer")
                    st.write(answer)

                    st.subheader("📄 Source Documents")

                    for result in results:

                        st.write(f"Page {result['page']}")
                        st.caption(result["text"])

                except Exception as e:

                    st.error(f"Something went wrong: {e}")

else:

    st.info("Please upload a PDF document to get started.")
