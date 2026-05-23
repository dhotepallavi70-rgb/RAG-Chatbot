import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PDF_FOLDER = "data/pdfs"
VECTOR_FOLDER = "vectorstore"

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #eef4ff 50%, #ffffff 100%);
}

html, body, [class*="css"] {
    color: #111827 !important;
}

h1, h2, h3, h4, h5, h6, p, div, span, label {
    color: #111827 !important;
}

.hero {
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(135deg, #0f172a 0%, #1e40af 50%, #2563eb 100%);
    box-shadow: 0 15px 35px rgba(37,99,235,0.25);
    margin-bottom: 25px;
}

.hero h1 {
    color: white !important;
    font-size: 44px;
}

.hero p {
    color: #e0e7ff !important;
    font-size: 18px;
}

.card {
    background: #ffffff;
    padding: 22px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(15,23,42,0.08);
    border: 1px solid #d1d5db;
    margin-bottom: 18px;
}

.metric-card {
    background: white;
    padding: 18px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(15,23,42,0.07);
    border: 1px solid #d1d5db;
}

.metric-card p {
    color: #374151 !important;
}

.answer-box {
    background: #ecfdf5;
    border-left: 6px solid #10b981;
    padding: 20px;
    border-radius: 18px;
    color: #111827 !important;
    font-size: 18px;
    font-weight: 600;
    line-height: 1.8;
}

.source-box {
    background: #ffffff;
    padding: 16px;
    border-radius: 14px;
    border-left: 5px solid #2563eb;
    margin-bottom: 10px;
    color: #111827 !important;
    font-size: 15px;
    font-weight: 500;
}

.warning-box {
    background: #fff7ed;
    border-left: 6px solid #f97316;
    padding: 16px;
    border-radius: 16px;
    color: #111827 !important;
}

section[data-testid="stSidebar"] {
    background: #ffffff;
}

.stTextInput input {
    background: white;
    color: #111827 !important;
    border-radius: 12px;
}

[data-testid="stMetricValue"] {
    color: #111827 !important;
}

[data-testid="stMetricLabel"] {
    color: #374151 !important;
}

.streamlit-expanderHeader {
    color: #111827 !important;
    font-weight: 600;
}

.streamlit-expanderContent {
    color: #111827 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1e40af);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 10px 20px;
    font-weight: 600;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8);
}

.footer {
    text-align: center;
    color: #64748b !important;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=450
    )


def save_uploaded_files(uploaded_files):
    for uploaded_file in uploaded_files:
        file_path = os.path.join(PDF_FOLDER, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())


def load_pdf_documents():
    documents = []

    for file_name in os.listdir(PDF_FOLDER):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(PDF_FOLDER, file_name)
            loader = PyPDFLoader(file_path)
            pages = loader.load()

            for page in pages:
                page.metadata["source"] = file_name
                page.metadata["page"] = page.metadata.get("page", 0) + 1

            documents.extend(pages)

    return documents


def create_vector_database():
    documents = load_pdf_documents()

    if not documents:
        return 0, 0, 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=850,
        chunk_overlap=120
    )

    chunks = splitter.split_documents(documents)

    db = FAISS.from_documents(chunks, get_embeddings())
    db.save_local(VECTOR_FOLDER)

    total_pdfs = len([f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")])
    total_pages = len(documents)
    total_chunks = len(chunks)

    return total_pdfs, total_pages, total_chunks


def load_vector_database():
    return FAISS.load_local(
        VECTOR_FOLDER,
        get_embeddings(),
        allow_dangerous_deserialization=True
    )


def build_context(docs):
    context = ""

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page", "Unknown Page")
        context += f"\nSource {i}: {source}, Page {page}\n{doc.page_content}\n"

    return context


def generate_answer(question, docs):
    context = build_context(docs)

    prompt = ChatPromptTemplate.from_template("""
You are DocuMind AI, a professional RAG assistant.

Use ONLY the provided PDF context to answer.

Rules:
1. Give a direct, clear, professional answer.
2. Do not use outside knowledge.
3. If the answer is not available, say:
"I could not find this information in the uploaded PDFs."
4. Keep answer short but complete.

PDF Context:
{context}

Question:
{question}

Answer:
""")

    chain = prompt | get_llm()
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content


def confidence_label(score):
    if score < 0.8:
        return "High"
    elif score < 1.2:
        return "Medium"
    else:
        return "Low"


st.markdown("""
<div class="hero">
    <h1>📄 DocuMind AI</h1>
    <p>Production-style Multi-PDF RAG Chatbot powered by Groq, FAISS and Open-source Embeddings.</p>
    <p><b>Upload PDFs → Ask Questions → Get Answers with Source Page Citations</b></p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>⚡ Fast Retrieval</h3>
        <p>Target latency: 2–4 sec</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>📚 Multi-PDF</h3>
        <p>Search across many documents</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>🎯 Cited Answers</h3>
        <p>PDF name + page number</p>
    </div>
    """, unsafe_allow_html=True)


with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/337/337946.png",
        width=90
    )

    st.title("Control Panel")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        save_uploaded_files(uploaded_files)
        st.success(f"{len(uploaded_files)} PDF uploaded successfully.")

    if st.button("🚀 Process Documents", use_container_width=True):
        with st.spinner("Creating knowledge base..."):
            total_pdfs, total_pages, total_chunks = create_vector_database()

        if total_chunks > 0:
            st.success("Knowledge base ready!")
            st.metric("PDFs", total_pdfs)
            st.metric("Pages", total_pages)
            st.metric("Chunks", total_chunks)
        else:
            st.error("Please upload PDFs first.")

    st.divider()
    st.subheader("System Stack")
    st.write("🧠 LLM: Groq Llama")
    st.write("🔎 Vector DB: FAISS")
    st.write("📌 Embeddings: HuggingFace")
    st.write("🎨 UI: Streamlit")


st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("💬 Ask a Question from Your PDFs")

question = st.text_input(
    "Type your question here",
    placeholder="Example: How many casual leaves are allowed?"
)

st.markdown('</div>', unsafe_allow_html=True)


if question:
    start_time = time.time()

    try:
        db = load_vector_database()

        retrieved_docs_with_scores = db.similarity_search_with_score(
            question,
            k=3
        )

        retrieved_docs = [doc for doc, score in retrieved_docs_with_scores]

        answer = generate_answer(question, retrieved_docs)
        response_time = round(time.time() - start_time, 2)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("✅ Final Answer")
        st.markdown(
            f'<div class="answer-box">{answer}</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        s1, s2 = st.columns(2)

        with s1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("⏱️ Latency")
            if response_time <= 4:
                st.success(f"{response_time} seconds")
            else:
                st.warning(f"{response_time} seconds")
            st.markdown('</div>', unsafe_allow_html=True)

        with s2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🎯 Retrieval Quality")
            best_score = retrieved_docs_with_scores[0][1]
            st.success(confidence_label(best_score))
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📌 Source Citations")

        shown_sources = set()

        for doc, score in retrieved_docs_with_scores:
            source = doc.metadata.get("source", "Unknown PDF")
            page = doc.metadata.get("page", "Unknown Page")
            key = f"{source}-{page}"

            if key not in shown_sources:
                st.markdown(
                    f"""
                    <div class="source-box">
                        <b>📄 {source}</b><br>
                        Page: <b>{page}</b><br>
                        Confidence: <b>{confidence_label(score)}</b>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                shown_sources.add(key)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔍 Retrieved Evidence")

        for i, (doc, score) in enumerate(retrieved_docs_with_scores, start=1):
            source = doc.metadata.get("source", "Unknown PDF")
            page = doc.metadata.get("page", "Unknown Page")

            with st.expander(f"Evidence {i}: {source} — Page {page}"):

                st.markdown(
                    f"""
                    <div style="
                        color:#111827;
                        font-size:16px;
                        line-height:1.8;
                        font-weight:500;
                        background:white;
                        padding:15px;
                        border-radius:12px;
                    ">
                    {doc.page_content[:1000]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(
                    f"Similarity Score: {round(float(score),4)}"
                )

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.markdown(
            """
            <div class="warning-box">
                Please upload PDFs and click <b>Process Documents</b> first.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write(e)


st.markdown("""
<div class="footer">
    <p>DocuMind AI | Multi-PDF RAG Chatbot | Groq + FAISS + HuggingFace</p>
</div>
""", unsafe_allow_html=True)