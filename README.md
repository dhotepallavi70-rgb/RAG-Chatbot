# DocuMind AI — Multi-PDF RAG Chatbot

DocuMind AI is a Retrieval-Augmented Generation chatbot that answers user questions from multiple uploaded PDFs with source PDF name and page number citations.

## Objective

To build a RAG chatbot that can read multiple PDFs, store their content in a FAISS vector database, retrieve relevant pages, and generate accurate answers with citations.

## Features

- Upload multiple PDFs
- Extract page-wise PDF text
- Chunk text with overlap
- Generate embeddings using open-source HuggingFace model
- Store embeddings in FAISS vector database
- Ask questions from uploaded PDFs
- Generate answers using Groq LLM
- Show source PDF name and page number
- Show retrieved evidence
- Show response latency
- No hallucination guard

## Tech Stack

- Python
- Streamlit
- LangChain
- FAISS
- HuggingFace Embeddings
- Groq LLM
- PyPDFLoader

## Architecture

PDF Upload  
→ Text Extraction  
→ Chunking  
→ Embedding Generation  
→ FAISS Vector Database  
→ User Question  
→ Semantic Search  
→ Groq LLM  
→ Answer with Source Citations

## How to Run

1. Clone the repository

```bash
git clone YOUR_GITHUB_REPO_LINK
cd DocuMind-AI