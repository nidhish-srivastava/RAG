# 📚 PDF-RAG: Financial and Fintech Query System

This project is a PDF-based RAG (Retrieval-Augmented Generation) API built using FastAPI that allows users to:

📄 Upload PDF documents and store their embeddings in Pinecone.

❓ Ask questions based on the content of uploaded PDFs.

🌐 Enhance answers with real-time financial data using Tavily.

🧠 Generate answers using Groq's LLM (LLaMA 3-70B).

### 🚀 Features

✅ Upload a PDF and extract text

✅ Generate embeddings with SentenceTransformer

✅ Store and query embeddings in Pinecone

✅ Query financial data from Tavily API

✅ Generate context-aware answers with Groq LLM

✅ Health check endpoint for monitoring


### ⚡️ Tech Stack

Backend: FastAPI

Vector DB: Pinecone

LLM Provider: Groq (LLaMA 3-70B)

Real-time Search: Tavily API

Embeddings: SentenceTransformer
