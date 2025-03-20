from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from groq import Groq
from tavily import TavilyClient
import PyPDF2
import io
import os

app = FastAPI()

# Load environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pdf-rag")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Check if required environment variables are set
if not all([PINECONE_API_KEY, GROQ_API_KEY, TAVILY_API_KEY]):
    raise ValueError("Missing one or more required environment variables")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize sentence transformer for embeddings
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Groq and Tavily clients
groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)


class Query(BaseModel):
    text: str
    id: str | None = None


def extract_text_from_pdf(file: UploadFile) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.file.read()))
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error extracting text: {str(e)}")


@app.post("/upload_pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    id: str = Form(default="pdf_doc")  # Allows user to pass ID optionally
):
    try:
        # Validate file type
        if not file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="File must be a PDF")

        # Extract text from PDF
        text = extract_text_from_pdf(file)
        print(f"Extracted text: {text[:100]}...")  # Debug info

        # Generate embedding
        embedding = encoder.encode(text).tolist()
        print(f"Embedding length: {len(embedding)}")  # Debug, should match expected size

        # Upsert into Pinecone with namespace (per ID)
        index.upsert(vectors=[(id, embedding, {"text": text})], namespace=id)
        print(f"Upserted ID: {id} to Pinecone under namespace: {id}")  # Confirm upsert

        return {"message": "PDF uploaded successfully", "id": id}

    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging error
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask_question(query: Query):
    try:
        if not query.id:
            raise HTTPException(status_code=400, detail="ID is required to query the document")

        # Generate query embedding
        query_embedding = encoder.encode(query.text).tolist()

        # Query Pinecone using the correct namespace (per ID isolation)
        pinecone_results = index.query(
            vector=query_embedding,
            top_k=1,  # Only fetch the most relevant document
            include_metadata=True,
            namespace=query.id  # Query within the correct namespace
        )

        # Check if we found relevant content
        if not pinecone_results['matches']:
            pdf_context = "No relevant document found for the provided ID."
        else:
            pdf_context = pinecone_results['matches'][0]['metadata'].get('text', '')

        # Enhance with Tavily if needed
        tavily_response = tavily_client.search(
            query=f"financial fintech {query.text}",
            search_depth="advanced",
            max_results=3,
            include_domains=[
                "*.org", "*.edu", "*.gov",
                "finance.yahoo.com", "bloomberg.com", "reuters.com"
            ]
        )
        tavily_context = "\n".join([result['content'] for result in tavily_response.get('results', [])])

        # Combine contexts
        combined_context = f"PDF Context (Document {query.id}):\n{pdf_context}\n\nReal-Time Financial Data (Tavily):\n{tavily_context}"

        # Generate answer with Groq
        groq_response = groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial and fintech expert. Answer questions based only on the provided context from financial documents and real-time data, avoiding general knowledge outside this domain."
                },
                {
                    "role": "user",
                    "content": f"Context: {combined_context}\nQuestion: {query.text}"
                }
            ],
            max_tokens=150,
            temperature=0.7
        )
        answer = groq_response.choices[0].message.content.strip()

        return {
            "question": query.text,
            "answer": answer,
            "pdf_context": pdf_context[:500] + "..." if len(pdf_context) > 500 else pdf_context,
            "tavily_context": tavily_context[:500] + "..." if len(tavily_context) > 500 else tavily_context
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
