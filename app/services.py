import os
import json
from typing import List
from datetime import datetime
from sentence_transformers import SentenceTransformer
from qdrant_client.models import PointStruct, VectorParams, Distance
from groq import Groq

from app.database import qdrant_client, redis_client, bookings_collection
from app.utils import fixed_chunking, recursive_chunking

# ---------- EMBEDDINGS ----------
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    cache_folder=os.getenv("HF_HOME", "C:/hf_cache")
)
EMBED_DIM = 384

# ---------- LLM ----------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_embeddings(texts: List[str]) -> List[List[float]]:
    return embedding_model.encode(texts, convert_to_numpy=True).tolist()

# ---------- INGEST SERVICE ----------
async def ingest_document_service(text: str, filename: str, strategy: str) -> int:
    chunks = fixed_chunking(text) if strategy == "fixed" else recursive_chunking(text)
    chunks = [c for c in chunks if c.strip()]
    
    embeddings = get_embeddings(chunks)

    # recreate_collection wipes existing data. Using it only for fresh uploads.
    qdrant_client.recreate_collection(
        collection_name="docs",
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
    )

    points = [
        PointStruct(id=i, vector=embeddings[i], payload={"text": chunks[i]})
        for i in range(len(chunks))
    ]
    qdrant_client.upsert("docs", points)
    return len(chunks)

# ---------- CHAT & BOOKING SERVICE ----------

async def rag_chat_service(session_id: str, query: str):
    history = json.loads(redis_client.get(session_id) or "[]")

    # Retrieval logic 
    q_vec = get_embeddings([query])[0]
    hits = qdrant_client.query_points("docs", query=q_vec, limit=3).points
    context = "\n".join(h.payload["text"] for h in hits)

    # SYSTEM PROMPT
    system_prompt = (
        "You are a professional RAG assistant.\n"
        "1. Answer questions using the provided Context.\n"
        "2. To book an interview, you MUST have exactly: Name, Email, Date (YYYY-MM-DD), and Time (HH:MM).\n"
        "3. If ANY of these 4 fields are missing or unclear, tell the user: "
        "'Booking cannot be placed at the moment. Please provide the [missing field] and try again.'\n"
        "4. If the user provides all info, call the 'book_interview' tool immediately.\n"
        f"Today's Date: {datetime.now().strftime('%Y-%m-%d')}\n\n"
        f"Context:\n{context}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": query})

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        tools=[{
            "type": "function",
            "function": {
                "name": "book_interview",
                "description": "Saves booking to DB",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"},
                        "date": {"type": "string"},
                        "time": {"type": "string"}
                    },
                    "required": ["name", "email", "date", "time"]
                }
            }
        }],
        tool_choice="auto"
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        # ALL INFO PRESENT -> now BOOKING
        try:
            data = json.loads(msg.tool_calls[0].function.arguments)
            bookings_collection.insert_one(data)
            answer = f"✅ Success! Your interview is booked for {data['date']} at {data['time']}."
        except Exception as e:
            
            # error message for DB issues
            answer = "⚠️ Booking cannot be placed at the moment due to a database error. Please try again later."
            print(f"DB Error: {e}")
    else:
        # If there is  MISSING INFO OR MISTAKE in query
        if "book" in query.lower() or any(field in msg.content.lower() for field in ["email", "date", "time"]):
            answer = f" {msg.content} due to some missing information. Please fill that and try again."
        else:
            answer = msg.content

    # Persistence
    history.append({"role": "user", "content": query})
    history.append({"role": "assistant", "content": answer})
    redis_client.setex(session_id, 3600, json.dumps(history))

    return answer