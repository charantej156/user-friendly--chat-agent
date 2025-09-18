import os
import warnings
import sys
import json
import sqlite3
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -------------------------------
# 0. Log suppression (TensorFlow / OneDNN)
# -------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)


# -------------------------------
# 1. OpenAI Client
# -------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -------------------------------
# 2. Embedding + FAISS
# -------------------------------
class LocalEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)


class FaissVectorDB:
    def __init__(self, embedder: LocalEmbedder, dim=384):
        self.embedder = embedder
        self.index = faiss.IndexFlatL2(dim)
        self.docs = []

    def add(self, docs):
        vecs = self.embedder.embed(docs)
        self.index.add(vecs)
        self.docs.extend(docs)

    def search(self, query, k=3):
        qvec = self.embedder.embed([query])
        D, I = self.index.search(qvec, k)
        return [self.docs[i] for i in I[0] if i < len(self.docs)]


# -------------------------------
# 3. PDF Loader
# -------------------------------
def load_pdf(file_path):
    docs = []
    if os.path.exists(file_path) and file_path.endswith(".pdf"):
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    docs.append(text)
    return docs


embedder = LocalEmbedder()
vector_db = FaissVectorDB(embedder)

pdf_path = r""
if os.path.exists(pdf_path):
    pdf_docs = load_pdf(pdf_path)
    if pdf_docs:
        vector_db.add(pdf_docs)


# -------------------------------
# 4. Customer DB
# -------------------------------
def get_customer_data(name: str, db_path="customers.db") -> str:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute(
            "SELECT id, name, email, phone, status FROM customers WHERE name LIKE ?",
            (f"%{name}%",),
        )
        customer = cur.fetchone()
        if not customer:
            return "❌ No customer found."

        customer_id, name, email, phone, status = customer

        cur.execute(
            "SELECT order_id, product, order_date, status FROM orders WHERE customer_id=?",
            (customer_id,),
        )
        orders = cur.fetchall()

        cur.execute(
            """
            SELECT p.payment_id, p.amount, p.method, p.status, o.product
            FROM payments p
            JOIN orders o ON p.order_id = o.order_id
            WHERE o.customer_id=?
            """,
            (customer_id,),
        )
        payments = cur.fetchall()

        response = [f"📌 Customer: {name} ({email}, {phone}) - Status: {status}"]

        if orders:
            response.append("\n🛒 Orders:")
            for o in orders:
                response.append(f"  • Order #{o[0]}: {o[1]} on {o[2]} [{o[3]}]")

        if payments:
            response.append("\n💳 Payments:")
            for p in payments:
                response.append(
                    f"  • Payment #{p[0]}: ₹{p[1]} via {p[2]} for {p[4]} [{p[3]}]"
                )

        return "\n".join(response)
    except Exception as e:
        return f"⚠️ DB error: {str(e)}"
    finally:
        if "conn" in locals():
            conn.close()


# -------------------------------
# 5. Tools
# -------------------------------
def recommender_tool(query):
    return "✅ Recommended products: Laptop, Phone, Headphones"


def reminder_tool(query):
    return f"⏰ Reminder set: {query}"


def rag_tool(query):
    results = vector_db.search(query, k=2)
    return "\n".join(results) if results else "No relevant info found in docs."


def compliance_tool(query):
    return f"📩 Complaint registered: {query}. Our team will respond soon."


def customer_db_tool(query):
    return get_customer_data(query)


# -------------------------------
# 6. Router Agent
# -------------------------------
def pick_tool_with_agent(user_input: str):
    system_prompt = """
    You are a routing agent for a customer service chatbot.
    Rules:
    1. If small talk (hi, hello, how are you) → {"tool": "none"}.
    2. Otherwise, pick ONE tool:
       - recommender
       - reminder
       - complaint
       - customer_db
       - rag
    JSON ONLY, format: {"tool": "<tool>"}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception:
        return {"tool": "rag"}  # fallback


def tool_picker(user_input):
    decision = pick_tool_with_agent(user_input)
    tool = decision.get("tool", "rag")

    if tool == "none":
        return "👋 Hi there! How can I assist you today?"
    elif tool == "recommender":
        return recommender_tool(user_input)
    elif tool == "reminder":
        return reminder_tool(user_input)
    elif tool == "complaint":
        return compliance_tool(user_input)
    elif tool == "customer_db":
        return customer_db_tool(user_input)
    else:
        return rag_tool(user_input)


# -------------------------------
# 7. FastAPI Backend
# -------------------------------
app = FastAPI(title="SDK Customer Service Agent")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict later to ["http://127.0.0.1:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    user_input: str
    response: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        response = tool_picker(req.message)
        return {"user_input": req.message, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"⚠️ Server error: {str(e)}")
