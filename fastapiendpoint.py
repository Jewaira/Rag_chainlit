from fastapi import FastAPI
from pydantic import BaseModel
import google.generativeai as genai
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY =os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)

loader1 = TextLoader("cats_fact.txt", encoding="utf-8")
loader2 = TextLoader("scientists_experiments.txt", encoding="utf-8")
documents = loader1.load() + loader2.load()

class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text
            )
            embeddings.append(result["embedding"])
        return embeddings
    
    def embed_query(self, text):
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text
        )
        return result["embedding"]

embedding_model = GeminiEmbeddings()
vectorstore = FAISS.from_documents(documents, embedding_model)

app = FastAPI()

class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []

def rewrite_query_with_context_check(query: str, history: list[dict]):
    """
    Single LLM call that checks if query needs context and rewrites it clearly
    
    Args:
        query (str): Original user query
        history (list[dict]): Conversation history
    
    Returns:
        str: Clear, context-aware query suitable for retrieval
    """
    formatted_history = ""
    for entry in history:
        role = entry.get("role", "")
        parts = entry.get("parts", [])
        if parts:
            content = parts[0] if isinstance(parts[0], str) else str(parts[0])
            formatted_history += f"{role}: {content}\n"
    
    prompt = (
        f"You are a query analyzer and rewriter. Given this conversation history:\n"
        f"{formatted_history}\n\n"
        f"User query: {query}\n\n"
        f"Task: Analyze if this query references previous conversation context (uses pronouns like 'it', 'its', 'they', 'them' or unclear references). "
        f"If yes, rewrite the query to be completely clear and specific by replacing pronouns and unclear references with explicit terms from the conversation history. "
        f"If no, return the original query unchanged.\n\n"
       
    )
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    rewritten_query = response.text.strip()
    
    print(f"Original query: {query}")
    print(f"Rewritten query: {rewritten_query}")
    
    return rewritten_query

def llm(query: str, history: list[dict]):
    search_query = rewrite_query_with_context_check(query, history)
    docs = vectorstore.similarity_search(search_query, k=1)
    context = "\n\n".join(d.page_content for d in docs)
    print(f"Retrieved context: {context[:200]}...")
    
    conversation = history.copy()
    conversation.append({
        "role": "user",
        "parts": [f"Query: {query}\n\nContext:\n{context}"]
    })
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(conversation)
    return response.text

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    result = llm(request.query, request.history)
    
    updated_history = request.history + [
        {"role": "user", "parts": [request.query]},
        {"role": "assistant", "parts": [result]}
    ]
    
    return {"response": result, "history": updated_history}

   
# @app.get("/")
# def read_root():
#     return {"message": "Enhanced Chatbot API is running"}

# Test function
# def test_context_rewrite():
#     """Test the context-aware query rewriting"""
#     test_history = [
#         {"role": "user", "parts": ["Tell me about cats"]},
#         {"role": "assistant", "parts": ["Cats are fascinating animals known for grooming habits and hairball formation."]},
#         {"role": "user", "parts": ["What happens when they groom?"]}
#     ]
    
#     test_queries = [
#         "the technical term for its hairball is?",
#         "how do they clean themselves?",
#         "what is the scientific name of dogs?"  # This shouldn't change
#     ]
    
#     for query in test_queries:
#         rewritten = rewrite_query_with_context_check(query, test_history)
#         print(f"\nTest Query: {query}")
#         print(f"Rewritten: {rewritten}")

if __name__ == "__main__":
    import uvicorn
    
    print("Testing context-aware query rewriting...")
    # test_context_rewrite()
    

