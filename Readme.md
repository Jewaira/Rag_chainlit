# Chainlit FastAPI Gemini RAG

This project demonstrates a Retrieval-Augmented Generation (RAG) chatbot using FastAPI, Google Gemini, and FAISS vector search.

## Features

- Context-aware query rewriting using Gemini LLM
- Document retrieval with FAISS and custom embeddings
- Simple chat API endpoint

## Usage

1. Set your `GOOGLE_API_KEY` in a `.env` file.
2. Place your text files (`cats_fact.txt`, `scientists_experiments.txt`) in the project root.
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Run the API:
   ```sh
   uvicorn fastapiendpoint:app --reload
   ```

## API

- **POST** `/chat`
  - Request: `{ "query": "your question", "history": [...] }`
  - Response: `{ "response": "...", "history": [...] }`

## Customization

- Edit `chainlit.md` for the Chainlit welcome screen.
- Add or update documents in the root directory for retrieval.

---