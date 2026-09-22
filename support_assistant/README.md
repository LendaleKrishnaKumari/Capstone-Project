# Zepto Support Assistant

This module implements a small retrieval-augmented support assistant for Zepto. It uses a local document corpus, ChromaDB for vector retrieval, and a LangGraph router for intent classification and answer selection. The default path runs in deterministic mock mode, which is the graded baseline.

## Architecture

The full retrieval pipeline is:

- Ingestion: the documents in the docs folder are loaded from text files and stored as chunks.
- Embedding: the local sentence-transformers model `all-MiniLM-L6-v2` embeds each chunk before it is saved into the ChromaDB collection named `zepto_policy`.
- Retrieval: the `retrieve_and_answer` LangGraph node queries ChromaDB for the top 3 similar chunks using cosine similarity.
- Generation: the final answer is produced in the same `retrieve_and_answer` or `direct_answer` node, depending on the route chosen by the graph. In mock mode, generation is rule-based and deterministic; in the optional `MOCK_LLM=0` path, a real LLM can be used instead.

The `MOCK_LLM` toggle affects the generation step, not the routing decision. The `classify_intent` node uses a keyword heuristic in default mock mode and only changes to a real LLM classifier when `MOCK_LLM=0` is explicitly set.

## Example JSON responses (default mock mode)

### Example 1: policy question
Request:

```json
{"query": "What is the delivery fee for Zepto?"}
```

Response:

```json
{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard delivery is free on orders over INR 149; orders below this threshold incur a flat INR 25 delivery fee.","sources":["doc_01"],"confidence":1.0}
```

### Example 2: general question
Request:

```json
{"query": "Tell me a joke"}
```

Response:

```json
{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}
```

## Local run

```bash
cd "D:\SEMESTERS\IIT course\support_assistant"
py -m uvicorn main:app --host 0.0.0.0 --port 7860
```

Then call the API:

```bash
curl -X POST http://127.0.0.1:7860/ask -H "Content-Type: application/json" -d "{\"query\":\"What is the delivery fee for Zepto?\"}"
```

## Docker usage

Build and run locally from this folder:

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

The container serves the same `/ask` FastAPI endpoint locally.

## Document corpus

The 8 Zepto policy documents are stored in the `docs` folder and used as the retrieval corpus for the support assistant.
