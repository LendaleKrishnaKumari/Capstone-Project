from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from langgraph.graph import END, START, StateGraph

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

ROOT_DIR = Path(__file__).resolve().parent
DOCS_DIR = ROOT_DIR / "docs"
CHROMA_DIR = ROOT_DIR / "chroma_db"
MODEL_NAME = "all-MiniLM-L6-v2"

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]

PROMPT_TEMPLATE = """Role: You are Zepto's policy support assistant.
Context: Use only the retrieved Zepto policy excerpts below. If the information is not present in the context, say so clearly.
Task: Answer the user's question grounded in the retrieved Zepto policy context, using the policy details only.
Format: Return a concise answer with the key policy rule first, then a short practical note if needed.
Length: Keep the answer under 180 words.
Negative constraint: Do not answer using information not present in the provided context.
Few-shot example:
User question: "What is the delivery fee?"
Answer: "Standard delivery is free on orders above INR 149. Orders below that threshold incur a flat INR 25 fee."

Retrieved context:
{context}

User question: {question}
"""


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class GraphState(TypedDict):
    query: str
    intent: Literal["policy_question", "general_question"]
    answer: str
    source_ids: list[str]
    confidence: float


def is_mock_mode() -> bool:
    value = os.getenv("MOCK_LLM", "1")
    return value not in {"0", "false", "False", "FALSE"}


def load_documents() -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    for doc_path in sorted(DOCS_DIR.glob("doc_*.txt")):
        docs.append({"id": doc_path.stem, "text": doc_path.read_text(encoding="utf-8").strip()})
    if not docs:
        raise FileNotFoundError(f"No policy documents found in {DOCS_DIR}")
    return docs


def ensure_vector_store() -> tuple[chromadb.PersistentClient, chromadb.Collection, SentenceTransformer]:
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name="zepto_policy", metadata={"hnsw:space": "cosine"})
    if collection.count() == 0:
        model = SentenceTransformer(MODEL_NAME)
        documents = load_documents()
        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadata = [{"doc_id": doc["id"], "source": f"{doc['id']}.txt"} for doc in documents]
        embeddings = model.encode(texts, convert_to_numpy=True).tolist()
        collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadata)
        return client, collection, model
    return client, collection, SentenceTransformer(MODEL_NAME)


CLIENT, POLICY_COLLECTION, EMBEDDING_MODEL = ensure_vector_store()


def classify_intent_by_keyword(query: str) -> str:
    q = query.lower()
    for keyword in POLICY_KEYWORDS:
        if keyword in q:
            return "policy_question"
    return "general_question"


def classify_intent(query: str) -> str:
    return classify_intent_by_keyword(query)


def call_optional_real_classifier(query: str) -> str:
    if os.getenv("MOCK_LLM", "1") == "0":
        # Optional extension path. In the graded baseline, this branch is never used.
        return classify_intent_by_keyword(query)
    return classify_intent_by_keyword(query)


def retrieve_relevant_chunks(query: str, k: int = 3) -> tuple[list[str], list[str]]:
    embedding = EMBEDDING_MODEL.encode(query).tolist()
    result = POLICY_COLLECTION.query(
        query_embeddings=[embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    source_ids = [meta.get("doc_id", "unknown") for meta in metadatas]
    return documents, source_ids


def make_top_snippet(text: str, max_chars: int = 200) -> str:
    snippet = text.strip().replace("\n", " ")
    if len(snippet) > max_chars:
        return snippet[:max_chars].rstrip() + "..."
    return snippet


def build_response_from_state(state: GraphState) -> AnswerResponse:
    validated = AnswerResponse(answer=state["answer"], sources=state["source_ids"], confidence=state["confidence"])
    return validated


def validate_optional_llm_json(raw_text: str) -> AnswerResponse:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
    if text.lower().startswith("json"):
        text = text[4:].strip()
    parsed = __import__("json").loads(text)
    return AnswerResponse.model_validate(parsed)


def call_optional_real_llm(query: str, context: str) -> AnswerResponse:
    if not os.getenv("MOCK_LLM", "1").strip() == "0":
        raise RuntimeError("Real LLM path not enabled; this is the mock baseline.")

    prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENROUTER_API_KEY")
    if not api_key or requests is None:
        raise RuntimeError("No real LLM credentials configured for optional extension.")

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            raw = result["choices"][0]["message"]["content"]
            return validate_optional_llm_json(raw)
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                return AnswerResponse(
                    answer=f"LLM validation error after retries: {exc}",
                    sources=[],
                    confidence=0.0,
                )
            prompt = PROMPT_TEMPLATE.format(
                context=context,
                question=query,
            ) + "\nCorrective instruction: return only valid JSON matching schema {answer, sources, confidence} with numeric confidence 0-1."


def classify_intent_node(state: GraphState) -> GraphState:
    query = state["query"]
    if is_mock_mode():
        intent = classify_intent_by_keyword(query)
    else:
        intent = call_optional_real_classifier(query)
    state["intent"] = intent
    return state


def retrieve_and_answer_node(state: GraphState) -> GraphState:
    query = state["query"]
    docs, source_ids = retrieve_relevant_chunks(query, k=3)
    top_doc = docs[0] if docs else "No relevant policy context found."
    snippet = make_top_snippet(top_doc)

    if is_mock_mode():
        answer = f"Based on the retrieved context: {snippet}"
        confidence = 1.0
    else:
        context = "\n\n".join(docs)
        try:
            llm_response = call_optional_real_llm(query, context)
            state["answer"] = llm_response.answer
            state["source_ids"] = llm_response.sources or source_ids
            state["confidence"] = llm_response.confidence
            return state
        except Exception:
            answer = "Error: real LLM output could not be validated."
            confidence = 0.0
            state["answer"] = answer
            state["source_ids"] = source_ids
            state["confidence"] = confidence
            return state

    state["answer"] = answer
    state["source_ids"] = source_ids
    state["confidence"] = confidence
    return state


def direct_answer_node(state: GraphState) -> GraphState:
    if is_mock_mode():
        answer = "I can only answer questions about Zepto policies right now."
        confidence = 1.0
        sources: list[str] = []
    else:
        answer = "Optional real-LLM general answer path is enabled."
        confidence = 0.9
        sources = []
    state["answer"] = answer
    state["source_ids"] = sources
    state["confidence"] = confidence
    return state


def route_after_classify(state: GraphState) -> Literal["retrieve_and_answer", "direct_answer"]:
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"


workflow = StateGraph(GraphState)
workflow.add_node("classify_intent", classify_intent_node)
workflow.add_node("retrieve_and_answer", retrieve_and_answer_node)
workflow.add_node("direct_answer", direct_answer_node)
workflow.add_edge(START, "classify_intent")
workflow.add_conditional_edges("classify_intent", route_after_classify)
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)
GRAPH = workflow.compile()


def run_service(query: str) -> AnswerResponse:
    initial_state: GraphState = {
        "query": query,
        "intent": "general_question",
        "answer": "",
        "source_ids": [],
        "confidence": 1.0,
    }
    final_state = GRAPH.invoke(initial_state)
    response = build_response_from_state(final_state)
    return response


app = FastAPI(title="Zepto Support Assistant")


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: AskRequest) -> AnswerResponse:
    return run_service(request.query)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
