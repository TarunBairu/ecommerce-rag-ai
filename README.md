# ShopWise AI: Production E-Commerce Retrieval-Augmented Generation (RAG) Platform

[![CI Pipeline](https://github.com/TarunBairu/ecommerce-rag-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/TarunBairu/ecommerce-rag-ai/actions)
![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)
![Groq](https://img.shields.io/badge/LLM-Groq%20Cloud-orange.svg)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

**ShopWise AI** is a production-grade, domain-specific Retrieval-Augmented Generation (RAG) platform built for modern retail and e-commerce enterprises. 

Unlike generic document Q&A chatbots, this project demonstrates how RAG architectures are specialized to solve high-impact commercial problems:
1. **Hybrid Catalog & Policy Indexing**: Seamlessly unifies structured product inventories (SKU, brand, price, live stock) with unstructured store guidelines (returns, warranties, shipping tiers, price matches).
2. **Dynamic Query Intent Classification**: Automatically routes queries into **Product Recommendation** (*"Suggest running shoes under $130"*), **Customer Policy Support** (*"Can I return an opened laptop after 20 days?"*), or **Hybrid Inquiries** (*"If I buy the Sony XM5, what warranty covers it and can I return it?"*).
3. **Anti-Hallucination Commercial Guardrails**: Strict prompts ensuring the model never hallucinates fake discounts, never quotes outdated return windows, and alerts shoppers when an item is out of stock.
4. **Interactive Shopping Copilot Web UI**: Includes a responsive web interface with live catalog previews, star ratings, price tags, and clickable policy citation cards.

---

## 🏗️ System Architecture

```
                                   +-----------------------------------------+
                                   |      ShopWise Web Chat UI (:8002)       |
                                   +-----------------------------------------+
                                                        │ HTTP / SSE
                                                        ▼
+─────────────────────────────────────────────────────────────────────────────────────────────────+
|                                    FastAPI Application Gateway                                   |
|   [RateLimiterMiddleware] ──▶ [RequestLogger (X-Request-ID)] ──▶ [ErrorHandler (JSON Envelope)]  |
+─────────────────────────────────────────────────────────────────────────────────────────────────+
                                                        │
                      ┌─────────────────────────────────┴─────────────────────────────────┐
                      ▼                                                                   ▼
       +──────────────────────────────+                                    +──────────────────────────────+
       |   Write Path: Ingestion      |                                    |    Read Path: RAG Inference  |
       |                              |                                    |                              |
       | • CatalogLoader (JSON/CSV)   |                                    | 1. Query Intent Classifier   |
       | • PolicyLoader (Markdown)    |                                    | 2. Vector Similarity Search  |
       | • TextChunker (Preserve ctx) |                                    | 3. EcommercePromptBuilder    |
       | • Embedder Interface         |                                    | 4. Groq LLM (gpt-oss-120b)   |
       +──────────────────────────────+                                    +──────────────────────────────+
                      │                                                                   ▲
                      ▼                                                                   │
+─────────────────────────────────────────────────────────────────────────────────────────┴─────+
|                                        Dual Persistence Layer                                 |
|                                                                                               |
|  • ChromaDB Vector Store: Cosine similarity indexing over 384-d product & policy vectors       |
|  • PostgreSQL / SQLite Fallback: Relational tables for Products, Policies, and QueryLog        |
+───────────────────────────────────────────────────────────────────────────────────────────────+
```

---

## 🎓 Teaching AI with this Project: E-Commerce Curriculum

This repository serves as an educational blueprint for teaching applied Generative AI:

### Module 1: The Anatomy of Domain-Specific RAG
- **Core Lesson**: Why generic document chunking fails in e-commerce. A product description without its price, stock status, or brand context leads to useless recommendations.
- **Implementation**: Inspect [`app/ingestion/catalog_loader.py`](file:///C:/Users/TARUN%20BAIRU/.gemini/antigravity/scratch/ecommerce-rag-ai/app/ingestion/catalog_loader.py) to see how structured JSON attributes are synthesized into high-density semantic text representations.

### Module 2: Query Intent Routing
- **Core Lesson**: Real users don't specify which database to search. An AI system must intelligently classify intent.
- **Implementation**: Inspect [`app/rag/pipeline.py`](file:///C:/Users/TARUN%20BAIRU/.gemini/antigravity/scratch/ecommerce-rag-ai/app/rag/pipeline.py) for the multi-intent detector distinguishing:
  - `product_recommendation`
  - `policy_support`
  - `hybrid`

### Module 3: Commercial Guardrails & Citations
- **Core Lesson**: LLMs cannot be trusted to generate promotional prices or return rules without deterministic grounding.
- **Implementation**: Inspect [`app/rag/prompt_builder.py`](file:///C:/Users/TARUN%20BAIRU/.gemini/antigravity/scratch/ecommerce-rag-ai/app/rag/prompt_builder.py) to review how the model is forced to cite specific products (`[Source 1: PRODUCT - AUD-SONY-001]`) or policy sections (`[Source 2: POLICY - Return and Refund Policy]`).

### Module 4: Dual Persistence (Relational + Vector)
- **Core Lesson**: Vector stores provide semantic recall, while relational databases enforce ACID transactions, inventory counts, and audit logs.
- **Implementation**: Inspect [`app/db/models.py`](file:///C:/Users/TARUN%20BAIRU/.gemini/antigravity/scratch/ecommerce-rag-ai/app/db/models.py) and [`app/vectorstore/vector_db.py`](file:///C:/Users/TARUN%20BAIRU/.gemini/antigravity/scratch/ecommerce-rag-ai/app/vectorstore/vector_db.py).

---

## 🚀 Quickstart Guide

### 1. Installation
Ensure Python 3.11+ is installed.

```bash
cd "C:\Users\TARUN BAIRU\.gemini\antigravity\scratch\ecommerce-rag-ai"
pip install -r requirements.txt
```

### 2. Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Ensure your `GROQ_API_KEY` is specified.

### 3. Run the Application
Start the FastAPI server on port **8002**:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8002 --reload
```

### 4. Open the Web Copilot
- **Interactive Web Chat UI**: 👉 [`http://localhost:8002/chat`](http://localhost:8002/chat) or [`http://localhost:8002/`](http://localhost:8002/)
- **Swagger OpenAPI Docs**: 👉 [`http://localhost:8002/docs`](http://localhost:8002/docs)

---

## 🧪 Testing

Run the automated test suite:
```bash
python -m pytest tests/ -v
```
All 10 unit and integration tests validate the complete end-to-end flow.

---

## 📡 API Reference & Examples

### Ask the Shopping Copilot
```bash
curl -X POST "http://localhost:8002/api/v1/chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Recommend running shoes under $130 with high customer ratings",
    "top_k": 3
  }'
```

### Ask a Store Policy Question
```bash
curl -X POST "http://localhost:8002/api/v1/chat/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Can I return an opened laptop after 20 days and who pays shipping?",
    "top_k": 3
  }'
```

### Stream Answer Tokens (SSE)
```bash
curl -N -X POST "http://localhost:8002/api/v1/chat/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare Sony XM5 vs Bose QC45"}'
```
