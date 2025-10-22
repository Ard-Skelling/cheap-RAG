# cheap-RAG Development Roadmap

## Overview

This roadmap outlines planned enhancements to transform cheap-RAG from a functional document retrieval system into a production-ready, state-of-the-art RAG framework. Priorities are based on impact vs. effort analysis and alignment with mainstream RAG best practices.

---

## Phase 1: Core RAG Completion & Production Readiness (Q1 2025)

**Goal:** Complete missing core functionality and ensure production stability

### 1.1 Generation Integration (Week 1-2)
**Priority:** 🔴 Critical
**Status:** Not Started

**Tasks:**
- [ ] Create `/api/v1/rag/complete` endpoint for full RAG pipeline
- [ ] Implement context formatting from retrieved chunks
- [ ] Add streaming response support for LLM generation
- [ ] Implement citation tracking (link answers to source chunks)
- [ ] Add configurable prompt templates
- [ ] Support multiple LLM backends (OpenAI, Anthropic, local models)

**Deliverables:**
```python
# Example endpoint
@router.post('/rag/complete')
async def rag_complete(request: RAGRequest):
    """Full RAG: Retrieval + Generation"""
    chunks = await retrieve(request.query, request.domain)
    answer = await generate(chunks, request.query)
    return {"answer": answer, "sources": chunks, "citations": [...]}
```

**Acceptance Criteria:**
- End-to-end RAG workflow functional
- Citations traceable to source documents
- Response time < 3s for typical queries

---

### 1.2 Advanced Reranking (Week 3-4)
**Priority:** 🔴 Critical
**Status:** Not Started

**Tasks:**
- [ ] Implement Reciprocal Rank Fusion (RRF) to replace naive cross-sort
- [ ] Add cross-encoder reranker (bge-reranker-v2-m3 or Cohere)
- [ ] Create reranker abstraction for pluggable models
- [ ] Add reranker performance benchmarks
- [ ] Make reranking optional via config flag

**Technical Details:**
```python
# cheap_rag/modules/workflow/workflow_paper/query_workflow/reranker.py
class RRFReranker:
    def rerank(self, results_lists: List[List[dict]], k: int = 60) -> List[dict]:
        """Reciprocal Rank Fusion"""

class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """Cross-encoder reranking for top-k refinement"""
```

**Expected Impact:**
- 20-30% improvement in retrieval relevance (NDCG@10)
- Better handling of semantic vs. keyword matching conflicts

---

### 1.3 Metadata Filtering (Week 5)
**Priority:** 🟡 High
**Status:** TODO exists in code (query_workflow/workflow_tasks.py:64)

**Tasks:**
- [ ] Design metadata schema (date, source, author, tags, etc.)
- [ ] Update document insertion to accept metadata
- [ ] Implement metadata filtering in Milvus search
- [ ] Implement metadata filtering in Elasticsearch search
- [ ] Add filter DSL to `/search` API
- [ ] Document filter syntax and examples

**API Example:**
```json
POST /api/v1/query/search
{
  "query": "machine learning trends",
  "domain": "research_papers",
  "filter": {
    "date_range": {"gte": "2024-01-01", "lte": "2024-12-31"},
    "source": ["arxiv", "pubmed"],
    "tags": {"any": ["ai", "ml"]}
  }
}
```

---

### 1.4 Production Hardening (Week 6-8)
**Priority:** 🟡 High
**Status:** Partial (basic error handling exists)

**Tasks:**
- [ ] **Error Handling:**
  - [ ] Replace generic exceptions with specific types
  - [ ] Add retry logic with exponential backoff (tenacity library)
  - [ ] Implement circuit breakers for external services
  - [ ] Structured error responses with error codes

- [ ] **Observability:**
  - [ ] Add Prometheus metrics (query latency, throughput, error rates)
  - [ ] Implement structured logging (query_id tracking)
  - [ ] Add health check endpoints for dependencies (Milvus, ES, MinIO)
  - [ ] Create Grafana dashboard templates

- [ ] **Rate Limiting:**
  - [ ] Per-user/per-IP rate limiting (slowapi)
  - [ ] Token bucket for embedding API calls
  - [ ] Queue management for bulk operations

- [ ] **Testing:**
  - [ ] Unit tests for core modules (target: 70% coverage)
  - [ ] Integration tests for workflow pipelines
  - [ ] Load testing scenarios (locust/k6)

**Metrics to Track:**
```python
# Prometheus metrics
rag_queries_total (counter)
rag_query_latency_seconds (histogram)
rag_retrieval_results (histogram)
rag_errors_total (counter by type)
embedding_cache_hit_rate (gauge)
```

---

## Phase 2: Performance & Cost Optimization (Q2 2025)

**Goal:** Reduce latency and operational costs without sacrificing quality

### 2.1 Intelligent Caching (Week 9-10)
**Priority:** 🟡 High
**Status:** Not Started

**Tasks:**
- [ ] **Embedding Cache:**
  - [ ] In-memory LRU cache for query embeddings
  - [ ] Redis backend for distributed caching
  - [ ] Cache warming strategies
  - [ ] TTL and invalidation policies

- [ ] **Result Cache:**
  - [ ] Cache full retrieval results (query hash → chunks)
  - [ ] Configurable TTL based on collection update frequency
  - [ ] Cache-aside pattern with automatic refresh

- [ ] **Metadata Cache:**
  - [ ] Cache file name mappings (MD5 → original name)
  - [ ] Cache collection schemas

**Expected Impact:**
- 5-10x speedup for repeated queries
- 60-80% reduction in embedding API calls
- Lower infrastructure costs

**Implementation:**
```python
# cheap_rag/modules/cache/
├── embedding_cache.py
├── result_cache.py
└── metadata_cache.py
```

---

### 2.2 Batch Processing & Concurrency (Week 11)
**Priority:** 🟢 Medium
**Status:** Partial (async exists, batching incomplete)

**Tasks:**
- [ ] Batch query embeddings (multiple queries → single embedding call)
- [ ] Implement request coalescing for identical queries
- [ ] Optimize Elasticsearch multi-search batch sizes
- [ ] Add connection pooling configuration tuning
- [ ] Implement worker auto-scaling based on queue depth

**Expected Impact:**
- 3-5x throughput improvement under load
- Better resource utilization

---

### 2.3 Model Optimization (Week 12)
**Priority:** 🟢 Medium
**Status:** Not Started

**Tasks:**
- [ ] Evaluate smaller embedding models for latency-sensitive queries
- [ ] Implement model quantization (ONNX int8)
- [ ] Support Matryoshka embeddings (variable dimensions)
- [ ] Binary/int8 vector search in Milvus
- [ ] A/B testing framework for model swaps

**Models to Evaluate:**
- **Current:** `multilingual-e5-large-instruct` (560M params, 1024-dim)
- **Alternatives:**
  - `bge-small-en-v1.5` (33M params, 384-dim) - 5x faster
  - `gte-small` (lightweight)
  - Custom fine-tuned models

---

## Phase 3: Advanced Retrieval Features (Q2-Q3 2025)

**Goal:** Implement state-of-the-art retrieval techniques

### 3.1 Query Enhancement (Week 13-15)
**Priority:** 🟢 Medium
**Status:** Not Started

**Tasks:**
- [ ] **HyDE (Hypothetical Document Embeddings):**
  - [ ] Generate hypothetical answer with LLM
  - [ ] Embed hypothetical answer instead of query
  - [ ] Configurable via API parameter

- [ ] **Multi-Query Expansion:**
  - [ ] Generate query variations with LLM
  - [ ] Parallel retrieval for each variation
  - [ ] Result fusion and deduplication

- [ ] **Query Decomposition:**
  - [ ] Break complex queries into sub-queries
  - [ ] Sequential or parallel sub-query execution
  - [ ] Result synthesis

- [ ] **Step-Back Prompting:**
  - [ ] Ask broader conceptual questions first
  - [ ] Use results to guide detailed retrieval

**Configuration:**
```yaml
# cheap_rag/modules/workflow/workflow_paper/config.py
query_enhancement:
  enabled: true
  strategies: ["hyde", "multi_query"]
  hyde:
    llm_model: "gpt-4o-mini"
    temperature: 0.7
  multi_query:
    num_variations: 3
```

---

### 3.2 Result Diversification (Week 16)
**Priority:** 🟢 Medium
**Status:** Not Started

**Tasks:**
- [ ] Implement Maximal Marginal Relevance (MMR)
- [ ] Add diversity parameter to API
- [ ] Support clustering-based diversification
- [ ] Time-based diversity (spread across documents/dates)

**API Enhancement:**
```python
@router.post('/search')
async def search(request: DataSearchEngineRequest):
    result = await QUERY_WORKFLOW.search(
        query=request.query,
        domain=request.domain,
        topk=request.topk,
        diversity=request.diversity,  # 0.0-1.0, 0=pure relevance, 1=max diversity
        diversity_method=request.diversity_method  # "mmr" | "clustering"
    )
```

---

### 3.3 Conversational RAG (Week 17-19)
**Priority:** 🟢 Medium
**Status:** Not Started

**Tasks:**
- [ ] Add session management (conversation history)
- [ ] Implement query rephrasing with conversation context
- [ ] Support follow-up questions
- [ ] Memory compression for long conversations
- [ ] Conversational endpoints (`/chat/start`, `/chat/message`, `/chat/end`)

**Architecture:**
```python
# cheap_rag/modules/conversation/
├── session_manager.py    # Redis-backed session storage
├── query_rewriter.py     # Rewrite queries with chat history
└── memory.py             # Conversation memory management
```

---

## Phase 4: Content Understanding (Q3 2025)

**Goal:** Better extraction and understanding of multimodal content

### 4.1 Semantic Chunking (Week 20-21)
**Priority:** 🟢 Medium
**Status:** Not Started (current: fixed-size chunking)

**Tasks:**
- [ ] Implement semantic boundary detection
- [ ] Use embedding similarity for chunk splitting
- [ ] LLM-based agentic chunking
- [ ] Proposition-based chunking
- [ ] A/B test against current token-based chunking

**Expected Impact:**
- More coherent chunks
- Better retrieval precision
- Reduced context loss at boundaries

---

### 4.2 Enhanced Table Understanding (Week 22-23)
**Priority:** 🟢 Medium
**Status:** Partial (basic extraction exists)

**Tasks:**
- [ ] Enable table Q&A generation (`build_table_qa=True`)
- [ ] Table summarization with LLM
- [ ] Table-to-text natural language conversion
- [ ] Column type detection and schema extraction
- [ ] Table chain-of-thought reasoning

---

### 4.3 Vision & Multimodal (Week 24-26)
**Priority:** 🔵 Low
**Status:** Basic (URL + caption only)

**Tasks:**
- [ ] Integrate vision LLM (GPT-4V, LLaVA, Qwen-VL)
- [ ] Generate image descriptions and Q&A pairs
- [ ] CLIP embeddings for image similarity search
- [ ] OCR for text extraction from images
- [ ] Chart/diagram understanding
- [ ] Multimodal retrieval (text + image queries)

---

## Phase 5: Advanced Features (Q4 2025)

**Goal:** Cutting-edge RAG capabilities

### 5.1 Knowledge Graph Integration (Week 27-30)
**Priority:** 🔵 Low
**Status:** Not Started

**Tasks:**
- [ ] Entity extraction from documents
- [ ] Relation extraction
- [ ] Knowledge graph construction (Neo4j)
- [ ] Graph-based retrieval (graph RAG)
- [ ] Hybrid: Vector + BM25 + Graph

---

### 5.2 Retrieval-Augmented Fine-tuning (Week 31-34)
**Priority:** 🔵 Low
**Status:** Not Started

**Tasks:**
- [ ] Collect query-document relevance feedback
- [ ] Fine-tune embedding models on domain data
- [ ] Hard negative mining
- [ ] Continuous learning pipeline
- [ ] Model versioning and rollback

---

### 5.3 Contextual Compression (Week 35-36)
**Priority:** 🔵 Low
**Status:** Not Started

**Tasks:**
- [ ] Retrieve large candidate set (topk=50)
- [ ] Extract relevant sentences/passages
- [ ] Compress with extractive/abstractive methods
- [ ] Return compressed context to LLM

---

## Phase 6: Evaluation & Iteration (Ongoing)

**Goal:** Continuous improvement through measurement

### 6.1 Evaluation Framework (Week 37-40)
**Priority:** 🟡 High
**Status:** Not Started

**Tasks:**
- [ ] **Dataset Creation:**
  - [ ] Golden query-answer pairs
  - [ ] Relevance judgments for retrieval
  - [ ] Adversarial/edge cases

- [ ] **Metrics Implementation:**
  - [ ] Retrieval: NDCG@k, MRR, Recall@k
  - [ ] Generation: BLEU, ROUGE, BERTScore
  - [ ] RAG-specific: Answer relevance, faithfulness, context precision
  - [ ] Ragas integration

- [ ] **Automated Testing:**
  - [ ] Regression tests on golden set
  - [ ] A/B testing framework
  - [ ] Champion/challenger model evaluation

---

## Milestones & Success Metrics

### Milestone 1: Production-Ready Core (End of Phase 1)
- ✅ Full RAG generation available
- ✅ 99.9% API uptime
- ✅ P95 latency < 2s
- ✅ Error rate < 0.1%
- ✅ 70% test coverage

### Milestone 2: Performance Optimized (End of Phase 2)
- ✅ 80% cache hit rate for common queries
- ✅ 5x throughput improvement vs. baseline
- ✅ 50% reduction in infrastructure costs

### Milestone 3: SOTA Retrieval (End of Phase 3)
- ✅ NDCG@10 > 0.85 on benchmark
- ✅ Support for 5+ query enhancement strategies
- ✅ Conversational RAG functional

### Milestone 4: Multimodal Excellence (End of Phase 4)
- ✅ Table understanding accuracy > 90%
- ✅ Image Q&A functional
- ✅ Semantic chunking shows 10%+ relevance improvement

### Milestone 5: Advanced Capabilities (End of Phase 5)
- ✅ Knowledge graph retrieval operational
- ✅ Domain-tuned embedding models deployed
- ✅ Contextual compression reduces token usage by 40%

---

## Risk Mitigation

### Technical Risks
- **Model size/latency trade-off:** Use tiered models (fast small model → accurate large model)
- **Cache invalidation complexity:** Implement conservative TTLs, manual invalidation API
- **Backward compatibility:** Version API endpoints, support legacy formats

### Operational Risks
- **Resource constraints:** Implement auto-scaling, queue management
- **Dependency failures:** Circuit breakers, fallback strategies
- **Data quality:** Validation pipelines, quality scoring

---

## Resource Requirements

### Team
- **Phase 1-2:** 2 backend engineers (full-time)
- **Phase 3-4:** +1 ML engineer (full-time)
- **Phase 5:** +1 research engineer (part-time)

### Infrastructure
- **Current:** Milvus + ES + MinIO (single node)
- **Phase 2:** Redis cache, load balancer
- **Phase 3:** Multi-node clusters, GPU for reranking
- **Phase 5:** Neo4j, model training infrastructure

### Budget (Estimated)
- **Phase 1:** $5K/month (API costs, compute)
- **Phase 2:** $8K/month (+caching, scaling)
- **Phase 3:** $12K/month (+GPU, LLM calls)
- **Phase 4-5:** $15K/month (+vision models, graph DB)

---

## Review & Adjustment

This roadmap will be reviewed **quarterly** with adjustments based on:
- User feedback and feature requests
- Performance metrics and bottlenecks
- Industry trends and new research
- Resource availability and priorities

**Next Review:** End of Q1 2025

---

## Contributing

For questions or suggestions about this roadmap, please:
1. Open an issue in the repository
2. Tag with `roadmap` label
3. Provide use case and impact analysis

**Last Updated:** 2025-10-22
