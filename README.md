<div align="center">

<img src="https://img.shields.io/badge/Self--RAG-Advanced%20RAG%20Pipeline-blueviolet?style=for-the-badge&logo=python&logoColor=white" alt="Self-RAG" />

# 🧠 Self-RAG: Self-Reflective Retrieval-Augmented Generation

### *A step-by-step implementation of Corrective & Adaptive RAG using LangGraph, LangChain, and LLaMA 3*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-1DB954?style=flat-square&logo=langchain&logoColor=white)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/LangChain-Latest-00ADB5?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-FF6F61?style=flat-square&logo=meta&logoColor=white)](https://github.com/facebookresearch/faiss)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-FFD43B?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.1-F55036?style=flat-square&logo=meta&logoColor=white)](https://groq.com)
[![Tavily](https://img.shields.io/badge/Tavily-Web%20Search-4A90D9?style=flat-square&logo=google&logoColor=white)](https://tavily.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

<br/>

> *Built over one week of focused research, experimentation, and incremental implementation — going from zero to a full adaptive, self-correcting RAG pipeline with 8 progressive steps.*

<br/>

[**📖 Research Paper**](#-research-paper) • [**🗺️ Architecture**](#️-architecture--system-design) • [**🚀 Getting Started**](#-getting-started) • [**📚 Step-by-Step Journey**](#-step-by-step-implementation-journey) • [**🧩 Tech Stack**](#-tech-stack)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [What Makes This Different from Standard RAG](#-what-makes-this-different-from-standard-rag)
- [Research Paper](#-research-paper)
- [Architecture & System Design](#️-architecture--system-design)
- [Step-by-Step Implementation Journey](#-step-by-step-implementation-journey)
- [Reflection Tokens Implemented](#-reflection-tokens-implemented)
- [Tech Stack](#-tech-stack)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Key Learnings](#-key-learnings)
- [Acknowledgements](#-acknowledgements)

---

## 🔍 Overview

This repository contains a **complete, production-style implementation of Self-RAG** — a state-of-the-art Retrieval-Augmented Generation framework that goes beyond naive RAG by making the LLM **reason about when to retrieve, whether retrieved docs are relevant, whether its own answer is supported, and whether the answer is actually useful**.

Unlike traditional RAG pipelines that blindly prepend retrieved documents to every query, **Self-RAG is adaptive and self-correcting**. It uses a LangGraph state machine to orchestrate a multi-node reasoning pipeline inspired by the original [Self-RAG paper (Asai et al., 2023)](https://arxiv.org/abs/2310.11511).

The project was built **step-by-step over 8 progressive notebooks**, each adding one new capability — mirroring how you'd incrementally build and understand a research system in practice.

---

## ⚡ What Makes This Different from Standard RAG

| Feature | Standard RAG | **Self-RAG (This Project)** |
|---|---|---|
| Retrieval decision | Always retrieves | 🧠 **Decides whether retrieval is needed** |
| Document filtering | Uses all retrieved docs | ✅ **Filters irrelevant documents (IsREL)** |
| Answer grounding | No verification | 🔍 **Verifies answer is supported by context (IsSUP)** |
| Answer quality | No quality check | 🎯 **Checks if answer is actually useful (IsUSE)** |
| Retry on failure | No retries | 🔄 **Loops back and revises if answer fails checks** |
| Query optimization | Fixed query | ✍️ **Rewrites query if retrieval fails** |
| Fallback strategy | Returns nothing | 🌐 **Falls back to live web search via Tavily** |

---

## 📄 Research Paper

This project is a from-scratch implementation inspired by:

> **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**  
> Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi  
> University of Washington & Allen Institute for AI  
> [arXiv:2310.11511](https://arxiv.org/abs/2310.11511) · October 2023

### Key Concepts from the Paper

The original Self-RAG paper introduces **4 types of reflection tokens** that an LLM learns to generate to guide its own reasoning:

| Token | Purpose | Values |
|-------|---------|--------|
| `Retrieve` | Should the model retrieve external documents? | `Yes` / `No` / `Continue` |
| `IsREL` | Is the retrieved document relevant to the question? | `Relevant` / `Irrelevant` |
| `IsSUP` | Is the generated answer supported by the retrieved context? | `Fully Supported` / `Partially Supported` / `No Support` |
| `IsUSE` | Is the final answer useful for the user? | `1–5` scale |

> 💡 In this implementation, instead of fine-tuning an LLM to produce these tokens natively, we **simulate them as structured LLM calls** using Pydantic + LangGraph nodes — making the approach accessible without requiring custom model training.

---

## 🗺️ Architecture & System Design

The final pipeline is a directed graph with conditional edges, implemented using **LangGraph's StateGraph**. Here is the complete flow:

```
                    ┌─────────────────────────────────┐
                    │           User Question         │
                    └────────────────┬────────────────┘
                                     │
                           ┌─────────▼───────── ─┐
                           │   decide_retrieval  │  ← IsRETRIEVE token
                           │  (Need docs or not?)│
                           └────┬──────────┬─────┘
                                │          │
                     [Yes]      │          │  [No]
                                │          │
               ┌────────────────▼──┐   ┌──▼─────────────────── ─┐
               │     retrieve      │   │    generate_direct     │
               │ (FAISS vector     │   │ (Answer from LLM       │
               │  similarity)      │   │  parametric knowledge) │
               └────────┬──────────┘   └──────────────┬────── ──┘
                        │                             │
              ┌─────────▼──────────┐                  │
              │     is_relevant    │  ← IsREL token   │
              │(Filter irrelevant  │                  │
              │    documents)      │                  │
              └──┬─────────────────┘                  │
                 │                                    │
    [Relevant]   │             [No Relevant Docs]     │
                 │                    │               │
    ┌────────────▼────────┐    ┌──────▼──────────┐    │
    │ generate_from_ctx   │    │  rewrite_query  │    │
    │(Answer grounded in  │    │ (Optimize query │    │
    │  retrieved context) │    │  for retrieval) │    │
    └────────┬────────────┘    └──────┬──────────┘    │
             │                        │               │
    ┌────────▼────────┐        ┌──────▼──────────┐    │
    │     is_sup      │◄─loop  │   web_search    │    │
    │(Is answer backed │       │(Tavily fallback │    │
    │  by evidence?)  │        │  to live web)   │    │
    └──┬──────────────┘        └─────────────────┘    │
       │                                              │
  [Fully/Partial]     [No Support]                    │
       │                    │                         │
  ┌────▼─────┐        ┌─────▼──────┐                  │
  │  is_use  │        │ revise_ans │                  │
  │(Is answer│        │  (Retry)   │                  │
  │ useful?) │        └─────┬──────┘                  │
  └──┬───────┘              │ (loops back to is_sup)  │
     │                      │                         │
  [Useful]  [Not Useful]    │                         │
     │           │          │                         │
     │    ┌──────▼────┐     │                         │
     │    │no_ans_fnd │     │                         │
     │    └───────────┘     │                         │
     │                      │                         │
  ┌──▼──────────────────────▼─────────────────────────▼─ ─┐
  │                        END                            │
  └───────────────────────────────────────────────────────┘
```

---

## 📚 Step-by-Step Implementation Journey

One of the core design philosophies of this project was **incremental complexity** — each notebook adds exactly one new concept, so you can understand every part of the system before the next one is introduced.

### Step 1 — `self_rag_step1.ipynb` · Basic Retrieval Decision
**Concept Introduced:** `IsRETRIEVE` — Should the LLM even retrieve documents?

The very first step breaks from the naive RAG assumption that every query needs retrieval. A structured LLM call decides whether the question requires external facts or can be answered from parametric knowledge alone.

```python
class RetrieveDecision(BaseModel):
    should_retrieve: bool  # True = fetch documents, False = answer directly

# Graph: decide_retrieval → [retrieve | generate_direct] → END
```

**What I learned:** Not every question needs retrieval. General definitions, math, reasoning — these don't benefit from retrieved documents and may actually be degraded by irrelevant noise.

---

### Step 2 — `self_rag_step2.ipynb` · Document Relevance Filtering (IsREL)
**Concept Introduced:** `IsREL` — Are the retrieved documents actually relevant?

Even when retrieval is triggered, the retrieved chunks may not be useful. This step adds a relevance filter that evaluates each document and keeps only the ones that help answer the question.

```python
# New state key: relevant_docs (filtered from raw docs)
# New node: is_relevant → filters docs before generation
```

**What I learned:** FAISS retrieves semantically similar chunks, not necessarily *relevant* ones. A secondary LLM-based filter dramatically improves answer quality by removing noisy context.

---

### Step 3 — `self_rag_step3.ipynb` · Context-Grounded Generation
**Concept Introduced:** Context stuffing + grounded answer generation

With relevant documents identified, this step builds a structured context block and passes it to the generation node. A fallback `no_relevant_docs` node handles the case when no useful documents are found.

```python
# New state key: context (formatted relevant docs)
# New nodes: generate_from_context, no_relevant_docs
```

**What I learned:** The format of the context passed to the LLM matters significantly. Structuring it clearly (with document separators) yields more faithful, grounded answers.

---

### Step 4 — `self_rag_step4.ipynb` · Answer Support Verification (IsSUP)
**Concept Introduced:** `IsSUP` — Is the generated answer actually supported by evidence?

This is where the "self-reflective" aspect becomes real. After generating an answer, the LLM is asked to verify whether every claim in that answer is explicitly supported by the retrieved context.

```python
class IsSUPDecision(BaseModel):
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]  # Which parts of context support the answer

# IsSUP grades: fully_supported → accept | partially/no_support → revise
```

**What I learned:** LLMs frequently "hallucinate" qualitative language like *"generous leave policy"* or *"employee-first culture"* even when the context only contains raw facts. IsSUP catches these fabrications and forces a revision.

---

### Step 5 — `self_rag_step5.ipynb` · Retry Loop with Max Attempts
**Concept Introduced:** `retries` counter + bounded revision loop

IsSUP alone triggers infinite revision loops if the LLM keeps generating unsupported answers. This step introduces a `MAX_RETRIES` guard and an `accept_answer` path that exits the loop gracefully.

```python
# New state key: retries (int)
MAX_RETRIES = 2

def route_after_issup(state):
    if state.get("retries", 0) >= MAX_RETRIES:
        return "accept_answer"   # exit loop even if partially supported
    return "revise_answer"       # try again
```

**What I learned:** Unbounded loops are a real engineering concern in agentic systems. Adding retry limits is not a hack — it's essential for production reliability.

---

### Step 6 — `self_rag_step6.ipynb` · Usefulness Check (IsUSE)
**Concept Introduced:** `IsUSE` — Does the answer actually help the user?

Even a factually-grounded answer can fail to answer the question. This step adds a final usefulness gate that checks whether the answer directly responds to what was asked — independently of whether it's factually supported.

```python
class IsUSEDecision(BaseModel):
    isuse: Literal["useful", "not_useful"]
    reason: str  # Short explanation

# IsUSE: useful → END | not_useful → no_answer_found
```

**What I learned:** IsSUP and IsUSE test completely different things. IsSUP = *"Is this grounded in evidence?"* IsUSE = *"Did we actually answer the question?"* A fully-supported answer can still be useless if it answers the wrong thing.

---

### Step 7 — `self_rag_step7.ipynb` · Query Rewriting for Better Retrieval
**Concept Introduced:** `retrieval_query` + query rewriting when IsUSE fails

When the answer is not useful, the original question may have been too vague for effective vector retrieval. This step adds a query rewriter that reformulates the question into a more retrieval-optimized form and tries again.

```python
# New state keys: retrieval_query (str), rewrite_tries (int)
# Supports multi-document KB: Company_Policies + Profile + Pricing PDFs

# retrieve() uses retrieval_query instead of raw question
q = state.get("retrieval_query") or state["question"]
docs = retriever.invoke(q)
```

**What I learned:** The phrasing of a retrieval query has a huge effect on what FAISS returns. A domain-specific rewriter ("Rewrite for internal company document retrieval") dramatically improves recall on structured knowledge bases.

---

### Bonus — `self_rag_web.ipynb` · Web Search Fallback via Tavily
**Concept Introduced:** Live web search as a fallback when local docs fail

When no relevant local documents are found, instead of giving up, the pipeline rewrites the query and searches the live web using the Tavily Search API. This makes the system handle questions that are outside the local knowledge base.

```python
# New nodes: rewrite_query, web_search
from langchain_community.tools.tavily_search import TavilySearchResults
tavily = TavilySearchResults(max_results=5)

def web_search_node(state: State):
    results = tavily.invoke({"query": state.get("web_query")})
    # Convert results to Document objects and pass to relevance filter
```

**What I learned:** Hybrid retrieval (local vector store + live web) is far more robust than either alone. Tavily's structured results integrate cleanly with LangChain's Document interface.

---

## 🏷️ Reflection Tokens Implemented

| Token | Node | Description | Values |
|-------|------|-------------|--------|
| `IsRETRIEVE` | `decide_retrieval` | Should we retrieve? | `True` / `False` |
| `IsREL` | `is_relevant` | Are docs relevant? | `relevant` / `irrelevant` |
| `IsSUP` | `is_sup` | Is answer grounded? | `fully_supported` / `partially_supported` / `no_support` |
| `IsUSE` | `is_use` | Is answer useful? | `useful` / `not_useful` |

---

## 🧩 Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) | State machine graph for multi-node RAG pipeline |
| **LLM Framework** | [LangChain](https://langchain.com) | Chains, prompts, structured output |
| **LLM Inference** | [Groq](https://groq.com) + LLaMA 3.1 8B Instant | Ultra-fast LLM inference |
| **Vector Store** | [FAISS](https://github.com/facebookresearch/faiss) | Efficient dense similarity search |
| **Embeddings** | [HuggingFace](https://huggingface.co) · `all-MiniLM-L6-v2` | Sentence-level semantic embeddings |
| **Document Loader** | LangChain `PyPDFLoader` | PDF ingestion and chunking |
| **Text Splitting** | `RecursiveCharacterTextSplitter` | Chunk size 600, overlap 150 |
| **Web Search** | [Tavily](https://tavily.com) | Live web search fallback |
| **Structured Output** | [Pydantic](https://docs.pydantic.dev) | Type-safe LLM responses for reflection tokens |
| **Env Management** | `python-dotenv` | API key management |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com) (free)
- A [Tavily API key](https://tavily.com) (for web search notebook, free tier available)
- A [HuggingFace account](https://huggingface.co) (for embeddings)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/self-rag.git
cd self-rag

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here   # Only needed for self_rag_web.ipynb
HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
```

### Running the Notebooks

Start with Step 1 and progress sequentially for the best learning experience:

```bash
jupyter notebook self_rag_step1.ipynb
```

Or open all notebooks:

```bash
jupyter lab
```

> 💡 **Tip:** Add your PDF documents to the `Books/` folder (for steps 1–6) or `documents/` folder (for step 7 and web) before running. Each notebook references these paths for document ingestion.

---

## 📁 Project Structure

```
self-rag/
│
├── 📓 self_rag_step1.ipynb       # Step 1: IsRETRIEVE — Adaptive retrieval decision
├── 📓 self_rag_step2.ipynb       # Step 2: IsREL — Document relevance filtering
├── 📓 self_rag_step3.ipynb       # Step 3: Context-grounded generation
├── 📓 self_rag_step4.ipynb       # Step 4: IsSUP — Answer support verification
├── 📓 self_rag_step5.ipynb       # Step 5: Retry loop with MAX_RETRIES guard
├── 📓 self_rag_step6.ipynb       # Step 6: IsUSE — Final usefulness check
├── 📓 self_rag_step7.ipynb       # Step 7: Query rewriting for better retrieval
├── 📓 self_rag_web.ipynb         # Bonus: Web search fallback via Tavily
│
├── 📄 Self-Rag-paper.pdf         # Original research paper (Asai et al., 2023)
├── 📄 requirements.txt           # Python dependencies
├── 📄 .env.example               # Template for API keys
│
├── 📂 Books/                     # PDF documents for steps 1–6
│   └── book1.pdf
│
└── 📂 documents/                 # Multi-document KB for steps 7 & web
    ├── Company_Policies.pdf
    ├── Company_Profile.pdf
    └── Product_and_Pricing.pdf
```

---

## 🧠 Key Learnings

This project was built over **one week of focused learning**, going from reading the research paper to implementing a fully functioning Self-RAG pipeline. Here are the most important takeaways:

### 1. Adaptive Retrieval Is Not Optional
The assumption that every question needs retrieval is the single biggest weakness in naive RAG. Adding `IsRETRIEVE` as the first node immediately makes the system faster and more accurate for general-knowledge queries.

### 2. Relevance Filtering Changes Everything
FAISS cosine similarity returns *similar* chunks, not *relevant* chunks. They're not the same thing. A secondary LLM-based relevance filter acts as a quality gate that transforms the quality of downstream generation.

### 3. LLMs Hallucinate on Context Too
This was the most surprising finding: even when given accurate retrieved context, LLMs add qualitative language that isn't supported by the evidence. IsSUP catches phrases like *"employee-first culture"* or *"designed to support professional development"* that the model infers but the document never says.

### 4. Grounded ≠ Useful
A fully-supported answer can still completely fail to answer the user's question. IsSUP and IsUSE serve complementary, non-overlapping roles and both are necessary.

### 5. LangGraph Is the Right Tool for This
LangGraph's `StateGraph` with `TypedDict` state, conditional edges, and cycle support is exactly what complex multi-hop, multi-check RAG pipelines need. Chains alone can't express this kind of dynamic control flow.

### 6. Retry Limits Are Architecture, Not Afterthoughts
Adding a `MAX_RETRIES` counter sounds simple, but it forced me to think carefully about what "giving up gracefully" means — and what information to preserve for the user when the system can't produce a verified answer.

### 7. Query Rewriting Matters as Much as Retrieval
The same FAISS index with a better-formulated query can return dramatically different (and better) results. Decoupling the user's question from the retrieval query is one of the highest-leverage improvements in the whole pipeline.

---

## 📖 References

- **Self-RAG Paper:** Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2023). *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection.* [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
- **LangGraph Docs:** https://langchain-ai.github.io/langgraph/
- **FAISS:** Johnson, J., Douze, M., & Jégou, H. (2017). *Billion-scale similarity search with GPUs.*
- **LLaMA 3:** Meta AI. (2024). *The Llama 3 Herd of Models.*
- **Tavily Search:** https://docs.tavily.com

---

## 🙏 Acknowledgements

Huge thanks to the original Self-RAG authors at the **University of Washington** and **Allen Institute for AI** for publishing such a clear and well-structured paper. The reflection token framework they introduced is elegant and surprisingly practical to implement even without fine-tuning.

---

<div align="center">

**Built with 🔥 dedication, curiosity, and a lot of debugging over one week.**

*If this project helped you understand Self-RAG or advanced RAG techniques, please consider giving it a ⭐*

<br/>

[![LinkedIn](https://img.shields.io/badge/Connect-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/saurabsharma/)

</div>
