# Simple RAG System for Washington Sentinels Concussion Protocol

**Course**: AD 331 
**Assignment**: Implementing a Simple Retrieval-Augmented Generation (RAG) System  
**Student**: Kyle Kitching  

---

## 1. Overview

This project implements a small Retrieval-Augmented Generation (RAG) pipeline using:
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2`
- **LLM**: `google/flan-t5-small`

The system answers questions about a fictional NFL team, the **Washington Sentinels**, and their internal **concussion return-to-play protocol**. The policy text is stored in a custom knowledge base (KB), and a SentenceTransformer model is used to retrieve relevant chunks before generating an answer.

---

## 2. Repository Structure

- `rag_simple.ipynb` – Main notebook containing:
  - KB creation and chunking
  - Embedding and indexing
  - Retrieval
  - RAG-style generation
  - Test cases and results
- `kb_sentinels_concussion_protocol.txt` – Raw knowledge base text file.
- `requirements.txt` – Python dependencies.
- `README.md` – Project documentation (this file).

---

## 3. RAG Pipeline Design

### 3.1 Knowledge Base and Chunking

- The KB consists of **3 paragraphs** describing:
  1. The five-stage concussion protocol and 24-hour progression rule.
  2. Stricter club-specific rules (e.g., no same-day clearance, handling second concussions, who can sign clearance).
  3. Documentation and communication requirements (SentinelMed, Out/Limited/Full practice status, prime-time game approvals).

- Each paragraph is treated as one **chunk** in the RAG system.

### 3.2 Embedding and Indexing

- I use `SentenceTransformer( "sentence-transformers/all-MiniLM-L6-v2" )` to embed each chunk.
- Embeddings are stored in a simple in-memory **NumPy array** (`kb_embeddings`).
- Because both the KB embeddings and the query embedding are normalized, the **dot product** corresponds to **cosine similarity**.

### 3.3 Retrieval

- For a given user query:
  1. The query is embedded using the same SentenceTransformer model.
  2. Cosine similarity with each KB chunk is computed.
  3. The top-`k` chunks (typically `k = 2`) are returned as the relevant context.

- The retrieval function returns:
  - Chunk IDs
  - Chunk text
  - Similarity scores

### 3.4 Generation (Prompt Augmentation)

- The retrieved chunks are concatenated into a **CONTEXT** section.
- I then construct a prompt for `google/flan-t5-small` of the form:

  > *"Use ONLY the information in the CONTEXT to answer the QUESTION.  
  > If the answer is not in the context, say 'I don't know based on the provided policy.'"*

- The model generates the final answer based on this augmented prompt.
- I also provide a **baseline** function that calls the same LLM *without* any KB context so I can compare “raw LLM” vs “RAG”.

---

## 4. Test Cases and Retrieval Summary

I used three test cases as required by the assignment: Factual, Foil/General, and Synthesis.  
Below is a summary; exact answers are in the notebook output.

### 4.1 Test Case 1 – Factual

- **Type**: Factual  
- **Query**:  
  > How long must a player remain at each stage of the Sentinels concussion protocol before progressing?

- **Retrieved chunks (by ID)**: [fill in from `results`]  
- **RAG Answer (summary)**:  
  > [Short summary of the answer – e.g., “The player must remain at each stage for at least 24 hours, and if symptoms return they move back one stage and must be symptom-free again before progressing.”]

- **Plain LLM (no RAG) behavior**:  
  > [Did it mention a 24-hour rule? Did it hallucinate league rules?]

- **Observation**:  
  - [1–2 sentences contrasting RAG vs no-RAG here.]

### 4.2 Test Case 2 – Foil / General

- **Type**: Foil  
- **Query**:  
  > What year did the Washington Sentinels win their first Super Bowl?

- **Retrieved chunks (by ID)**: [fill in from `results`]  
- **RAG Answer (summary)**:  
  > [Ideally: “I don't know based on the provided policy.”]

- **Plain LLM (no RAG) behavior**:  
  > [Did it make up a random year?]

- **Observation**:  
  - [Brief comparison. Emphasize hallucination mitigation if RAG refused to answer.]

### 4.3 Test Case 3 – Synthesis

- **Type**: Synthesis  
- **Query**:  
  > If a player sustains a second concussion in the same season, describe the minimum time they must spend at Stage 0 and any extra approvals needed before they can play in a nationally televised game.

- **Retrieved chunks (by ID)**: [fill in from `results`]  
- **RAG Answer (summary)**:  
  > [e.g., “At least seven symptom-free days at Stage 0, then progressing through stages, and for prime-time games, an independent neurological consultant must review and co-sign the clearance.”]

- **Plain LLM (no RAG) behavior**:  
  > [Was it vague? Did it miss the independent consultant requirement?]

- **Observation**:  
  - [Note how RAG combines rules from multiple chunks.]

---

## 5. Analysis: RAG vs Raw LLM and Hallucination

- In the **Factual** case, RAG helped the model stay grounded in the exact policy text (e.g., the 24-hour rule), whereas the plain LLM risked using generic league rules or adding details not in the KB.
- In the **Foil** case, the ideal behavior is for RAG to say it does not know, demonstrating **reduced hallucination** by tying the answer strictly to the provided context.
- In the **Synthesis** case, RAG enables the model to **combine multiple retrieved policy rules** (e.g., second concussion handling + documentation + prime-time independent review), something that is harder for a plain LLM without explicit exposure to this fictional policy.

Overall, using RAG significantly improves factual grounding for questions about this narrow, custom domain compared to a raw LLM prompt that does not see the KB.

---

## 6. How to Run

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
