# ============================================================
# RAG ENGINE
# ============================================================
#
# Retrieval pipeline:
#
# User Question
#       ↓
# Vector Search
#       +
# BM25 Search
#       ↓
# Reciprocal Rank Fusion (RRF)
#       ↓
# Cross-Encoder Reranking
#       ↓
# Relevant Chunk Selection
#       ↓
# Neighbor Expansion
#       ↓
# Context Formatting
#       ↓
# Gemini LLM
#       ↓
# Answer + Timestamp Sources
#
# ============================================================


# ============================================================
# 1. ENVIRONMENT VARIABLES
# ============================================================

from dotenv import load_dotenv

load_dotenv(override=True)


# ============================================================
# 2. STANDARD LIBRARY IMPORTS
# ============================================================

import os


# ============================================================
# 3. LANGCHAIN IMPORTS
# ============================================================

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableLambda
)

from langchain_core.documents import Document


# ============================================================
# 4. CROSS-ENCODER IMPORT
# ============================================================

from sentence_transformers import CrossEncoder


# ============================================================
# 5. VECTOR STORE IMPORTS
# ============================================================

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever,
    create_timestamped_chunks
)


# ============================================================
# 6. BM25 IMPORTS
# ============================================================

from core.bm25_retriever import (
    build_bm25_index,
    search_bm25
)


# ============================================================
# 7. RETRIEVAL CONFIGURATION
# ============================================================

# Number of documents retrieved independently by
# Vector Search and BM25 before fusion.
#
# We use 10 instead of 4 because the Cross-Encoder
# needs enough candidate chunks to compare and rerank.
VECTOR_SEARCH_K = 10
BM25_SEARCH_K = 10


# Maximum number of chunks selected after
# Cross-Encoder reranking.
MAX_RELEVANT_CHUNKS = 5


# Cross-Encoder relevance threshold.
#
# From our testing:
#
# Positive scores → generally relevant
# Negative scores → generally less relevant
#
# We are keeping the tested baseline of 0.0.
CROSS_ENCODER_THRESHOLD = 0.0


# Number of neighboring chunks to add around
# the selected relevant chunks.
#
# window=1 means:
#
# selected chunk
#       ↓
# previous chunk + selected chunk + next chunk
#
NEIGHBOR_WINDOW = 1


# Cross-Encoder model used for reranking.
CROSS_ENCODER_MODEL = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# 8. LOAD THE LLM
# ============================================================

def get_llm():
    """
    Create and return the Gemini LLM.

    The Gemini API key is read from the .env file.
    """

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
    )


# ============================================================
# 9. FORMAT DOCUMENTS FOR THE LLM
# ============================================================

def format_docs(docs):
    """
    Convert retrieved Document objects into one text string.

    Each document contains:

        - transcript text
        - start timestamp
        - end timestamp

    The timestamps are included in the context so that
    the answer can later be associated with the relevant
    part of the video.
    """

    formatted_docs = []

    for doc in docs:

        start = doc.metadata["start_time"]
        end = doc.metadata["end_time"]

        formatted_docs.append(
            f"[Timestamp: {start:.2f}s - {end:.2f}s]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted_docs)


# ============================================================
# 10. RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    vector_docs,
    bm25_results,
    rrf_k=60
):
    """
    Combine Vector Search and BM25 rankings.

    Vector Search understands semantic meaning.

    BM25 performs lexical keyword matching.

    RRF combines their rankings without requiring
    their scores to be on the same scale.

    RRF formula:

        score = 1 / (k + rank)

    If a document appears in both retrieval systems,
    it receives a score from both systems.
    Therefore, documents appearing in both rankings
    move higher in the fused ranking.
    """

    scores = {}

    chunks = {}


    # --------------------------------------------------------
    # Add Vector Search rankings
    # --------------------------------------------------------

    for rank, doc in enumerate(
        vector_docs,
        start=1
    ):

        chunk_index = doc.metadata["chunk_index"]

        scores[chunk_index] = (
            scores.get(chunk_index, 0)
            + 1 / (rrf_k + rank)
        )

        chunks[chunk_index] = doc


    # --------------------------------------------------------
    # Add BM25 rankings
    # --------------------------------------------------------

    for rank, result in enumerate(
        bm25_results,
        start=1
    ):

        chunk_index = result["chunk_index"]

        scores[chunk_index] = (
            scores.get(chunk_index, 0)
            + 1 / (rrf_k + rank)
        )


        # If this chunk was not already returned by
        # Vector Search, create a LangChain Document
        # from the BM25 result.
        if chunk_index not in chunks:

            chunks[chunk_index] = Document(
                page_content=result["text"],
                metadata={
                    "chunk_index": chunk_index,
                    "start_time": result["start_time"],
                    "end_time": result["end_time"]
                }
            )


    # --------------------------------------------------------
    # Sort chunks according to their RRF score
    # --------------------------------------------------------

    ranked_chunk_indices = sorted(
        scores,
        key=scores.get,
        reverse=True
    )


    # --------------------------------------------------------
    # Return documents in fused ranking order
    # --------------------------------------------------------

    return [
        chunks[chunk_index]
        for chunk_index in ranked_chunk_indices
    ]


# ============================================================
# 11. CROSS-ENCODER RERANKING
# ============================================================

def rerank_with_cross_encoder(
    question,
    docs,
    reranker
):
    """
    Rerank candidate documents using a Cross-Encoder.

    Unlike embedding-based retrieval, the Cross-Encoder
    receives the question and document together:

        [question, document]

    It then calculates a relevance score for that pair.

    This allows the model to perform deeper query-document
    interaction than ordinary vector similarity.

    Returns:

        List of (score, document) tuples
        sorted from highest score to lowest score.
    """

    if not docs:
        return []


    # --------------------------------------------------------
    # Create question-document pairs
    # --------------------------------------------------------

    pairs = [
        [question, doc.page_content]
        for doc in docs
    ]


    # --------------------------------------------------------
    # Get Cross-Encoder relevance scores
    # --------------------------------------------------------

    scores = reranker.predict(pairs)


    # --------------------------------------------------------
    # Combine scores with their documents
    # --------------------------------------------------------

    scored_docs = list(
        zip(scores, docs)
    )


    # --------------------------------------------------------
    # Sort from highest relevance to lowest relevance
    # --------------------------------------------------------

    scored_docs.sort(
        key=lambda x: x[0],
        reverse=True
    )


    return scored_docs


# ============================================================
# 12. SELECT RELEVANT CHUNKS
# ============================================================

def select_relevant_chunks(
    scored_docs,
    max_k=MAX_RELEVANT_CHUNKS,
    threshold=CROSS_ENCODER_THRESHOLD
):
    """
    Select the most relevant chunks according to
    Cross-Encoder scores.

    Two conditions are applied:

    1. The Cross-Encoder score must be >= threshold.

    2. At most max_k chunks are selected.

    If no chunk passes the threshold, an empty list
    is returned.

    This is important because we do not want to send
    obviously unrelated transcript chunks to the LLM.
    """

    relevant_docs = [
        doc
        for score, doc in scored_docs
        if score >= threshold
    ]


    # --------------------------------------------------------
    # If nothing is relevant, return no documents.
    # --------------------------------------------------------

    if not relevant_docs:
        return []


    # --------------------------------------------------------
    # Limit the number of highly relevant chunks.
    # --------------------------------------------------------

    return relevant_docs[:max_k]


# ============================================================
# 13. NEIGHBOR EXPANSION
# ============================================================

def expand_with_neighbors(
    reranked_docs,
    fused_docs,
    window=NEIGHBOR_WINDOW
):
    """
    Add neighboring transcript chunks around the
    Cross-Encoder-selected chunks.

    Example:

        Chunk 5 selected

        window = 1

        → Chunk 4
        → Chunk 5
        → Chunk 6

    Why?

    A relevant answer can sometimes continue into
    the previous or next chunk.

    This is especially useful because our transcript
    is divided according to character count rather
    than semantic boundaries.
    """

    if not reranked_docs:
        return []


    # --------------------------------------------------------
    # Create lookup:
    #
    # chunk_index → Document
    # --------------------------------------------------------

    all_docs_by_index = {
        doc.metadata["chunk_index"]: doc
        for doc in fused_docs
    }


    # --------------------------------------------------------
    # Get the chunk indexes selected by Cross-Encoder
    # --------------------------------------------------------

    selected_indices = {
        doc.metadata["chunk_index"]
        for doc in reranked_docs
    }


    # Start with the selected chunks.
    expanded_indices = set(
        selected_indices
    )


    # --------------------------------------------------------
    # Add neighboring chunks
    # --------------------------------------------------------

    for idx in selected_indices:

        for offset in range(
            1,
            window + 1
        ):

            # Previous chunk
            expanded_indices.add(
                idx - offset
            )

            # Next chunk
            expanded_indices.add(
                idx + offset
            )


    # --------------------------------------------------------
    # Convert indexes back into Documents
    #
    # Only chunks that actually exist are included.
    # --------------------------------------------------------

    expanded_docs = [
        all_docs_by_index[i]
        for i in sorted(expanded_indices)
        if i in all_docs_by_index
    ]


    return expanded_docs


# ============================================================
# 14. BUILD RAG CHAIN
# ============================================================

def build_rag_chain(
    transcript: list,
    video_id: str
):
    """
    Build the complete production RAG pipeline.

    This function is called after a video has been
    processed and transcribed.

    It prepares:

        1. Vector Store
        2. Timestamped chunks
        3. BM25 index
        4. Vector retriever
        5. Cross-Encoder
        6. Gemini LLM

    It then returns a function called
    retrieve_and_answer() which is used for every question.
    """


    # ========================================================
    # STEP 1: BUILD / LOAD VECTOR STORE
    # ========================================================

    print("\nBuilding vector store...")

    vector_store = build_vector_store(
        transcript,
        video_id
    )


    # ========================================================
    # STEP 2: CREATE TIMESTAMPED SEARCH CHUNKS
    # ========================================================

    print("Creating timestamped chunks...")

    chunks = create_timestamped_chunks(
        transcript
    )

    print(
        f"Created {len(chunks)} search chunks."
    )


    # ========================================================
    # STEP 3: BUILD BM25 INDEX
    # ========================================================

    print("\nBuilding BM25 index...")

    bm25 = build_bm25_index(
        chunks
    )

    print("BM25 index ready.")


    # ========================================================
    # STEP 4: CREATE VECTOR RETRIEVER
    # ========================================================

    print("\nCreating vector retriever...")

    retriever = get_retriever(
        vector_store,
        video_id,
        k=VECTOR_SEARCH_K
    )

    print(
        f"Vector retriever ready. k={VECTOR_SEARCH_K}"
    )


    # ========================================================
    # STEP 5: LOAD CROSS-ENCODER
    # ========================================================

    print("\nLoading Cross-Encoder...")

    reranker = CrossEncoder(
        CROSS_ENCODER_MODEL
    )

    print(
        "Cross-Encoder ready."
    )


    # ========================================================
    # STEP 6: CREATE LLM
    # ========================================================

    print("\nCreating Gemini LLM...")

    llm = get_llm()

    print("Gemini LLM ready.")


    # ========================================================
    # STEP 7: CREATE PROMPT
    # ========================================================

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",

            """You are an expert meeting assistant.

Answer the user's question based ONLY on the meeting
transcript context provided below.

If the answer is not found in the context, say:

"I could not find this information in the meeting transcript."

Do not use outside knowledge.

Always be concise and precise.

The transcript context contains timestamps.
Use the provided context to answer the question.

Context from meeting transcript:
{context}"""
        ),

        (
            "human",
            "{question}"
        ),
    ])


    # ========================================================
    # STEP 8: QUESTION → RETRIEVAL → LLM
    # ========================================================

    def retrieve_and_answer(question):
        """
        Execute the complete retrieval pipeline for one
        user question.

        Pipeline:

            Vector Search
                    +
            BM25 Search
                    ↓
            RRF
                    ↓
            Cross-Encoder
                    ↓
            Relevant Chunk Selection
                    ↓
            Neighbor Expansion
                    ↓
            Gemini
        """


        print("\n")
        print("=" * 70)
        print("QUESTION")
        print("=" * 70)

        print(question)


        # ====================================================
        # STEP 8.1: VECTOR SEARCH
        # ====================================================

        print("\nRunning Vector Search...")

        vector_docs = retriever.invoke(
            question
        )

        print(
            f"Vector results: {len(vector_docs)}"
        )


        # ====================================================
        # STEP 8.2: BM25 SEARCH
        # ====================================================

        print("\nRunning BM25 Search...")

        bm25_results = search_bm25(
            bm25,
            chunks,
            question,
            k=BM25_SEARCH_K
        )

        print(
            f"BM25 results: {len(bm25_results)}"
        )


        # ====================================================
        # STEP 8.3: RECIPROCAL RANK FUSION
        # ====================================================

        print(
            "\nApplying Reciprocal Rank Fusion..."
        )

        fused_docs = reciprocal_rank_fusion(
            vector_docs,
            bm25_results
        )

        print(
            f"Fused results: {len(fused_docs)}"
        )


        # ====================================================
        # STEP 8.4: CROSS-ENCODER RERANKING
        # ====================================================

        print(
            "\nRunning Cross-Encoder reranking..."
        )

        scored_docs = rerank_with_cross_encoder(
            question,
            fused_docs,
            reranker
        )


        # ----------------------------------------------------
        # Display Cross-Encoder scores
        # ----------------------------------------------------

        print("\nCROSS-ENCODER SCORES")
        print("-" * 70)

        for score, doc in scored_docs:

            print(
                f"Score: {score:.4f} | "
                f"Chunk: "
                f"{doc.metadata['chunk_index']} | "
                f"Timestamp: "
                f"{doc.metadata['start_time']:.2f}s - "
                f"{doc.metadata['end_time']:.2f}s"
            )


        # ====================================================
        # STEP 8.5: SELECT RELEVANT CHUNKS
        # ====================================================

        reranked_docs = select_relevant_chunks(
            scored_docs,
            max_k=MAX_RELEVANT_CHUNKS,
            threshold=CROSS_ENCODER_THRESHOLD
        )

        print(
            f"\nCross-Encoder selected: "
            f"{len(reranked_docs)}"
        )


        # ====================================================
        # STEP 8.6: HANDLE NO RELEVANT RESULTS
        # ====================================================

        if not reranked_docs:

            print(
                "\nNo relevant transcript chunks "
                "passed the Cross-Encoder threshold."
            )

            return {
                "answer": (
                    "I could not find this information "
                    "in the meeting transcript."
                ),
                "sources": []
            }


        # ====================================================
        # STEP 8.7: NEIGHBOR EXPANSION
        # ====================================================

        final_docs = expand_with_neighbors(
            reranked_docs,
            fused_docs,
            window=NEIGHBOR_WINDOW
        )

        print(
            f"Final results after neighbor expansion: "
            f"{len(final_docs)}"
        )


        # ====================================================
        # STEP 8.8: DISPLAY FINAL RETRIEVAL RESULTS
        # ====================================================

        print("\nFINAL RETRIEVAL RESULTS")
        print("=" * 70)

        for rank, doc in enumerate(
            final_docs,
            start=1
        ):

            print(
                f"\nRank {rank}"
            )

            print(
                f"Chunk: "
                f"{doc.metadata['chunk_index']}"
            )

            print(
                f"Timestamp: "
                f"{doc.metadata['start_time']:.2f}s - "
                f"{doc.metadata['end_time']:.2f}s"
            )

            print(
                f"Text: "
                f"{doc.page_content}"
            )


        # ====================================================
        # STEP 8.9: FORMAT CONTEXT FOR GEMINI
        # ====================================================

        context = format_docs(
            final_docs
        )


        # ====================================================
        # STEP 8.10: SEND CONTEXT + QUESTION TO GEMINI
        # ====================================================

        print(
            "\nSending retrieved context to Gemini..."
        )

        response = llm.invoke(
            prompt.format_messages(
                context=context,
                question=question
            )
        )


        # ====================================================
        # STEP 8.11: EXTRACT ANSWER
        # ====================================================

        answer = response.content


        print("\nANSWER")
        print("=" * 70)

        print(answer)


        # ====================================================
        # STEP 8.12: CREATE TIMESTAMP SOURCES
        # ====================================================

        sources = []

        for doc in final_docs:

            sources.append({
                "start_time": doc.metadata["start_time"],
                "end_time": doc.metadata["end_time"]
            })


        # ====================================================
        # STEP 8.13: RETURN RESULT TO FASTAPI
        # ====================================================

        return {
            "answer": answer,
            "sources": sources
        }


    # ========================================================
    # RETURN THE QUESTION-ANSWERING FUNCTION
    # ========================================================

    return retrieve_and_answer


# ============================================================
# 15. LOAD EXISTING RAG CHAIN
# ============================================================

def load_rag_chain(video_id: str):
    """
    Load an existing Vector Store for a video.

    NOTE:
    The current production analysis flow uses
    build_rag_chain() because it has access to the
    transcript and therefore can build the BM25 index.

    This function is kept for compatibility with the
    existing project architecture.
    """

    # --------------------------------------------------------
    # Load existing ChromaDB
    # --------------------------------------------------------

    vector_store = load_vector_store()


    # --------------------------------------------------------
    # Create Vector Retriever
    # --------------------------------------------------------

    retriever = get_retriever(
        vector_store,
        video_id,
        k=VECTOR_SEARCH_K
    )


    # --------------------------------------------------------
    # Create LLM
    # --------------------------------------------------------

    llm = get_llm()


    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",

            """You are an expert meeting assistant.

Answer the user's question based ONLY on the meeting
transcript context provided below.

If the answer is not found in the context, say:

"I could not find this information in the meeting transcript."

Always be concise and precise.

Context from meeting transcript:
{context}"""
        ),

        (
            "human",
            "{question}"
        ),
    ])


    # --------------------------------------------------------
    # Existing LCEL Vector RAG pipeline
    # --------------------------------------------------------

    rag_chain = (
        {
            "context": (
                retriever
                | RunnableLambda(format_docs)
            ),

            "question": RunnablePassthrough(),
        }

        | prompt
        | llm
        | StrOutputParser()
    )


    return rag_chain


# ============================================================
# 16. ASK QUESTION HELPER
# ============================================================

def ask_question(
    rag_chain,
    question: str
):
    """
    Send a question to the RAG chain.

    This helper is retained for compatibility with
    the existing project.
    """

    print(
        f"Question: {question}"
    )

    result = rag_chain(
        question
    )

    print(
        f"Answer: {result['answer']}"
    )

    return result