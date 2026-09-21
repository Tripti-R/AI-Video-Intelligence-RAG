from utils.audio_processor import process_input

from core.transcriber import transcribe_all

from core.vector_store import (
    create_timestamped_chunks,
    build_vector_store,
    get_retriever
)

from core.bm25_retriever import (
    build_bm25_index,
    search_bm25
)

from core.rag_engine import reciprocal_rank_fusion
from sentence_transformers import CrossEncoder


# ---------------------------------------------------------
# Neighbor Expansion
# ---------------------------------------------------------

def expand_with_neighbors(
    reranked_docs,
    fused_docs,
    window=1
):
    all_docs_by_index = {
        doc.metadata["chunk_index"]: doc
        for doc in fused_docs
    }

    selected_indices = {
        doc.metadata["chunk_index"]
        for doc in reranked_docs
    }

    expanded_indices = set(selected_indices)

    for idx in selected_indices:
        for offset in range(1, window + 1):
            expanded_indices.add(idx - offset)
            expanded_indices.add(idx + offset)

    expanded_docs = [
        all_docs_by_index[i]
        for i in sorted(expanded_indices)
        if i in all_docs_by_index
    ]

    return expanded_docs


# ---------------------------------------------------------
# Select Relevant Chunks
# ---------------------------------------------------------

def select_relevant_chunks(
    scored_docs,
    max_k=5,
    threshold=0.0
):
    # Keep only chunks whose Cross-Encoder
    # score indicates relevance.
    relevant_docs = [
        doc
        for score, doc in scored_docs
        if score >= threshold
    ]

    # If nothing passes the relevance threshold,
    # do NOT force unrelated chunks into the context.
    if not relevant_docs:
        return []

    # Limit the number of highly relevant chunks.
    return relevant_docs[:max_k]


# ---------------------------------------------------------
# 1. Get YouTube URL
# ---------------------------------------------------------

source = input("Enter YouTube URL: ").strip()

print("\nProcessing video...")

chunks = process_input(source)


# ---------------------------------------------------------
# 2. Transcribe video
# ---------------------------------------------------------

print("\nTranscribing video...")

transcript = transcribe_all(chunks)

print("\nTranscription complete.")
print(f"Total transcript segments: {len(transcript)}")


# ---------------------------------------------------------
# 3. Print full transcript
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FULL TRANSCRIPT")
print("=" * 70)

for segment in transcript:
    print(
        f"[{segment['start']:.2f}s - {segment['end']:.2f}s] "
        f"{segment['text']}"
    )


# ---------------------------------------------------------
# 4. Create search chunks
# ---------------------------------------------------------

chunks_for_search = create_timestamped_chunks(transcript)

print(
    f"\nSearch chunks created: "
    f"{len(chunks_for_search)}"
)


# ---------------------------------------------------------
# 5. Build Vector Search
# ---------------------------------------------------------

print("\nBuilding vector store...")

vector_store = build_vector_store(
    transcript,
    source
)

retriever = get_retriever(
    vector_store,
    source,
    k=10
)

print("Vector search ready.")


# ---------------------------------------------------------
# 6. Build BM25
# ---------------------------------------------------------

print("\nBuilding BM25 index...")

bm25 = build_bm25_index(
    chunks_for_search
)

print("BM25 search ready.")


# ---------------------------------------------------------
# Load Cross-Encoder
# ---------------------------------------------------------

print("\nLoading Cross-Encoder...")

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Cross-Encoder ready.")


# ---------------------------------------------------------
# 7. QUESTION LOOP
# ---------------------------------------------------------

while True:

    question = input(
        "\nEnter your question (or type 'exit' to quit): "
    ).strip()

    if question.lower() == "exit":
        print("\nExiting...")
        break


    # -----------------------------------------------------
    # Vector Search
    # -----------------------------------------------------

    print("\nRunning Vector Search...")

    vector_docs = retriever.invoke(question)

    print(
        f"Vector results: {len(vector_docs)}"
    )


    # -----------------------------------------------------
    # BM25 Search
    # -----------------------------------------------------

    print("\nRunning BM25 Search...")

    bm25_results = search_bm25(
        bm25,
        chunks_for_search,
        question,
        k=10
    )

    print(
        f"BM25 results: {len(bm25_results)}"
    )


    # -----------------------------------------------------
    # RRF
    # -----------------------------------------------------

    print("\nApplying Reciprocal Rank Fusion...")

    fused_docs = reciprocal_rank_fusion(
        vector_docs,
        bm25_results
    )

    print(
        f"Fused results: {len(fused_docs)}"
    )


    # -----------------------------------------------------
    # Cross-Encoder Reranking
    # -----------------------------------------------------

    print("\nRunning Cross-Encoder reranking...")

    pairs = [
        [question, doc.page_content]
        for doc in fused_docs
    ]

    scores = reranker.predict(pairs)


    # -----------------------------------------------------
    # Display Cross-Encoder Scores
    # -----------------------------------------------------

    print("\nCROSS-ENCODER SCORES")
    print("=" * 60)

    for score, doc in sorted(
        zip(scores, fused_docs),
        key=lambda x: x[0],
        reverse=True
    ):
        print(
            f"Score: {score:.4f} | "
            f"Chunk: {doc.metadata['chunk_index']} | "
            f"Timestamp: "
            f"{doc.metadata['start_time']:.2f}s - "
            f"{doc.metadata['end_time']:.2f}s"
        )


    # -----------------------------------------------------
    # Score-Based Chunk Selection
    # -----------------------------------------------------

    scored_docs = sorted(
        zip(scores, fused_docs),
        key=lambda x: x[0],
        reverse=True
    )

    reranked_docs = select_relevant_chunks(
        scored_docs,
        max_k=5,
        threshold=0.0
    )

    print(
        f"\nCross-Encoder selected: "
        f"{len(reranked_docs)}"
    )


    # -----------------------------------------------------
    # Neighbor Expansion
    # -----------------------------------------------------

    final_docs = expand_with_neighbors(
        reranked_docs,
        fused_docs,
        window=1
    )

    print(
        f"Final results after neighbor expansion: "
        f"{len(final_docs)}"
    )


    # -----------------------------------------------------
    # Display Final Results
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL RESULTS AFTER NEIGHBOR EXPANSION")
    print("=" * 60)

    for rank, doc in enumerate(
        final_docs,
        start=1
    ):

        print(f"\nRank {rank}")

        print(
            f"Timestamp: "
            f"{doc.metadata['start_time']:.2f}s - "
            f"{doc.metadata['end_time']:.2f}s"
        )

        print(
            f"Chunk index: "
            f"{doc.metadata['chunk_index']}"
        )

        print(
            f"Text: "
            f"{doc.page_content}"
        )


    # -----------------------------------------------------
    # Ask whether to continue
    # -----------------------------------------------------

    choice = input(
        "\nAsk another question? (y/n): "
    ).strip().lower()

    if choice != "y":
        print("\nExiting...")
        break