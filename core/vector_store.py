import os 
from langchain_chroma import Chroma 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )

def create_timestamped_chunks(
    transcript: list,
    chunk_size: int = 500
) -> list:

    chunks = []
    current_text = []
    current_start = None
    current_end = None

    for segment in transcript:
        text = segment["text"]

        if current_start is None:
            current_start = segment["start"]

        current_text.append(text)
        current_end = segment["end"]

        combined_text = " ".join(current_text)

        if len(combined_text) >= chunk_size:
            chunks.append({
                "text": combined_text,
                "start_time": current_start,
                "end_time": current_end
            })

            current_text = []
            current_start = None
            current_end = None

    if current_text:
        chunks.append({
            "text": " ".join(current_text),
            "start_time": current_start,
            "end_time": current_end
        })

    return chunks

def build_vector_store(transcript: list, video_id: str) -> Chroma:
    print("Building vector Store")

    chunks = create_timestamped_chunks(transcript)

    docs = [
        Document(
            page_content=chunk["text"],
            metadata={
                "video_id": video_id,
                "chunk_index": i,
                "start_time": chunk["start_time"],
                "end_time": chunk["end_time"]
            }
        )
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    if video_exists(vector_store, video_id):
        print("Video already exists. Skipping vector store creation.")
        return vector_store

    vector_store.add_documents(docs)

    print(f"Added {len(docs)} chunks to vector store.")

    return vector_store

def load_vector_store() -> Chroma:
    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vector_store


def get_retriever(vector_store: Chroma,video_id:str, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k,
            "filter": {
                "video_id": video_id
            }
            }
    )
    
def video_exists(vector_store: Chroma, video_id: str) -> bool:

    data = vector_store.get(
        where={"video_id": video_id},
        limit=1
    )

    return len(data["ids"]) > 0