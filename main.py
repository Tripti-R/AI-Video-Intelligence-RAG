from dotenv import load_dotenv

load_dotenv(override=True)
import hashlib
import os
from urllib.parse import urlparse, parse_qs
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


from core.transcriber import transcribe_all, format_transcript
from core.analyzer import analyze_transcript
from core.rag_engine import build_rag_chain, ask_question
from utils.audio_processor import process_input


app = FastAPI(title="AI Meeting Assistant API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    source: str


class QuestionRequest(BaseModel):
    question: str

def get_video_id(source: str) -> str:

    # YouTube URL
    if "youtube.com/watch" in source:
        query = parse_qs(urlparse(source).query)
        return query.get("v", [source])[0]

    # Short YouTube URL
    if "youtu.be/" in source:
        return source.split("youtu.be/")[1].split("?")[0]

    # Local file
    with open(source, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    return file_hash

rag_chain = None


@app.get("/")
def home():
    return {"message": "AI Meeting Assistant API is running"}


@app.post("/analyze")
def analyze_meeting(request: AnalyzeRequest):
    global rag_chain

    try:
        # ----------------------------------------------------
        # 1. Identify the video
        # ----------------------------------------------------

        video_id = get_video_id(
            request.source
        )


        # ----------------------------------------------------
        # 2. Download / convert / chunk audio
        # ----------------------------------------------------

        chunks = process_input(
            request.source
        )


        # ----------------------------------------------------
        # 3. Transcribe the complete video
        # ----------------------------------------------------

        transcript = transcribe_all(
            chunks
        )


        # ----------------------------------------------------
        # 4. Convert transcript into plain text
        # ----------------------------------------------------

        transcript_text = format_transcript(
            transcript
        )


        # ----------------------------------------------------
        # 5. Analyze transcript
        #
        # Short transcript:
        #     ONE Gemini call
        #
        # Long transcript:
        #     chunk calls + ONE merge call
        # ----------------------------------------------------

        analysis = analyze_transcript(
            transcript_text
        )


        # ----------------------------------------------------
        # 6. Build the RAG pipeline
        #
        # This does NOT call Gemini.
        #
        # It prepares:
        #
        # Vector Search
        # BM25
        # RRF
        # Cross-Encoder
        # ----------------------------------------------------

        rag_chain = build_rag_chain(
            transcript,
            video_id
        )


        # ----------------------------------------------------
        # 7. Return response to frontend
        # ----------------------------------------------------

        return {
            "title": analysis.title,
            "transcript": transcript,
            "summary": analysis.summary,
            "action_items": analysis.action_items,
            "key_decisions": analysis.key_decisions,
            "open_questions": analysis.open_questions,
        }


    except Exception as e:

        print(
            "ANALYZE ERROR:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.post("/ask")
def ask_meeting_question(request: QuestionRequest):

    if rag_chain is None:
        raise HTTPException(
            status_code=400,
            detail="Please analyze a meeting before asking questions."
        )

    try:
        result = ask_question(
            rag_chain,
            request.question
        )

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )