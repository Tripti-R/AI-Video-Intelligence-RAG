from dotenv import load_dotenv

load_dotenv(override=True)
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
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
    language: str = "english"


class QuestionRequest(BaseModel):
    question: str


rag_chain = None


@app.get("/")
def home():
    return {"message": "AI Meeting Assistant API is running"}


@app.post("/analyze")
def analyze_meeting(request: AnalyzeRequest):
    global rag_chain

    try:
        chunks = process_input(request.source)

        transcript = transcribe_all(
            chunks,
            request.language
        )

        title = generate_title(transcript)

        summary = summarize(transcript)

        action_items = extract_action_items(transcript)

        decisions = extract_key_decisions(transcript)

        questions = extract_questions(transcript)

        rag_chain = build_rag_chain(transcript)

        return {
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": decisions,
            "open_questions": questions,
        }

    except Exception as e:
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
        answer = ask_question(
            rag_chain,
            request.question
        )

        return {
            "question": request.question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )