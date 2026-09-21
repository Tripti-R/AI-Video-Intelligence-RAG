import os
import whisper


FFMPEG_DIR = r"C:\Users\Tripti\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin"

os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ.get("PATH", "")


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


_model = None


def load_model():

    global _model

    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")

        _model = whisper.load_model(WHISPER_MODEL)

        print("Whisper model loaded.")

    return _model


def transcribe_chunk_whisper(chunk_path: str):

    model = load_model()

    result = model.transcribe(
        chunk_path,
        task="transcribe"
    )

    return result["segments"]


def transcribe_chunk(chunk_path: str):

    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list) -> list:

    full_transcript = []

    print("Using Whisper for transcription.")

    for i, chunk in enumerate(chunks):

        print(
            f"Transcribing chunk {i + 1}/{len(chunks)}..."
        )

        chunk_path = chunk["path"]

        chunk_start_time = chunk["start_time"]

        segments = transcribe_chunk(chunk_path)

        for segment in segments:

            full_transcript.append({
                "text": segment["text"].strip(),

                "start": (
                    chunk_start_time
                    + segment["start"]
                ),

                "end": (
                    chunk_start_time
                    + segment["end"]
                )
            })

    print("Transcription complete.")

    return full_transcript

def format_transcript(transcript: list) -> str:
    return "\n".join(
        segment["text"]
        for segment in transcript
    )