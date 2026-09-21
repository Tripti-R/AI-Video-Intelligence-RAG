import yt_dlp
from pydub import AudioSegment
import os

# ---------------------------------------------------------
# FFmpeg configuration
# ---------------------------------------------------------

FFMPEG_DIR = r"C:\Users\Tripti\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.1-full_build\bin"
FFMPEG_PATH = os.path.join(FFMPEG_DIR, "ffmpeg.exe")
FFPROBE_PATH = os.path.join(FFMPEG_DIR, "ffprobe.exe")

# Tell pydub exactly where FFmpeg is
AudioSegment.converter = FFMPEG_PATH
AudioSegment.ffprobe = FFPROBE_PATH

# ---------------------------------------------------------
# Download directory
# ---------------------------------------------------------
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------
# Download YouTube audio
# ---------------------------------------------------------
def download_youtube_audio(url: str) -> str:
    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "cookiefile": "cookies.txt",
        # Tell yt-dlp exactly where FFmpeg is
        "ffmpeg_location": FFMPEG_DIR,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(
            url,
            download=True
        )
        filename = ydl.prepare_filename(info)
        filename = (
            filename
            .replace(".webm", ".wav")
            .replace(".m4a", ".wav")
        )
    return filename


# ---------------------------------------------------------
# Convert audio/video to WAV
# ---------------------------------------------------------

def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to WAV format using pydub."""
    output_path = (
        os.path.splitext(input_path)[0]
        + "_converted.wav"
    )
    audio = AudioSegment.from_file(input_path)
    # Whisper works well with mono 16 kHz audio
    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(16000)
    )
    audio.export(
        output_path,
        format="wav"
    )
    return output_path


# ---------------------------------------------------------
# Split audio into chunks
# ---------------------------------------------------------

def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list:

    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    chunks = []
    for i, start in enumerate(
        range(0, len(audio), chunk_ms)
    ):

        chunk = audio[
            start:start + chunk_ms
        ]

        chunk_path = ( f"{wav_path}_chunk_{i}.wav")
        chunk.export( chunk_path,format="wav" )
        chunks.append({
            "path": chunk_path,
            "start_time": start / 1000
        })
    return chunks


# ---------------------------------------------------------
# Main audio processing function
# ---------------------------------------------------------

def process_input(source: str) -> list:

    if ( source.startswith("http://")  or source.startswith("https://")   ):
        print("Detected YouTube URL. Downloading audio..." )
        wav_path = download_youtube_audio(source)

    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)
    print("Chunking audio...")
    chunks = chunk_audio(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks