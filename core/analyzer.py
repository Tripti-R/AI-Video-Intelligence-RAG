# ============================================================
# MEETING ANALYZER
# ============================================================
#
# This module performs the analysis of the complete transcript.
#
# Instead of making separate Gemini calls for:
#
#   - title
#   - summary
#   - action items
#   - key decisions
#   - open questions
#
# we ask Gemini for ALL five fields in one structured response.
#
# For short transcripts:
#
#       Transcript
#           ↓
#       ONE Gemini call
#           ↓
#       Complete MeetingAnalysis
#
#
# For long transcripts:
#
#       Transcript
#           ↓
#       Split into chunks
#           ↓
#       ONE combined extraction call per chunk
#           ↓
#       ONE final merge call
#           ↓
#       Complete MeetingAnalysis
#
# ============================================================


# ============================================================
# 1. ENVIRONMENT
# ============================================================

from dotenv import load_dotenv

load_dotenv(override=True)


# ============================================================
# 2. STANDARD LIBRARY IMPORTS
# ============================================================

import os
import time
from typing import List


# ============================================================
# 3. LANGCHAIN IMPORTS
# ============================================================

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# 4. PYDANTIC IMPORTS
# ============================================================

from pydantic import BaseModel, Field


# ============================================================
# 5. EXISTING TRANSCRIPT CHUNKER
# ============================================================

from core.summarizer import split_transcript


# ============================================================
# 6. LLM CONFIGURATION
# ============================================================

def get_llm():
    """
    Create and return the Gemini model.

    The API key is loaded from the .env file.
    """

    return ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.3,
    )


# ============================================================
# 7. FINAL OUTPUT SCHEMA
# ============================================================

class MeetingAnalysis(BaseModel):
    """
    Structured representation of the complete meeting analysis.

    These fields intentionally match the fields that the
    existing frontend already expects.
    """

    title: str = Field(
        description=(
            "A short professional meeting title, "
            "maximum 8 words."
        )
    )

    summary: str = Field(
        description=(
            "A professional meeting summary "
            "written in bullet points."
        )
    )

    action_items: str = Field(
        description=(
            "Numbered list of action items. "
            "Each item must contain Task, Owner "
            "(or 'Not specified'), and Deadline "
            "(or 'Not specified'). "
            "If there are no action items, write exactly: "
            "'No action items found.'"
        )
    )

    key_decisions: str = Field(
        description=(
            "Numbered list of key decisions made "
            "during the meeting. "
            "If there are no key decisions, write exactly: "
            "'No key decisions found.'"
        )
    )

    open_questions: str = Field(
        description=(
            "Numbered list of unresolved questions "
            "or follow-up topics. "
            "If there are no open questions, write exactly: "
            "'No open questions found.'"
        )
    )


# ============================================================
# 8. SINGLE-PASS CHARACTER LIMIT
# ============================================================

# Short transcripts are processed in ONE Gemini call.
#
# We use the 6000-character threshold discussed earlier.
#
# If the transcript is longer than this, we use the
# chunked analysis path below.

SINGLE_PASS_CHAR_LIMIT = 6000


# ============================================================
# 9. SINGLE-PASS SYSTEM PROMPT
# ============================================================

ANALYSIS_SYSTEM_PROMPT = """
You are an expert meeting assistant.

Read the meeting transcript below and produce a complete
structured analysis based ONLY on the transcript.

You must fill all five fields:

1. TITLE

Create a short, professional meeting title.
Maximum 8 words.

2. SUMMARY

Create a professional summary of the meeting
using bullet points.

3. ACTION ITEMS

Extract ALL action items mentioned in the transcript.

For every action item include:

- Task
- Owner, if known
- Deadline, if mentioned

If the owner is not mentioned, write:

"Not specified"

If the deadline is not mentioned, write:

"Not specified"

If there are no action items, write exactly:

"No action items found."

4. KEY DECISIONS

Extract the important decisions that were actually
made during the meeting.

Do not invent decisions.

If there are no key decisions, write exactly:

"No key decisions found."

5. OPEN QUESTIONS

Extract unresolved questions, pending issues,
or topics requiring follow-up.

If there are no open questions, write exactly:

"No open questions found."

Use ONLY information contained in the transcript.
Do not use outside knowledge.

Fill every field completely.
"""


# ============================================================
# 10. SINGLE-PASS ANALYSIS
# ============================================================

def _single_pass_analysis(
    transcript: str
) -> MeetingAnalysis:
    """
    Analyze a short transcript using ONE Gemini call.

    with_structured_output() makes Gemini return data
    matching the MeetingAnalysis Pydantic schema.
    """

    # --------------------------------------------------------
    # Create Gemini model with structured output.
    # --------------------------------------------------------

    llm = get_llm().with_structured_output(
        MeetingAnalysis
    )


    # --------------------------------------------------------
    # Create prompt.
    # --------------------------------------------------------

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            ANALYSIS_SYSTEM_PROMPT
        ),
        (
            "human",
            "{text}"
        ),
    ])


    # --------------------------------------------------------
    # Connect prompt → structured Gemini model.
    # --------------------------------------------------------

    chain = prompt | llm


    # --------------------------------------------------------
    # Make ONE Gemini call.
    # --------------------------------------------------------

    result = chain.invoke({
        "text": transcript
    })


    # --------------------------------------------------------
    # Small delay to avoid immediately sending another
    # request if this function is followed by another call.
    # --------------------------------------------------------

    time.sleep(1.2)


    return result


# ============================================================
# 11. CHUNK ANALYSIS SCHEMA
# ============================================================

class ChunkAnalysis(BaseModel):
    """
    Represents the information extracted from ONE portion
    of a long transcript.
    """

    partial_summary: str = Field(
        description=(
            "Concise summary of this transcript portion."
        )
    )

    action_items: str = Field(
        description=(
            "Action items found in THIS portion only, "
            "as a numbered list, or 'None'."
        )
    )

    key_decisions: str = Field(
        description=(
            "Key decisions found in THIS portion only, "
            "as a numbered list, or 'None'."
        )
    )

    open_questions: str = Field(
        description=(
            "Open questions found in THIS portion only, "
            "as a numbered list, or 'None'."
        )
    )


# ============================================================
# 12. CHUNK ANALYSIS PROMPT
# ============================================================

CHUNK_SYSTEM_PROMPT = """
You are an expert meeting analyst reviewing ONE portion
of a longer meeting transcript.

Analyze ONLY this transcript portion.

Extract:

1. partial_summary

Give a concise summary of what happened in this portion.

2. action_items

Extract action items from this portion.

For each action item include:

- Task
- Owner, if known
- Deadline, if known

If there are none, write:

"None"

3. key_decisions

Extract decisions actually made in this portion.

If there are none, write:

"None"

4. open_questions

Extract unresolved questions or follow-up topics
from this portion.

If there are none, write:

"None"

Do not use outside knowledge.
Do not invent information.
"""


# ============================================================
# 13. FINAL MERGE PROMPT
# ============================================================

MERGE_SYSTEM_PROMPT = """
You are an expert meeting assistant.

You are given partial analyses from consecutive portions
of the SAME meeting.

Combine them into ONE final structured analysis.

Produce:

1. title

Create a short professional meeting title.
Maximum 8 words.

2. summary

Create one cohesive professional summary using
bullet points.

Merge related information from the partial summaries.

3. action_items

Create ONE deduplicated numbered list containing
all action items across the meeting.

Each action item should contain:

- Task
- Owner
- Deadline

If an owner is unavailable, use:

"Not specified"

If a deadline is unavailable, use:

"Not specified"

If there are no action items anywhere, write:

"No action items found."

4. key_decisions

Create ONE deduplicated numbered list containing
the important decisions made during the meeting.

If there are no decisions, write:

"No key decisions found."

5. open_questions

Create ONE deduplicated numbered list containing
all unresolved questions or follow-up topics.

If there are none, write:

"No open questions found."

Do not invent information.

Use only the information contained in the
partial analyses.
"""


# ============================================================
# 14. LONG-TRANSCRIPT ANALYSIS
# ============================================================

def _chunked_analysis(
    transcript: str
) -> MeetingAnalysis:
    """
    Analyze a long transcript.

    Each transcript chunk receives ONE combined
    extraction call.

    After all chunks are processed, ONE final Gemini
    call merges the partial results.
    """

    # --------------------------------------------------------
    # Split the transcript using the existing chunker.
    # --------------------------------------------------------

    chunks = split_transcript(
        transcript
    )

    print(
        f"Long transcript detected. "
        f"Created {len(chunks)} analysis chunks."
    )


    # ========================================================
    # STEP 1: ANALYZE EACH CHUNK
    # ========================================================

    chunk_llm = get_llm().with_structured_output(
        ChunkAnalysis
    )


    chunk_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            CHUNK_SYSTEM_PROMPT
        ),
        (
            "human",
            "{text}"
        ),
    ])


    chunk_chain = (
        chunk_prompt
        | chunk_llm
    )


    partials: List[ChunkAnalysis] = []


    for i, chunk in enumerate(
        chunks,
        start=1
    ):

        print(
            f"Analyzing transcript portion "
            f"{i}/{len(chunks)}..."
        )


        # One Gemini call for this chunk.
        result = chunk_chain.invoke({
            "text": chunk
        })


        partials.append(
            result
        )


        # Small delay between API calls.
        time.sleep(1.2)


    # ========================================================
    # STEP 2: PREPARE PARTIAL RESULTS FOR MERGING
    # ========================================================

    merged_input = "\n\n".join(
        (
            f"--- Portion {i + 1} ---\n"
            f"Summary: {partial.partial_summary}\n"
            f"Action items: {partial.action_items}\n"
            f"Decisions: {partial.key_decisions}\n"
            f"Questions: {partial.open_questions}"
        )
        for i, partial in enumerate(partials)
    )


    # ========================================================
    # STEP 3: FINAL MERGE
    # ========================================================

    print(
        "Merging transcript analysis..."
    )


    merge_llm = get_llm().with_structured_output(
        MeetingAnalysis
    )


    merge_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            MERGE_SYSTEM_PROMPT
        ),
        (
            "human",
            "{text}"
        ),
    ])


    merge_chain = (
        merge_prompt
        | merge_llm
    )


    # One final Gemini call.
    result = merge_chain.invoke({
        "text": merged_input
    })


    time.sleep(1.2)


    return result


# ============================================================
# 15. PUBLIC ANALYSIS FUNCTION
# ============================================================

def analyze_transcript(
    transcript: str
) -> MeetingAnalysis:
    """
    Main entry point used by main.py.

    Short transcript:
        → ONE Gemini call

    Long transcript:
        → ONE call per chunk
        → ONE final merge call
    """

    if len(transcript) <= SINGLE_PASS_CHAR_LIMIT:

        print(
            "Transcript is short enough "
            "for single-pass analysis."
        )

        return _single_pass_analysis(
            transcript
        )


    print(
        "Transcript exceeds single-pass limit."
    )

    return _chunked_analysis(
        transcript
    )